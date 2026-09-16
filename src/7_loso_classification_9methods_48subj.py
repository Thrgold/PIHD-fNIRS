# -*- coding: utf-8 -*-
"""
Step 7 (9 methods): LOSO跨被试分类验证 + trial-level 5-fold CV
===============================================================
对比9种方法:
  Raw HbO / Bandpass / GLM / ICA / PCA / Wavelet / SSA / EMD / PIHD

基本信息:
  - 方法数: 9种
  - 被试数: 48名 (全部可用被试)
  - Trial数: 480 trials (48 subjects × 10 trials/subject)
  - 特征类型: Task特征 (ME/MI窗口的mean/peak/slope/variance, 非task-rest差值)
  - 分类器: L2-penalized Logistic Regression (C=1.0)
  - 评估1: Trial-level 20×5-fold Stratified CV
  - 评估2: LOSO (Leave-One-Subject-Out) 跨被试验证
  - 特征数: 8个 (ME: mean/peak/slope/variance, MI: mean/peak/slope/variance, 对称)

运行方式:
  python 7_loso_classification_9methods_48subj.py 42    # seed=42
  python 7_loso_classification_9methods_48subj.py 123   # seed=123
  python 7_loso_classification_9methods_48subj.py 999   # seed=999

依赖:
  pip install torch numpy scipy scikit-learn matplotlib PyWavelettes EMD-signal

数据路径 (需根据你的实际路径修改 REPO_ROOT):
  - 特征数据: {REPO_ROOT}/data/eeg_features_all_v6.npy
  - 被试数据: {REPO_ROOT}/data/processed/subject_*.npy
  - 结果输出: {REPO_ROOT}/results/
  - 图片输出: {REPO_ROOT}/figures/
"""
import os, glob, numpy as np, torch, torch.nn as nn, time, warnings, sys, random
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 尝试导入 pywt
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("[WARN] pywt not installed. Wavelet baseline will fall back to bandpass.")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ===== 随机种子设置 + 命令行参数支持 =====
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
print(f"随机种子: {SEED}")

# ---- 超参数 ----
ALPHA, E0 = 0.32, 0.34
SAFE = 1e-6
N_EPOCHS_S1 = 400; N_EPOCHS_S2 = 800
LR = 2e-3
N_SUBJECTS = 48  # 使用全部48名可用被试
SYS_FREQS = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 1.0]
N_PTS = 80
NORM_FLOOR = 0.05
N_CV_REPEATS = 20
FNIRS_SFREQ = 11.0
STIM = 8.0
DUR = 8.0
ICA_MAX_COMP = 4
ICA_RANDOM_STATE = 42
PCA_VAR_RATIO = 0.90
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 4

# ===== 路径配置 (根据你的实际项目路径修改) =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
# 如果直接在VSCode中运行, 可以手动指定路径:
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
FEAT_PATH = os.path.join(REPO_ROOT, 'data', 'eeg_features_all_v6.npy')
FIG_DIR = os.path.join(REPO_ROOT, 'figures')
RES_DIR = os.path.join(REPO_ROOT, 'results')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)

# ============================================================
# 信号处理基础函数
# ============================================================
def bandpass_filter(signal, fs, lowcut=0.01, highcut=0.15, order=4):
    nyq = fs/2.0
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def make_hrf_ir(dt, dur=32.0, a1=6.0, a2=12.0, b1=0.9, b2=0.9, c=0.35):
    t = np.arange(0, dur, dt)
    hrf = ((t**(a1-1))*(b1**a1)*np.exp(-b1*t)/gamma_fn(a1)
           - c*(t**(a2-1))*(b2**a2)*np.exp(-b2*t)/gamma_fn(a2))
    hrf = hrf/(hrf.max()+1e-10)
    return t, hrf

def glm_denoise(signal, t, fs):
    dt = t[1]-t[0]
    _, hrf = make_hrf_ir(dt, dur=20.0)
    boxcar = np.zeros_like(t)
    boxcar[(t>=8)&(t<16)] = 1.0
    boxcar[(t>=24)&(t<32)] = 1.0
    regressor = np.convolve(boxcar, hrf, mode='same')
    regressor = regressor/(regressor.max()+1e-10)
    X = np.column_stack([regressor, np.ones_like(t), np.linspace(0,1,len(t))])
    beta, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
    return X[:,0]*beta[0]

# ============================================================
# ICA 盲源分离去噪 (多通道, 基于频谱噪声识别)
# ============================================================
def ica_denoise_multichannel(hbo_multi, t, hand, fs):
    n_ch, T = hbo_multi.shape
    ch_mean = np.mean(hbo_multi, axis=1, keepdims=True)
    ch_std = np.std(hbo_multi, axis=1, keepdims=True)
    ch_std = np.maximum(ch_std, 1e-10)
    X = (hbo_multi - ch_mean) / ch_std
    n_comp = max(1, min(n_ch - 1, ICA_MAX_COMP))
    try:
        ica = FastICA(n_components=n_comp, random_state=ICA_RANDOM_STATE,
                      max_iter=1000, whiten='unit-variance')
        S = ica.fit_transform(X.T)
        noise_mask = np.zeros(n_comp, dtype=bool)
        peak_freqs = []
        for k in range(n_comp):
            ic = S[:, k]
            N_fft = len(ic)
            freqs = np.fft.rfftfreq(N_fft, d=1.0/fs)
            power = np.abs(np.fft.rfft(ic))**2
            power[0] = 0
            peak_idx = np.argmax(power)
            peak_f = freqs[peak_idx]
            peak_freqs.append(peak_f)
            if peak_f > 0.05:
                noise_mask[k] = True
        n_noise = int(np.sum(noise_mask))
        n_keep = n_comp - n_noise
        if n_keep == 0:
            variances = np.var(S, axis=0)
            keep_idx = np.argmax(variances)
            noise_mask[keep_idx] = False
            n_keep = 1
        S_clean = S.copy()
        S_clean[:, noise_mask] = 0.0
        X_recon = ica.inverse_transform(S_clean).T
        hbo_recon = X_recon * ch_std + ch_mean
        sig_ica = np.mean(hbo_recon, axis=0)
        pre_mask = t < STIM
        sig_ica = sig_ica - np.mean(sig_ica[pre_mask])
        info = {'n_components': n_comp, 'n_noise_removed': n_noise,
                'n_kept': n_keep, 'peak_freqs': [float(f) for f in peak_freqs],
                'converged': True}
    except Exception as e:
        sig_ica = np.mean(hbo_multi, axis=0)
        pre_mask = t < STIM
        sig_ica = sig_ica - np.mean(sig_ica[pre_mask])
        info = {'n_components': n_comp, 'converged': False, 'error': str(e)}
    return sig_ica, info

# ============================================================
# PCA 去噪 (多通道, 保留90%方差主成分)
# ============================================================
def pca_denoise_multichannel(hbo_multi, t):
    n_ch, T = hbo_multi.shape
    try:
        pca = PCA(n_components=min(n_ch, T))
        X_reduced = pca.fit_transform(hbo_multi.T)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, PCA_VAR_RATIO)) + 1
        n_keep = max(1, min(n_keep, X_reduced.shape[1]))
        X_reduced_trim = X_reduced[:, :n_keep]
        comp_keep = pca.components_[:n_keep, :]
        X_recon = X_reduced_trim @ comp_keep
        sig_pca = np.mean(X_recon.T, axis=0)
        pre_mask = t < STIM
        sig_pca = sig_pca - np.mean(sig_pca[pre_mask])
        info = {'n_components': n_keep,
                'variance_ratio': float(np.sum(pca.explained_variance_ratio_[:n_keep])),
                'converged': True}
    except Exception as e:
        sig_pca = np.mean(hbo_multi, axis=0)
        pre_mask = t < STIM
        sig_pca = sig_pca - np.mean(sig_pca[pre_mask])
        info = {'converged': False, 'error': str(e)}
    return sig_pca, info

# ============================================================
# Wavelet 去噪 (单通道, db4 软阈值)
# ============================================================
def wavelet_denoise(signal, wavelet=WAVELET_NAME, level=WAVELET_LEVEL):
    if not HAS_PYWT:
        return signal
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            new_coeffs.append(pywt.threshold(detail, value=uthresh, mode='soft'))
        sig_wt = pywt.waverec(new_coeffs, wavelet)
        if len(sig_wt) > len(signal):
            sig_wt = sig_wt[:len(signal)]
        elif len(sig_wt) < len(signal):
            sig_wt = np.pad(sig_wt, (0, len(signal)-len(sig_wt)))
        return sig_wt
    except Exception as e:
        return signal

# ============================================================
# SSA 去噪 (单通道盲源分离, 奇异谱分析)
# ============================================================
SSA_WINDOW_LENGTH = 20

def ssa_denoise(signal, L=SSA_WINDOW_LENGTH, var_threshold=0.95):
    N = len(signal)
    L = min(L, N // 2)
    K = N - L + 1
    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = signal[i:i+L]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    cumvar = np.cumsum(s**2) / (np.sum(s**2) + 1e-10)
    n_keep = int(np.searchsorted(cumvar, var_threshold)) + 1
    n_keep = max(1, min(n_keep, len(s)))
    X_recon = U[:, :n_keep] @ np.diag(s[:n_keep]) @ Vt[:n_keep, :]
    denoised = np.zeros(N)
    counts = np.zeros(N)
    for i in range(K):
        denoised[i:i+L] += X_recon[:, i]
        counts[i:i+L] += 1
    denoised = denoised / np.maximum(counts, 1)
    return denoised

# ============================================================
# EMD 去噪 (单通道盲源分离, 经验模态分解)
# ============================================================
EMD_MAX_FREQ = 0.15

try:
    from PyEMD import EMD as PyEMD_EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False
    print("[WARN] PyEMD not installed (pip install EMD-signal). EMD will fall back to SSA.")

def emd_denoise(signal, fs=2.5, max_freq=EMD_MAX_FREQ):
    if not HAS_EMD:
        return ssa_denoise(signal)
    try:
        emd = PyEMD_EMD()
        emd.FIXE_H = 5
        imfs = emd(signal)
        keep_imfs = []
        for k, imf in enumerate(imfs):
            N_fft = len(imf)
            freqs = np.fft.rfftfreq(N_fft, d=1.0/fs)
            power = np.abs(np.fft.rfft(imf))**2
            if len(power) > 1:
                power[0] = 0
            peak_f = freqs[np.argmax(power)] if len(power) > 0 else 0
            if peak_f <= max_freq or k == 0:
                keep_imfs.append(imf)
        if len(keep_imfs) == 0:
            return signal
        return np.sum(keep_imfs, axis=0)
    except Exception as e:
        return ssa_denoise(signal)

# ============================================================
# PIHD 网络
# ============================================================
class FVQNet(nn.Module):
    def __init__(self, hidden=[20,40,40,20]):
        super().__init__()
        layers=[]; prev=1
        for h in hidden:
            layers.append(nn.Linear(prev,h)); prev=h
        layers.append(nn.Linear(prev,3))
        self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,t):
        x=t
        for i,layer in enumerate(self.net):
            x=layer(x)
            if i<len(self.net)-1: x=nn.functional.softplus(x)
        f=nn.functional.softplus(x[:,0:1])
        v=nn.functional.softplus(x[:,1:2])
        q=1.0+x[:,2:3]
        return f,v,q

class FinNet(nn.Module):
    def __init__(self,hidden=[16,16]):
        super().__init__()
        layers=[]; prev=1
        for h in hidden:
            layers.append(nn.Linear(prev,h)); layers.append(nn.Softplus()); prev=h
        layers.append(nn.Linear(prev,1))
        self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,t):
        return nn.functional.softplus(self.net(t))

class SystemicNoise(nn.Module):
    def __init__(self,freqs,device):
        super().__init__()
        n=len(freqs)
        self.cos_coeffs=nn.Parameter(torch.zeros(n,device=device))
        self.sin_coeffs=nn.Parameter(torch.zeros(n,device=device))
        self.bias=nn.Parameter(torch.tensor(0.0,device=device))
        self.drift=nn.Parameter(torch.tensor(0.0,device=device))
        freq_t=torch.tensor(freqs,dtype=torch.float32,device=device)
        self.register_buffer('freq_t',freq_t)
    def forward(self,t,cos_c=None,sin_c=None,b=None,d=None):
        cc = self.cos_coeffs if cos_c is None else cos_c
        sc = self.sin_coeffs if sin_c is None else sin_c
        bb = self.bias if b is None else b
        dd = self.drift if d is None else d
        angles=2*np.pi*t*self.freq_t.unsqueeze(0)
        cos_p=torch.cos(angles)@cc.unsqueeze(1)
        sin_p=torch.sin(angles)@sc.unsqueeze(1)
        return cos_p+sin_p+bb+dd*t

def make_hrf_prior_torch(t_np,onset=8.0,dur=8.0):
    f_in=np.zeros_like(t_np)
    mask=(t_np>=onset)&(t_np<onset+dur)
    dt=t_np[1]-t_np[0]
    box=np.where(mask,1.0,0.0)
    t_krn=np.arange(0,8,dt)
    krn=(t_krn**(2.5-1))*(1.5**2.5)*np.exp(-1.5*t_krn)/gamma_fn(2.5)
    krn=krn/krn.sum()
    conv=np.convolve(box,krn,mode='same')
    conv=conv/(conv.max()+1e-10)
    return conv*0.5

def bw_residual(t,f_in,f,v,q,tau,eps):
    ones=torch.ones_like(t)
    df=torch.autograd.grad(f,t,ones,create_graph=True)[0]
    dv=torch.autograd.grad(v,t,ones,create_graph=True)[0]
    dq=torch.autograd.grad(q,t,ones,create_graph=True)[0]
    f_s=torch.clamp(f,min=SAFE); v_s=torch.clamp(v,min=SAFE); eps_s=torch.clamp(eps,min=SAFE)
    inv_f=1.0/f_s
    e_arg=torch.clamp(1.0-(1.0/eps_s)*(1.0-inv_f),min=-0.99)
    E_t=E0/(1.0+E0*e_arg)
    rf=df-(f_in-(f-1)/tau)
    rv=dv-(f-v_s**(1/ALPHA))/tau
    rq=dq-(f*E_t-v_s**(1/ALPHA)*q/v_s)/tau
    return rf,rv,rq

def pihd_denoise(t_np, hbo_norm, device=DEVICE):
    set_seed(SEED)
    t_t=torch.tensor(t_np,dtype=torch.float32,device=device).unsqueeze(1)
    h_t=torch.tensor(hbo_norm,dtype=torch.float32,device=device).unsqueeze(1)
    rest_mask_np=(t_np<7.5)|((t_np>16.5)&(t_np<23.5))
    task_mask_np=((t_np>=8)&(t_np<16))|((t_np>=24)&(t_np<32))
    rest_idx=torch.tensor(rest_mask_np,dtype=torch.bool,device=device)
    task_idx=torch.tensor(task_mask_np,dtype=torch.bool,device=device)

    noise_s1=SystemicNoise(SYS_FREQS,device).to(device)
    opt1=torch.optim.Adam(noise_s1.parameters(),lr=LR)
    sched1=torch.optim.lr_scheduler.CosineAnnealingLR(opt1,T_max=N_EPOCHS_S1,eta_min=1e-5)
    best_loss1=float('inf'); best_noise_sd=None
    for ep in range(N_EPOCHS_S1):
        opt1.zero_grad()
        noise_pred=noise_s1(t_t)
        loss=torch.mean((noise_pred[rest_idx]-h_t[rest_idx])**2)
        loss+=torch.mean(noise_pred**2)*0.001
        loss+=noise_s1.drift**2*0.01
        if torch.isnan(loss): opt1.zero_grad(); continue
        loss.backward(); opt1.step(); sched1.step()
        if loss.item()<best_loss1:
            best_loss1=loss.item()
            best_noise_sd={k:v.clone().detach() for k,v in noise_s1.state_dict().items()}
    fix_cos=best_noise_sd['cos_coeffs']; fix_sin=best_noise_sd['sin_coeffs']

    t_t.requires_grad_(True)
    fvq=FVQNet().to(device); fin=FinNet().to(device)
    noise_bias=nn.Parameter(best_noise_sd['bias'].clone())
    noise_drift=nn.Parameter(best_noise_sd['drift'].clone())
    tau_raw=nn.Parameter(torch.tensor(0.0,device=device))
    eps_raw=nn.Parameter(torch.tensor(0.0,device=device))
    params=list(fvq.parameters())+list(fin.parameters())+[tau_raw,eps_raw,noise_bias,noise_drift]
    opt2=torch.optim.Adam(params,lr=LR)
    sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=N_EPOCHS_S2,eta_min=1e-5)
    hrf_prior=make_hrf_prior_torch(t_np)
    hrf_t=torch.tensor(hrf_prior,dtype=torch.float32,device=device).unsqueeze(1)
    best_loss2=float('inf'); best_sd=None
    for ep in range(N_EPOCHS_S2):
        opt2.zero_grad()
        tau=0.5+5.5*torch.sigmoid(tau_raw); eps=0.1+1.9*torch.sigmoid(eps_raw)
        f_in=fin(t_t); f,v,q=fvq(t_t); hbo_neural=v-q
        noise_pred=noise_s1(t_t,cos_c=fix_cos,sin_c=fix_sin,b=noise_bias,d=noise_drift)
        hbo_pred=hbo_neural+noise_pred
        loss_data=torch.mean((hbo_pred-h_t)**2)
        rf,rv,rq=bw_residual(t_t,f_in,f,v,q,tau,eps)
        loss_eq=torch.mean(rf**2)+torch.mean(rv**2)+torch.mean(rq**2)
        t0=torch.tensor([[0.0]],device=device)
        with torch.enable_grad():
            t0.requires_grad_(True); f0,v0,q0=fvq(t0); f_in0=fin(t0)
        loss_ic=((f0-1)**2+(v0-1)**2+(q0-1)**2+f_in0**2).mean()
        f_in_rest=f_in[rest_idx.unsqueeze(1)]
        loss_f_rest=torch.mean(f_in_rest**2)*2.0
        loss_hrf=torch.mean((f_in-hrf_t)**2)*1.0 if ep<400 else torch.tensor(0.0,device=device)
        hbo_neural_task=hbo_neural[task_idx.unsqueeze(1)]
        loss_var=torch.exp(-torch.var(hbo_neural_task)*5.0)*0.5
        loss=loss_data+loss_eq+0.5*loss_ic+loss_f_rest+loss_hrf+loss_var
        if torch.isnan(loss): opt2.zero_grad(); continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0)
        opt2.step(); sched2.step()
        if loss.item()<best_loss2:
            best_loss2=loss.item()
            best_sd={
                'fvq':{k:v.clone().detach() for k,v in fvq.state_dict().items()},
                'fin':{k:v.clone().detach() for k,v in fin.state_dict().items()},
                'tau':tau.item(),'eps':eps.item(),
                'noise_bias':noise_bias.item(),'noise_drift':noise_drift.item()
            }
    fvq.load_state_dict(best_sd['fvq']); fvq.eval()
    with torch.no_grad():
        t_t.requires_grad_(False)
        f,v,q=fvq(t_t)
        hbo_neural_pred=(v-q).cpu().numpy().flatten()
    rest_bl=np.mean(hbo_neural_pred[rest_mask_np])
    hbo_neural_pred=hbo_neural_pred-rest_bl
    return hbo_neural_pred

# ============================================================
# 特征提取 (Task特征, 非task-rest差值)
# ME和MI各4个特征: mean, peak, slope, variance (共8个, 对称)
# ============================================================
def extract_features(sig, t):
    me_mask=(t>=8)&(t<16)
    mi_mask=(t>=24)&(t<32)
    me_sig=sig[me_mask]; mi_sig=sig[mi_mask]
    me_t=t[me_mask]; mi_t=t[mi_mask]
    # ME特征 (4个)
    me_mean=np.mean(me_sig)
    me_peak=np.max(me_sig)
    me_slope=np.polyfit(me_t-np.mean(me_t), me_sig, 1)[0] if len(me_t)>1 else 0.0
    me_var=np.var(me_sig)
    # MI特征 (4个, 与ME对称)
    mi_mean=np.mean(mi_sig)
    mi_peak=np.max(mi_sig)
    mi_slope=np.polyfit(mi_t-np.mean(mi_t), mi_sig, 1)[0] if len(mi_t)>1 else 0.0
    mi_var=np.var(mi_sig)
    return np.array([me_mean, mi_mean, me_peak, mi_peak, me_slope, mi_slope, me_var, mi_var])

# ============================================================
# 主实验: 信号处理+特征提取
# ============================================================
print("Loading v6 feature data...")
feat=np.load(FEAT_PATH,allow_pickle=True).item()
trials=feat['trials']
print(f"  Total trials: {feat['params']['n_total_trials']}")

subjects=sorted(set(t['subject'] for t in trials))
if len(subjects)>N_SUBJECTS:
    rng_select=np.random.RandomState(42)
    selected_subj=list(rng_select.choice(subjects,N_SUBJECTS,replace=False))
else:
    selected_subj=subjects
selected_trials=[t for t in trials if t['subject'] in selected_subj]
print(f"  Using {len(selected_subj)} subjects, {len(selected_trials)} trials")

t_ds=np.linspace(0,32,N_PTS)
fs_ds=1.0/(t_ds[1]-t_ds[0])

methods=['raw','bandpass','glm','ica','pca','wavelet','ssa','emd','pihd']
method_names={'raw':'Raw HbO','bandpass':'Bandpass','glm':'GLM',
              'ica':'ICA','pca':'PCA','wavelet':'Wavelet','ssa':'SSA','emd':'EMD','pihd':'PIHD (Ours)'}

all_features={m:[] for m in methods}
all_labels=[]
all_groups=[]
all_subj_ids=sorted(selected_subj)
subj_to_idx={s:i for i,s in enumerate(all_subj_ids)}

print(f"\nSignal processing & feature extraction ({len(selected_trials)} trials)...")
print("-"*70)

# 预加载 subject 原始数据 (ICA/PCA 多通道需要)
print("Pre-loading subject files for multi-channel baselines...")
subject_cache = {}
needed_subjects = set(t['subject'] for t in selected_trials)
for sf in glob.glob(os.path.join(DATA_DIR, 'subject_*.npy')):
    subj_id = os.path.basename(sf).replace('subject_','').replace('.npy','')
    if subj_id in needed_subjects:
        subject_cache[subj_id] = np.load(sf, allow_pickle=True).item()
print(f"  Loaded {len(subject_cache)} subject files")

t0_total=time.time()
ica_converge=0; ica_total=0
pca_total=0
for i,tr in enumerate(selected_trials):
    ti=time.time()
    t_orig=tr['t']; hbo_orig=tr['hbo_n']
    fs_orig=1.0/(t_orig[1]-t_orig[0])
    label = 1 if tr['hand']=='right' else 0
    subj_id = tr['subject']
    all_labels.append(label)
    all_groups.append(subj_to_idx[subj_id])

    hbo_ds=np.interp(t_ds,t_orig,hbo_orig)
    rest_mask=t_ds<8
    me_mask=(t_ds>=8)&(t_ds<16)

    bl_mean=np.mean(hbo_ds[rest_mask])
    bl_std=np.std(hbo_ds[rest_mask])
    bl_std=max(bl_std,NORM_FLOOR)
    hbo_norm=(hbo_ds-bl_mean)/bl_std

    # Raw
    sig_raw=hbo_norm.copy()
    all_features['raw'].append(extract_features(sig_raw, t_ds))

    # Bandpass
    hbo_bl_orig=hbo_orig-np.mean(hbo_orig[:int(8*fs_orig)])
    hbo_bp_orig=bandpass_filter(hbo_bl_orig,fs_orig,lowcut=0.01,highcut=0.15)
    hbo_bp_ds=np.interp(t_ds,t_orig,hbo_bp_orig)
    sig_bp=(hbo_bp_ds-np.mean(hbo_bp_ds[rest_mask]))/bl_std
    all_features['bandpass'].append(extract_features(sig_bp, t_ds))

    # GLM
    sig_glm_raw=glm_denoise(hbo_norm,t_ds,fs_ds)
    sig_glm=sig_glm_raw-np.mean(sig_glm_raw[rest_mask])
    all_features['glm'].append(extract_features(sig_glm, t_ds))

    # PIHD
    sig_pihd=pihd_denoise(t_ds,hbo_norm)
    all_features['pihd'].append(extract_features(sig_pihd, t_ds))

    # ---- 多通道基线: ICA + PCA ----
    try:
        hand = tr['hand']
        good_chs = tr['good_chs']
        subj_data = subject_cache[subj_id]
        subj_trials_raw = subj_data['trials']
        tn = tr.get('trial_number', -1)
        matching = [st for st in subj_trials_raw
                    if st.get('hand') == hand and st.get('trial_number', -1) == tn]
        if len(matching) == 0:
            hand_trials_raw = [st for st in subj_trials_raw if st.get('hand') == hand]
            hand_count = sum(1 for prev in selected_trials[:i]
                             if prev['subject'] == subj_id and prev['hand'] == hand)
            if hand_count < len(hand_trials_raw):
                matching = [hand_trials_raw[hand_count]]

        if len(matching) > 0:
            raw_trial = matching[0]
            hbo_all = raw_trial['fnirs_hbo_all']
            T_orig_mc = hbo_all.shape[1]
            t_orig_raw = np.arange(T_orig_mc) / FNIRS_SFREQ
            n_keep = int(32.0 * FNIRS_SFREQ)
            if T_orig_mc >= n_keep:
                hbo_all_crop = hbo_all[:, :n_keep]
                t_orig_crop = t_orig_raw[:n_keep]
                hbo_multi = hbo_all_crop[good_chs].copy()
                pre_mask_orig = t_orig_crop < STIM
                for ch_i in range(hbo_multi.shape[0]):
                    hbo_multi[ch_i] = hbo_multi[ch_i] - np.mean(hbo_multi[ch_i, pre_mask_orig])

                # ICA
                sig_ica_orig, ica_info = ica_denoise_multichannel(hbo_multi, t_orig_crop, hand, FNIRS_SFREQ)
                sig_ica_ds = np.interp(t_ds, t_orig_crop, sig_ica_orig)
                sig_ica = (sig_ica_ds - np.mean(sig_ica_ds[rest_mask])) / bl_std
                if ica_info['converged']:
                    ica_converge += 1
                ica_total += 1

                # PCA
                sig_pca_orig, pca_info = pca_denoise_multichannel(hbo_multi, t_orig_crop)
                sig_pca_ds = np.interp(t_ds, t_orig_crop, sig_pca_orig)
                sig_pca = (sig_pca_ds - np.mean(sig_pca_ds[rest_mask])) / bl_std
                pca_total += 1
            else:
                sig_ica = sig_raw.copy()
                sig_pca = sig_raw.copy()
                ica_total += 1; pca_total += 1
        else:
            sig_ica = sig_raw.copy()
            sig_pca = sig_raw.copy()
            ica_total += 1; pca_total += 1
    except Exception as e:
        sig_ica = sig_raw.copy()
        sig_pca = sig_raw.copy()
        ica_total += 1; pca_total += 1

    all_features['ica'].append(extract_features(sig_ica, t_ds))
    all_features['pca'].append(extract_features(sig_pca, t_ds))

    # Wavelet (单通道)
    if HAS_PYWT:
        sig_wt_raw = wavelet_denoise(hbo_norm)
        sig_wt = sig_wt_raw - np.mean(sig_wt_raw[rest_mask])
    else:
        sig_wt = sig_raw.copy()
    all_features['wavelet'].append(extract_features(sig_wt, t_ds))

    # SSA (单通道盲源分离)
    sig_ssa_raw = ssa_denoise(hbo_norm)
    sig_ssa = sig_ssa_raw - np.mean(sig_ssa_raw[rest_mask])
    all_features['ssa'].append(extract_features(sig_ssa, t_ds))

    # EMD (单通道盲源分离)
    sig_emd_raw = emd_denoise(hbo_norm, fs=fs_ds)
    sig_emd = sig_emd_raw - np.mean(sig_emd_raw[rest_mask])
    all_features['emd'].append(extract_features(sig_emd, t_ds))

    dt_run=time.time()-ti
    if (i+1)%20==0 or i==0:
        print(f"  [{i+1:3d}/{len(selected_trials)}] {tr['hand']:<5} "
              f"ME_raw={np.mean(sig_raw[me_mask]):+.3f} ME_pihd={np.mean(sig_pihd[me_mask]):+.3f} "
              f"({dt_run:.1f}s)",flush=True)

total_time=time.time()-t0_total
print(f"\nFeature extraction done: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"ICA convergence: {ica_converge}/{ica_total} ({100*ica_converge/max(1,ica_total):.1f}%)")
print(f"PCA processed: {pca_total} trials")

y=np.array(all_labels)
groups=np.array(all_groups)
print(f"Label distribution: right={np.sum(y==1)}, left={np.sum(y==0)}")
print(f"Number of unique subjects for LOSO: {len(np.unique(groups))}")

# ============================================================
# 评估1: Trial-level 5-fold CV
# ============================================================
print(f"\n{'='*70}")
print(f"EVALUATION 1: Trial-level {N_CV_REPEATS}x5-fold Stratified CV")
print(f"{'='*70}")

cv_trial={m:{'acc':[],'auc':[],'f1':[]} for m in methods}

for rep in range(N_CV_REPEATS):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=rep)
    for train_idx,test_idx in skf.split(np.zeros(len(y)),y):
        for m in methods:
            X=np.array(all_features[m])
            X_train,X_test=X[train_idx],X[test_idx]
            y_train,y_test=y[train_idx],y[test_idx]
            scaler=StandardScaler()
            X_train_s=scaler.fit_transform(X_train)
            X_test_s=scaler.transform(X_test)
            clf=LogisticRegression(C=1.0,max_iter=1000,random_state=rep)
            clf.fit(X_train_s,y_train)
            y_pred=clf.predict(X_test_s)
            y_prob=clf.predict_proba(X_test_s)[:,1]
            acc=accuracy_score(y_test,y_pred)
            try:
                auc=roc_auc_score(y_test,y_prob)
            except:
                auc=0.5
            f1=f1_score(y_test,y_pred)
            cv_trial[m]['acc'].append(acc)
            cv_trial[m]['auc'].append(auc)
            cv_trial[m]['f1'].append(f1)

print(f"\n{'Method':<15} {'Accuracy':>14} {'AUC':>14} {'F1':>14}")
print("-"*60)
trial_summary=[]
for m in methods:
    acc_mean=np.mean(cv_trial[m]['acc']); acc_std=np.std(cv_trial[m]['acc'])
    auc_mean=np.mean(cv_trial[m]['auc']); auc_std=np.std(cv_trial[m]['auc'])
    f1_mean=np.mean(cv_trial[m]['f1']); f1_std=np.std(cv_trial[m]['f1'])
    print(f"{method_names[m]:<15} {acc_mean:>6.3f}+/-{acc_std:.3f} {auc_mean:>6.3f}+/-{auc_std:.3f} "
          f"{f1_mean:>6.3f}+/-{f1_std:.3f}")
    trial_summary.append({'method':m,'acc':acc_mean,'acc_std':acc_std,
                          'auc':auc_mean,'auc_std':auc_std,'f1':f1_mean,'f1_std':f1_std})

# Paired t-test for trial-level CV
print(f"\n--- Paired t-test: PIHD vs baselines (trial-level AUC) ---")
pihd_auc_trial=np.array(cv_trial['pihd']['auc'])
for m in ['raw','bandpass','glm','ica','pca','wavelet','ssa','emd']:
    base_auc=np.array(cv_trial[m]['auc'])
    t_comp,p_comp=stats.ttest_rel(pihd_auc_trial,base_auc)
    print(f"  PIHD vs {method_names[m]:<12}: AUC delta={np.mean(pihd_auc_trial-base_auc):+.4f} p={p_comp:.2e}")

# ============================================================
# 评估2: LOSO
# ============================================================
print(f"\n{'='*70}")
print("EVALUATION 2: LOSO (Leave-One-Subject-Out)")
print(f"{'='*70}")

logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(groups=groups)
print(f"Number of LOSO folds: {n_splits} (one per held-out subject)")

cv_loso={m:{'acc':[],'auc':[],'f1':[],'y_true':[],'y_prob':[]} for m in methods}

for fold_i, (train_idx, test_idx) in enumerate(logo.split(np.zeros(len(y)), y, groups)):
    held_out_subj = all_subj_ids[groups[test_idx[0]]]
    for m in methods:
        X=np.array(all_features[m])
        X_train,X_test=X[train_idx],X[test_idx]
        y_train,y_test=y[train_idx],y[test_idx]
        scaler=StandardScaler()
        X_train_s=scaler.fit_transform(X_train)
        X_test_s=scaler.transform(X_test)
        clf=LogisticRegression(C=1.0,max_iter=1000,random_state=42)
        clf.fit(X_train_s,y_train)
        y_pred=clf.predict(X_test_s)
        y_prob=clf.predict_proba(X_test_s)[:,1]
        acc=accuracy_score(y_test,y_pred)
        try:
            auc=roc_auc_score(y_test,y_prob)
        except:
            auc=np.nan
        f1=f1_score(y_test,y_pred,zero_division=0)
        cv_loso[m]['acc'].append(acc)
        cv_loso[m]['auc'].append(auc)
        cv_loso[m]['f1'].append(f1)
        cv_loso[m]['y_true'].extend(y_test.tolist())
        cv_loso[m]['y_prob'].extend(y_prob.tolist())

    if (fold_i+1)%5==0 or fold_i==0:
        cur_pihd_acc=np.mean([a for a in cv_loso['pihd']['acc'] if not np.isnan(a)])
        print(f"  Fold {fold_i+1:2d}/{n_splits}: held-out subj={held_out_subj}, "
              f"PIHD acc so far={cur_pihd_acc:.3f}", flush=True)

# Compute LOSO summary
print(f"\n{'Method':<15} {'Accuracy':>14} {'AUC':>14} {'F1':>14} {'n_auc':>6}")
print("-"*65)
loso_summary=[]
for m in methods:
    acc_arr=np.array(cv_loso[m]['acc'])
    auc_arr=np.array([a for a in cv_loso[m]['auc'] if not np.isnan(a)])
    f1_arr=np.array(cv_loso[m]['f1'])
    acc_mean=np.mean(acc_arr); acc_std=np.std(acc_arr)
    auc_mean=np.mean(auc_arr); auc_std=np.std(auc_arr)
    f1_mean=np.mean(f1_arr); f1_std=np.std(f1_arr)
    n_valid_auc=len(auc_arr)
    print(f"{method_names[m]:<15} {acc_mean:>6.3f}+/-{acc_std:.3f} {auc_mean:>6.3f}+/-{auc_std:.3f} "
          f"{f1_mean:>6.3f}+/-{f1_std:.3f} {n_valid_auc:>6d}")
    loso_summary.append({'method':m,'acc':acc_mean,'acc_std':acc_std,
                         'auc':auc_mean,'auc_std':auc_std,'f1':f1_mean,'f1_std':f1_std,
                         'n_valid_auc':n_valid_auc,
                         'acc_per_subj':acc_arr,'auc_per_subj':np.array(cv_loso[m]['auc'])})

# Pooled AUC
print(f"\n--- Pooled AUC (all test predictions concatenated) ---")
for m in methods:
    yt=np.array(cv_loso[m]['y_true'])
    yp=np.array(cv_loso[m]['y_prob'])
    try:
        pooled_auc=roc_auc_score(yt,yp)
    except:
        pooled_auc=0.5
    pooled_acc=accuracy_score(yt,(yp>0.5).astype(int))
    print(f"  {method_names[m]:<15}: pooled Acc={pooled_acc:.3f}, pooled AUC={pooled_auc:.3f}")

# Paired t-test for LOSO
print(f"\n--- Paired t-test: PIHD vs baselines (LOSO, per-subject AUC) ---")
for m in ['raw','bandpass','glm','ica','pca','wavelet','ssa','emd']:
    base_auc_all=np.array(cv_loso[m]['auc'])
    pihd_auc_all=np.array(cv_loso['pihd']['auc'])
    valid=~(np.isnan(base_auc_all)|np.isnan(pihd_auc_all))
    base_valid=base_auc_all[valid]
    pihd_valid=pihd_auc_all[valid]
    if len(base_valid)>5:
        t_comp,p_comp=stats.ttest_rel(pihd_valid,base_valid)
        print(f"  PIHD vs {method_names[m]:<12}: AUC delta={np.mean(pihd_valid-base_valid):+.4f} "
              f"p={p_comp:.2e} (n={len(base_valid)} subjects)")
    else:
        print(f"  PIHD vs {method_names[m]:<12}: insufficient valid subjects for t-test")

print(f"\n--- Paired t-test: PIHD vs baselines (LOSO, per-subject Accuracy) ---")
pihd_acc_loso=np.array(cv_loso['pihd']['acc'])
for m in ['raw','bandpass','glm','ica','pca','wavelet','ssa','emd']:
    base_acc=np.array(cv_loso[m]['acc'])
    t_comp,p_comp=stats.ttest_rel(pihd_acc_loso,base_acc)
    print(f"  PIHD vs {method_names[m]:<12}: Acc delta={np.mean(pihd_acc_loso-base_acc):+.4f} p={p_comp:.2e}")

# Compare to chance
print(f"\n--- One-sample t-test vs chance (0.50) ---")
for m in methods:
    acc_arr=np.array(cv_loso[m]['acc'])
    t_chance,p_chance=stats.ttest_1samp(acc_arr,0.5)
    sig="***" if p_chance<0.001 else "**" if p_chance<0.01 else "*" if p_chance<0.05 else "n.s."
    print(f"  {method_names[m]:<15} LOSO Acc vs chance: {np.mean(acc_arr):.3f} p={p_chance:.2e} {sig}")

# ============================================================
# 保存结果
# ============================================================
save_path=os.path.join(RES_DIR,f'loso_classification_9methods_seed{SEED}.npy')
np.save(save_path,{
    'seed':SEED,
    'trial_cv':cv_trial,'trial_summary':trial_summary,
    'loso_cv':cv_loso,'loso_summary':loso_summary,
    'n_trials':len(selected_trials),'n_subjects':len(selected_subj),
    'n_cv_repeats':N_CV_REPEATS,
    'features':all_features,'labels':y,'groups':groups,
    'subject_ids':all_subj_ids,
    'ica_converge_rate':ica_converge/max(1,ica_total)
})
print(f"\nResults saved: {save_path}")

# JSON结果汇总
import json
json_summary = {
    'seed': SEED,
    'trial_cv': {m: {
        'auc_mean': float(np.mean(cv_trial[m]['auc'])),
        'auc_std': float(np.std(cv_trial[m]['auc'])),
        'acc_mean': float(np.mean(cv_trial[m]['acc'])),
        'f1_mean': float(np.mean(cv_trial[m]['f1']))
    } for m in methods},
    'loso_cv': {m: {
        'auc_mean': float(np.mean([a for a in cv_loso[m]['auc'] if not np.isnan(a)])),
        'auc_std': float(np.std([a for a in cv_loso[m]['auc'] if not np.isnan(a)])),
        'acc_mean': float(np.mean(cv_loso[m]['acc'])),
        'f1_mean': float(np.mean(cv_loso[m]['f1']))
    } for m in methods}
}
json_path = os.path.join(RES_DIR, f'loso_9methods_results_seed{SEED}.json')
with open(json_path, 'w') as f:
    json.dump(json_summary, f, indent=2)
print(f"JSON results saved: {json_path}")

# ============================================================
# 总结对比表
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY COMPARISON (9 methods)")
print(f"{'='*70}")
print(f"{'Method':<15} | {'Trial 5-fold AUC':>16} | {'LOSO AUC':>16} | {'AUC drop':>9}")
print("-"*65)
for i,m in enumerate(methods):
    t_auc=trial_summary[i]['auc']
    l_auc=loso_summary[i]['auc']
    drop=t_auc-l_auc
    print(f"{method_names[m]:<15} | {t_auc:>7.3f}+/-{trial_summary[i]['auc_std']:.3f}  | "
          f"{l_auc:>7.3f}+/-{loso_summary[i]['auc_std']:.3f}  | {drop:>7.3f}")

# ============================================================
# 画图: 三子图对比
# ============================================================
print(f"\nGenerating comparison figure...")

method_colors = {
    'raw': '#636E72', 'bandpass': '#00B894', 'glm': '#FDCB6E',
    'ica': '#E17055', 'pca': '#6C5CE7', 'wavelet': '#00CEC9',
    'ssa': '#D63031', 'emd': '#74B9FF',
    'pihd': '#0984E3'
}
method_labels = [method_names[m] for m in methods]
x_pos = np.arange(len(methods))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) LOSO per-subject AUC 箱线图
ax = axes[0]
data_boxes = []
for mi, m in enumerate(methods):
    auc_valid = [a for a in cv_loso[m]['auc'] if not np.isnan(a)]
    data_boxes.append(auc_valid)
bp = ax.boxplot(data_boxes, positions=x_pos, widths=0.5, patch_artist=True,
            showfliers=False, medianprops=dict(color='black', lw=1.5))
for mi, patch in enumerate(bp['boxes']):
    patch.set_facecolor(method_colors[methods[mi]])
    patch.set_alpha(0.6)
ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Chance')
ax.set_xticks(x_pos); ax.set_xticklabels(method_labels, rotation=25, ha='right', fontsize=8.5)
ax.set_ylabel("AUC"); ax.set_title("(a) LOSO Per-subject AUC")
ax.set_ylim(0.25, 0.85); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

# (b) Trial-level AUC
ax = axes[1]
auc_mt = [trial_summary[i]['auc'] for i in range(len(methods))]
auc_st = [trial_summary[i]['auc_std'] for i in range(len(methods))]
bars = ax.bar(x_pos, auc_mt, yerr=auc_st, capsize=4,
              color=[method_colors[m] for m in methods], alpha=0.85, edgecolor='white')
ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Chance')
ax.set_xticks(x_pos); ax.set_xticklabels(method_labels, rotation=25, ha='right', fontsize=8.5)
ax.set_ylabel("AUC"); ax.set_title(f"(b) Trial-level {N_CV_REPEATS}x5-fold AUC")
ax.set_ylim(0.4, 0.75); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, auc_mt):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.008, f'{val:.3f}', ha='center', fontsize=8.5)

# (c) LOSO mean AUC
ax = axes[2]
auc_ml = [loso_summary[i]['auc'] for i in range(len(methods))]
auc_sl = [loso_summary[i]['auc_std'] for i in range(len(methods))]
bars = ax.bar(x_pos, auc_ml, yerr=auc_sl, capsize=4,
              color=[method_colors[m] for m in methods], alpha=0.85, edgecolor='white')
ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Chance')
ax.set_xticks(x_pos); ax.set_xticklabels(method_labels, rotation=25, ha='right', fontsize=8.5)
ax.set_ylabel("AUC"); ax.set_title("(c) LOSO Cross-subject AUC")
ax.set_ylim(0.4, 0.75); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, auc_ml):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.008, f'{val:.3f}', ha='center', fontsize=8.5)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, f'loso_classification_9methods_seed{SEED}.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Figure saved: {fig_path}")

print(f"\nDone! (seed={SEED})")
