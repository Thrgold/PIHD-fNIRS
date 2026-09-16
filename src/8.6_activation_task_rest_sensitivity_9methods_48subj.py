# -*- coding: utf-8 -*-
"""Task-minus-preceding-rest activation sensitivity analysis.

This is an independent companion to Step 8. It does not modify existing
results. The same 48 participants, 480 trials, preprocessing, nine methods,
and subject-level inference are used. PIHD and the baselines are rerun because
the existing result files do not contain Rest1/Rest2 means or time series.

Contrasts
---------
ME contrast = mean(signal, 8--16 s) - mean(signal, 2--7 s)
MI contrast = mean(signal, 24--32 s) - mean(signal, 17--23 s)

For each method, trial contrasts are averaged within participant (n=48), then
Cohen's d_z and a two-sided one-sample t-test versus zero are computed.
Bonferroni correction is applied across 9 methods x 2 contrasts (18 tests).

Usage
-----
python 8.6_activation_task_rest_sensitivity_9methods_48subj.py 42
python 8.6_activation_task_rest_sensitivity_9methods_48subj.py 123
python 8.6_activation_task_rest_sensitivity_9methods_48subj.py 999

Outputs
-------
results/activation_task_rest_9methods_seed{SEED}.npy
results/json/activation_task_rest_9methods_seed{SEED}.json
figures/activation_task_rest_9methods_seed{SEED}.png
"""
import os, glob, sys, random, json, numpy as np, torch, torch.nn as nn, time, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA, PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 尝试导入 pywt
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("[WARN] pywt not installed. Wavelet will fall back to raw signal.")

# 尝试导入 PyEMD
try:
    from PyEMD import EMD as PyEMD_EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False
    print("[WARN] PyEMD not installed (pip install EMD-signal). EMD will fall back to SSA.")

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

# ---- 超参数 (与7_loso_classification_9methods完全一致) ----
ALPHA, E0 = 0.32, 0.34
SAFE = 1e-6
N_EPOCHS_S1 = 400; N_EPOCHS_S2 = 800
LR = 2e-3
N_SUBJECTS = 48  # 使用全部48名可用被试
SYS_FREQS = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 1.0]
N_PTS = 80
NORM_FLOOR = 0.05
FNIRS_SFREQ = 11.0
STIM = 8.0
DUR = 8.0
ICA_MAX_COMP = 4
ICA_RANDOM_STATE = 42
PCA_VAR_RATIO = 0.90
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 4
SSA_WINDOW_LENGTH = 20
EMD_MAX_FREQ = 0.15

# ---- 路径 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
# 如果直接在VSCode中运行, 可以手动指定路径:
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
FEAT_PATH = os.path.join(REPO_ROOT, 'data', 'eeg_features_all_v6.npy')
FIG_DIR = os.path.join(REPO_ROOT, 'figures')
RES_DIR = os.path.join(REPO_ROOT, 'results')
JSON_DIR = os.path.join(RES_DIR, 'json')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True); os.makedirs(JSON_DIR, exist_ok=True)

# ============================================================
# 信号处理基础函数 (与分类脚本完全一致)
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
        info = {'converged': True}
    except Exception as e:
        sig_ica = np.mean(hbo_multi, axis=0)
        pre_mask = t < STIM
        sig_ica = sig_ica - np.mean(sig_ica[pre_mask])
        info = {'converged': False, 'error': str(e)}
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
        info = {'converged': True}
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
    except:
        return signal

# ============================================================
# SSA 去噪 (单通道盲源分离, 奇异谱分析)
# ============================================================
def ssa_denoise(signal, L=SSA_WINDOW_LENGTH, var_threshold=0.95):
    """Singular Spectrum Analysis: SVD分解轨迹矩阵, 保留主要成分重构"""
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
def emd_denoise(signal, fs=2.5, max_freq=EMD_MAX_FREQ):
    """EMD分解为IMF, 剔除峰值频率>max_freq的噪声IMF, 剩余IMF重构"""
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
    except:
        return ssa_denoise(signal)

# ============================================================
# PIHD 网络 (与分类脚本完全一致)
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
    # 清理GPU显存, 防止480 trial累积导致OOM
    del t_t, h_t, fvq, fin, noise_s1, opt1, opt2, sched1, sched2
    del fix_cos, fix_sin, noise_bias, noise_drift, tau_raw, eps_raw
    del hrf_t, best_sd, best_noise_sd, rest_idx, task_idx
    torch.cuda.empty_cache()
    return hbo_neural_pred

# ============================================================
# 主实验: 加载数据 → 信号处理 → 计算激活检测 d_z
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

# 【修改】9种方法
methods=['raw','bandpass','glm','ica','pca','wavelet','ssa','emd','pihd']
method_names={'raw':'Raw HbO','bandpass':'Bandpass','glm':'GLM',
              'ica':'ICA','pca':'PCA','wavelet':'Wavelet',
              'ssa':'SSA','emd':'EMD','pihd':'PIHD (Ours)'}

# Store task-minus-preceding-rest contrasts for every trial.
me_means={m:[] for m in methods}
mi_means={m:[] for m in methods}
trial_subjects=[]

print(f"\nSignal processing & activation detection ({len(selected_trials)} trials, seed={SEED})...")
print("-"*70)

# 预加载 subject 原始数据
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

for i,tr in enumerate(selected_trials):
    ti=time.time()
    t_orig=tr['t']; hbo_orig=tr['hbo_n']
    fs_orig=1.0/(t_orig[1]-t_orig[0])

    trial_subjects.append(tr['subject'])

    hbo_ds=np.interp(t_ds,t_orig,hbo_orig)
    rest_mask=t_ds<8
    rest1_mask = (t_ds>=2)  & (t_ds<7)
    rest2_mask = (t_ds>=17) & (t_ds<23)
    me_mask    = (t_ds>=8)  & (t_ds<16)
    mi_mask    = (t_ds>=24) & (t_ds<32)

    def me_contrast(signal):
        return float(np.mean(signal[me_mask]) - np.mean(signal[rest1_mask]))

    def mi_contrast(signal):
        return float(np.mean(signal[mi_mask]) - np.mean(signal[rest2_mask]))

    bl_mean=np.mean(hbo_ds[rest_mask])
    bl_std=np.std(hbo_ds[rest_mask])
    bl_std=max(bl_std,NORM_FLOOR)
    hbo_norm=(hbo_ds-bl_mean)/bl_std

    # Raw — task均值
    sig_raw=hbo_norm.copy()
    me_means['raw'].append(me_contrast(sig_raw))
    mi_means['raw'].append(mi_contrast(sig_raw))

    # Bandpass — task均值
    hbo_bl_orig=hbo_orig-np.mean(hbo_orig[:int(8*fs_orig)])
    hbo_bp_orig=bandpass_filter(hbo_bl_orig,fs_orig,lowcut=0.01,highcut=0.15)
    hbo_bp_ds=np.interp(t_ds,t_orig,hbo_bp_orig)
    sig_bp=(hbo_bp_ds-np.mean(hbo_bp_ds[rest_mask]))/bl_std
    me_means['bandpass'].append(me_contrast(sig_bp))
    mi_means['bandpass'].append(mi_contrast(sig_bp))

    # GLM — task均值
    sig_glm_raw=glm_denoise(hbo_norm,t_ds,fs_ds)
    sig_glm=sig_glm_raw-np.mean(sig_glm_raw[rest_mask])
    me_means['glm'].append(me_contrast(sig_glm))
    mi_means['glm'].append(mi_contrast(sig_glm))

    # PIHD — task均值
    sig_pihd=pihd_denoise(t_ds,hbo_norm)
    me_means['pihd'].append(me_contrast(sig_pihd))
    mi_means['pihd'].append(mi_contrast(sig_pihd))

    # ---- 多通道基线: ICA + PCA ----
    try:
        subj_id = tr['subject']
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
            else:
                sig_ica = sig_raw.copy()
                sig_pca = sig_raw.copy()
                ica_total += 1
        else:
            sig_ica = sig_raw.copy()
            sig_pca = sig_raw.copy()
            ica_total += 1
    except Exception as e:
        sig_ica = sig_raw.copy()
        sig_pca = sig_raw.copy()
        ica_total += 1

    me_means['ica'].append(me_contrast(sig_ica))
    mi_means['ica'].append(mi_contrast(sig_ica))
    me_means['pca'].append(me_contrast(sig_pca))
    mi_means['pca'].append(mi_contrast(sig_pca))

    # Wavelet (单通道) — task均值
    if HAS_PYWT:
        sig_wt_raw = wavelet_denoise(hbo_norm)
        sig_wt = sig_wt_raw - np.mean(sig_wt_raw[rest_mask])
    else:
        sig_wt = sig_raw.copy()
    me_means['wavelet'].append(me_contrast(sig_wt))
    mi_means['wavelet'].append(mi_contrast(sig_wt))

    # SSA (单通道盲源分离) — task均值
    sig_ssa_raw = ssa_denoise(hbo_norm)
    sig_ssa = sig_ssa_raw - np.mean(sig_ssa_raw[rest_mask])
    me_means['ssa'].append(me_contrast(sig_ssa))
    mi_means['ssa'].append(mi_contrast(sig_ssa))

    # EMD (单通道盲源分离) — task均值
    sig_emd_raw = emd_denoise(hbo_norm, fs=fs_ds)
    sig_emd = sig_emd_raw - np.mean(sig_emd_raw[rest_mask])
    me_means['emd'].append(me_contrast(sig_emd))
    mi_means['emd'].append(mi_contrast(sig_emd))

    dt_run=time.time()-ti
    if (i+1)%20==0 or i==0:
        me_mean_raw = me_contrast(sig_raw)
        me_mean_pihd = me_contrast(sig_pihd)
        print(f"  [{i+1:3d}/{len(selected_trials)}] {tr['hand']:<5} "
              f"ME_mean_raw={me_mean_raw:+.3f} ME_mean_pihd={me_mean_pihd:+.3f} "
              f"({dt_run:.1f}s)",flush=True)

total_time=time.time()-t0_total
print(f"\nSignal processing done: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"ICA convergence: {ica_converge}/{ica_total}")

# ============================================================
# Subject-level task-minus-preceding-rest inference.
# ============================================================
print(f"\n{'='*70}")
print(f"Task-minus-preceding-rest sensitivity analysis (subject-level, n={len(selected_subj)}, seed={SEED})")
print(f"{'='*70}")

dz_results = {}
subj_arr = np.array(trial_subjects)

print(f"\n{'Method':<15} {'ME d_z':>9} {'ME p(raw)':>12} {'ME p(adj)':>12} {'MI d_z':>9} {'MI p(raw)':>12} {'MI p(adj)':>12}")
print("-"*92)
for m in methods:
    me_arr = np.array(me_means[m])
    mi_arr = np.array(mi_means[m])

    # 按被试聚合: 每个被试的trial取均值 → 48个被试均值
    me_subject = []
    mi_subject = []
    for subj in selected_subj:
        mask = subj_arr == subj
        me_subject.append(np.mean(me_arr[mask]))
        mi_subject.append(np.mean(mi_arr[mask]))
    me_subject = np.array(me_subject)
    mi_subject = np.array(mi_subject)

    me_dz = np.mean(me_subject) / (np.std(me_subject, ddof=1) + 1e-10)
    mi_dz = np.mean(mi_subject) / (np.std(mi_subject, ddof=1) + 1e-10)

    me_t, me_p = stats.ttest_1samp(me_subject, 0)
    mi_t, mi_p = stats.ttest_1samp(mi_subject, 0)

    me_p_bonf = min(float(me_p) * 18, 1.0)
    mi_p_bonf = min(float(mi_p) * 18, 1.0)
    me_sig = "***" if me_p_bonf<0.001 else "**" if me_p_bonf<0.01 else "*" if me_p_bonf<0.05 else "n.s."
    mi_sig = "***" if mi_p_bonf<0.001 else "**" if mi_p_bonf<0.01 else "*" if mi_p_bonf<0.05 else "n.s."

    print(f"{method_names[m]:<15} {me_dz:>+8.3f} {me_p:>12.2e} {me_p_bonf:>12.2e} "
          f"{mi_dz:>+8.3f} {mi_p:>12.2e} {mi_p_bonf:>12.2e}")

    dz_results[m] = {
        'me_dz': me_dz, 'me_t': me_t, 'me_p_raw': me_p,
        'me_p_bonferroni': me_p_bonf, 'me_sig_bonferroni': me_sig,
        'mi_dz': mi_dz, 'mi_t': mi_t, 'mi_p_raw': mi_p,
        'mi_p_bonferroni': mi_p_bonf, 'mi_sig_bonferroni': mi_sig,
        'me_values': me_arr, 'mi_values': mi_arr,
        'me_subject_values': me_subject, 'mi_subject_values': mi_subject
    }

dz_save_path = os.path.join(RES_DIR, f'activation_task_rest_9methods_seed{SEED}.npy')
np.save(dz_save_path, {
    'seed': SEED,
    'dz_results': dz_results,
    'me_means': me_means,
    'mi_means': mi_means,
    'trial_subjects': trial_subjects,
    'n_trials': len(selected_trials),
    'n_subjects': len(selected_subj),
    'analysis_level': 'subject',
    'diff_method': 'task minus preceding rest',
    'windows_seconds': {'rest1': [2, 7], 'me': [8, 16], 'rest2': [17, 23], 'mi': [24, 32]},
    'multiple_comparison': {'method': 'Bonferroni', 'n_tests': 18, 'alpha': 0.05}
})
print(f"\nActivation detection results saved: {dz_save_path}")

# JSON summary for paper updates.
json_result = {
    'seed': SEED,
    'analysis_level': 'subject',
    'diff_method': 'task minus preceding rest',
    'windows_seconds': {'rest1': [2, 7], 'me': [8, 16], 'rest2': [17, 23], 'mi': [24, 32]},
    'n_subjects': len(selected_subj), 'n_trials': len(selected_trials),
    'multiple_comparison': {'method': 'Bonferroni', 'n_tests': 18, 'alpha': 0.05}
}
for m in methods:
    json_result[m] = {
        'me_dz': float(dz_results[m]['me_dz']),
        'me_p_raw': float(dz_results[m]['me_p_raw']),
        'me_p_bonferroni': float(dz_results[m]['me_p_bonferroni']),
        'me_sig_bonferroni': dz_results[m]['me_sig_bonferroni'],
        'mi_dz': float(dz_results[m]['mi_dz']),
        'mi_p_raw': float(dz_results[m]['mi_p_raw']),
        'mi_p_bonferroni': float(dz_results[m]['mi_p_bonferroni']),
        'mi_sig_bonferroni': dz_results[m]['mi_sig_bonferroni'],
    }
json_path = os.path.join(JSON_DIR, f'activation_task_rest_9methods_seed{SEED}.json')
with open(json_path, 'w') as f:
    json.dump(json_result, f, indent=2)
print(f"JSON results saved: {json_path}")

# Compact diagnostic figure; inference values come from participant means.
method_colors = {
    'raw': '#7F8C8D', 'bandpass': '#2CB49C', 'glm': '#F4C35E',
    'ica': '#E98263', 'pca': '#7564D8', 'wavelet': '#2CBFC1',
    'ssa': '#DE4548', 'emd': '#79AFE2', 'pihd': '#2D8FCF'
}
x_pos = np.arange(len(methods))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, task in zip(axes, ('me', 'mi')):
    values = [dz_results[m][f'{task}_dz'] for m in methods]
    ax.bar(x_pos, values, color=[method_colors[m] for m in methods], width=0.68)
    ax.axhline(0, color='black', linewidth=0.8)
    for j, m in enumerate(methods):
        sig = dz_results[m][f'{task}_sig_bonferroni']
        if sig != 'n.s.':
            offset = 0.035 if values[j] >= 0 else -0.075
            ax.text(j, values[j] + offset, sig, ha='center', va='bottom' if values[j]>=0 else 'top',
                    color='#C8102E', fontweight='bold', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([method_names[m] for m in methods], rotation=42, ha='right')
    ax.set_title('ME minus preceding Rest1' if task == 'me' else 'MI minus preceding Rest2')
    ax.set_ylabel("Cohen's $d_z$")
    ax.grid(axis='y', alpha=0.25)
fig.suptitle(f'Task-minus-preceding-rest sensitivity analysis (seed={SEED}, n=48)')
fig.tight_layout()
fig_path = os.path.join(FIG_DIR, f'activation_task_rest_9methods_seed{SEED}.png')
fig.savefig(fig_path, dpi=220, bbox_inches='tight')
plt.close(fig)
print(f"Figure saved: {fig_path}")
print("Done. Existing Step 8 results were not overwritten.")
sys.exit(0)

# ============================================================
# 加载已有分类结果 (支持JSON和npy两种格式)
# ============================================================
print(f"\n{'='*70}")
print("Loading existing classification results...")
print(f"{'='*70}")

# 【修改】优先加载JSON格式 (新9methods结果), 其次npy
cls_json_path = os.path.join(RES_DIR, f'loso_9methods_results_seed{SEED}.json')
cls_npy_path = os.path.join(RES_DIR, f'loso_classification_9methods_seed{SEED}.npy')

trial_summary = None
loso_summary = None

if os.path.exists(cls_json_path):
    # JSON格式 (新结果)
    with open(cls_json_path, 'r') as f:
        cls_data = json.load(f)
    print(f"  Loaded JSON: {cls_json_path}")
    # 转换为 list 格式 (与旧npy格式兼容)
    trial_summary = [{'auc': cls_data['trial_cv'][m]['auc_mean'],
                      'auc_std': cls_data['trial_cv'][m]['auc_std']} for m in methods]
    loso_summary = [{'auc': cls_data['loso_cv'][m]['auc_mean'],
                     'auc_std': cls_data['loso_cv'][m]['auc_std']} for m in methods]
    print(f"  Trial-level and LOSO results for {len(methods)} methods (JSON)")
elif os.path.exists(cls_npy_path):
    cls_data = np.load(cls_npy_path, allow_pickle=True).item()
    trial_summary = cls_data['trial_summary']
    loso_summary = cls_data['loso_summary']
    print(f"  Loaded NPY: {cls_npy_path}")
else:
    print(f"  [ERROR] Classification results not found!")
    print(f"  Tried: {cls_json_path}")
    print(f"  Tried: {cls_npy_path}")
    print(f"  Please run 7_loso_classification_9methods_48subj.py {SEED} first!")
    sys.exit(1)

# ============================================================
# 合并3子图: (a) d_z, (b) Trial AUC, (c) LOSO AUC
# ============================================================
print(f"\n{'='*70}")
print("Generating combined figure (3 panels)...")
print(f"{'='*70}")

# 【修改】9种方法的颜色
method_colors = {
    'raw': '#636E72', 'bandpass': '#00B894', 'glm': '#FDCB6E',
    'ica': '#E17055', 'pca': '#6C5CE7', 'wavelet': '#00CEC9',
    'ssa': '#D63031', 'emd': '#74B9FF', 'pihd': '#0984E3'
}
method_labels = [method_names[m] for m in methods]
x_pos = np.arange(len(methods))
bar_width = 0.35

# 【修改】图更宽以容纳9个方法
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# ---- (a) Activation Detection: Cohen's d_z ----
ax = axes[0]
me_dz_vals = [dz_results[m]['me_dz'] for m in methods]
mi_dz_vals = [dz_results[m]['mi_dz'] for m in methods]

bars_me = ax.bar(x_pos - bar_width/2, me_dz_vals, bar_width,
                 color=[method_colors[m] for m in methods], alpha=0.85,
                 edgecolor='white', label='ME (8-16s)')
bars_mi = ax.bar(x_pos + bar_width/2, mi_dz_vals, bar_width,
                 color=[method_colors[m] for m in methods], alpha=0.45,
                 edgecolor='gray', linewidth=0.8, hatch='///', label='MI (24-32s)')

# 显著性标注
for j, m in enumerate(methods):
    if dz_results[m]['me_sig'] != 'n.s.':
        ax.text(x_pos[j] - bar_width/2, me_dz_vals[j] + 0.03,
                dz_results[m]['me_sig'], ha='center', fontsize=8, fontweight='bold', color='red')
    if dz_results[m]['mi_sig'] != 'n.s.':
        ax.text(x_pos[j] + bar_width/2, mi_dz_vals[j] + 0.03,
                dz_results[m]['mi_sig'], ha='center', fontsize=8, fontweight='bold', color='red')

ax.axhline(0, color='black', ls='-', lw=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(method_labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel("Cohen's $d_z$ (baseline-referenced task mean)"); ax.set_title("(a) Task-window Activation Detection")
ax.legend(fontsize=8, loc='upper left'); ax.grid(axis='y', alpha=0.3)

# ---- (b) Trial-level 5-fold CV AUC ----
ax = axes[1]
auc_mt = [trial_summary[i]['auc'] for i in range(len(methods))]
auc_st = [trial_summary[i]['auc_std'] for i in range(len(methods))]
bars = ax.bar(x_pos, auc_mt, yerr=auc_st, capsize=4,
              color=[method_colors[m] for m in methods], alpha=0.85, edgecolor='white')
ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Chance')
ax.set_xticks(x_pos); ax.set_xticklabels(method_labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel("AUC"); ax.set_title("(b) Trial-level 5-fold CV AUC")
ax.set_ylim(0.4, 0.75); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, auc_mt):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.008, f'{val:.3f}', ha='center', fontsize=7.5)

# ---- (c) LOSO Cross-subject AUC ----
ax = axes[2]
auc_ml = [loso_summary[i]['auc'] for i in range(len(methods))]
auc_sl = [loso_summary[i]['auc_std'] for i in range(len(methods))]
bars = ax.bar(x_pos, auc_ml, yerr=auc_sl, capsize=4,
              color=[method_colors[m] for m in methods], alpha=0.85, edgecolor='white')
ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Chance')
ax.set_xticks(x_pos); ax.set_xticklabels(method_labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel("AUC"); ax.set_title("(c) LOSO Cross-subject AUC")
ax.set_ylim(0.4, 0.75); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, auc_ml):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.008, f'{val:.3f}', ha='center', fontsize=7.5)

# 【修改】LOSO显著性: 从数据计算而非硬编码
# 使用 one-sample t-test vs 0.5 (近似, 后续替换为permutation test)
n_loso = len(selected_subj)
for j, m in enumerate(methods):
    mean_auc = loso_summary[j]['auc']
    std_auc = loso_summary[j]['auc_std']
    if std_auc > 1e-10:
        t_stat = (mean_auc - 0.5) / (std_auc / np.sqrt(n_loso))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_loso - 1))
    else:
        p_val = 1.0
    sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "n.s."
    if sig != "n.s.":
        ax.text(x_pos[j], auc_ml[j] + auc_sl[j] + 0.02, sig,
                ha='center', fontsize=9, fontweight='bold', color='red')

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, f'combined_9methods_seed{SEED}.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Combined figure saved: {fig_path}")

# ============================================================
# 总结
# ============================================================
print(f"\n{'='*70}")
print(f"SUMMARY (9 methods, 3 evaluations, seed={SEED}, baseline-referenced task-window means, n={len(selected_subj)})")
print(f"{'='*70}")
print(f"{'Method':<15} | {'ME d_z':>8} | {'MI d_z':>8} | {'Trial AUC':>10} | {'LOSO AUC':>10}")
print("-"*62)
for j, m in enumerate(methods):
    me_dz = dz_results[m]['me_dz']
    mi_dz = dz_results[m]['mi_dz']
    t_auc = trial_summary[j]['auc'] if trial_summary else 0
    l_auc = loso_summary[j]['auc'] if loso_summary else 0
    print(f"{method_names[m]:<15} | {me_dz:>+7.3f} | {mi_dz:>+7.3f} | {t_auc:>9.3f}  | {l_auc:>9.3f}")

print(f"\nDone! (subject-level analysis, baseline-referenced task-window means, n={len(selected_subj)})")
