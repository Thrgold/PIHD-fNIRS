# -*- coding: utf-8 -*-
"""Key 48-participant PIHD ablation study.

Conditions
----------
1. Full PIHD
2. w/o L_ode
3. w/o Stage 1

The evaluation matches the main experiment:
- 48 participants / 480 trials
- 8 symmetric ME/MI classification features
- trial-level 20 x 5-fold CV
- pooled LOSO AUC from concatenated held-out probabilities
- subject-level task-minus-preceding-rest Cohen's d_z

Usage
-----
python 6.5_ablation_key3_48subj.py 42
python 6.5_ablation_key3_48subj.py 123
python 6.5_ablation_key3_48subj.py 999

Run this only after the three Step 8.6 jobs have finished; it retrains PIHD.
"""
import os, sys, random, json, numpy as np, torch, torch.nn as nn, time, warnings
warnings.filterwarnings('ignore')
from scipy.special import gamma as gamma_fn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {DEVICE}")

# ===== 随机种子设置 =====
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

ALPHA, E0 = 0.32, 0.34
SAFE = 1e-6
N_EPOCHS_S1 = 400; N_EPOCHS_S2 = 800
LR = 2e-3
N_SUBJECTS = 48
SYS_FREQS = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 1.0]
N_PTS = 80
NORM_FLOOR = 0.05
N_CV_REPEATS = 20

# 自动检测数据路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FEAT_PATH = os.path.join(REPO_ROOT, 'data', 'eeg_features_all_v6.npy')
RES_DIR = os.path.join(REPO_ROOT, 'results')
NPY_DIR = os.path.join(RES_DIR, 'npy')
JSON_DIR = os.path.join(RES_DIR, 'json')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# ============================================================
# 网络定义 (与其他脚本一致)
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

# ============================================================
# PIHD 去噪函数（带消融开关）
# ============================================================
def pihd_denoise_ablation(t_np, hbo_norm, device=DEVICE,
                          use_ode=True, use_rest_init=True,
                          use_rest_constraint=True, use_var_reg=True):
    """
    PIHD两阶段去噪，带消融控制开关

    Parameters
    ----------
    use_ode : bool
        是否使用B-W ODE物理约束 (L_ode)
    use_rest_init : bool
        是否使用第一阶段静息期噪声预训练 (Stage 1)
    use_rest_constraint : bool
        是否在第二阶段约束静息期f_in≈0 (L_rest)
    use_var_reg : bool
        是否使用神经信号方差正则化 (L_var)
    """
    set_seed(SEED)

    t_t=torch.tensor(t_np,dtype=torch.float32,device=device).unsqueeze(1)
    h_t=torch.tensor(hbo_norm,dtype=torch.float32,device=device).unsqueeze(1)
    rest_mask_np=(t_np<7.5)|((t_np>16.5)&(t_np<23.5))
    task_mask_np=((t_np>=8)&(t_np<16))|((t_np>=24)&(t_np<32))
    rest_idx=torch.tensor(rest_mask_np,dtype=torch.bool,device=device)
    task_idx=torch.tensor(task_mask_np,dtype=torch.bool,device=device)

    init_bias = torch.tensor(0.0, device=device)
    init_drift = torch.tensor(0.0, device=device)
    fix_cos = None; fix_sin = None

    # ===== Stage 1: 静息期噪声预训练 =====
    noise_s1 = None
    if use_rest_init:
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
        init_bias=best_noise_sd['bias'].clone()
        init_drift=best_noise_sd['drift'].clone()

    # ===== Stage 2: 神经成分提取 =====
    t_t.requires_grad_(True)
    fvq=FVQNet().to(device); fin=FinNet().to(device)
    noise_bias=nn.Parameter(init_bias.clone())
    noise_drift=nn.Parameter(init_drift.clone())
    tau_raw=nn.Parameter(torch.tensor(0.0,device=device))
    eps_raw=nn.Parameter(torch.tensor(0.0,device=device))

    params=list(fvq.parameters())+list(fin.parameters())+[tau_raw,eps_raw,noise_bias,noise_drift]
    if not use_rest_init:
        noise_full=SystemicNoise(SYS_FREQS,device).to(device)
        params += list(noise_full.parameters())
    opt2=torch.optim.Adam(params,lr=LR)
    sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=N_EPOCHS_S2,eta_min=1e-5)

    hrf_prior=make_hrf_prior_torch(t_np)
    hrf_t=torch.tensor(hrf_prior,dtype=torch.float32,device=device).unsqueeze(1)

    best_loss2=float('inf'); best_sd=None
    for ep in range(N_EPOCHS_S2):
        opt2.zero_grad()
        tau=0.5+5.5*torch.sigmoid(tau_raw)
        eps=0.1+1.9*torch.sigmoid(eps_raw)
        f_in=fin(t_t); f,v,q=fvq(t_t)
        hbo_neural=v-q

        if use_rest_init:
            noise_pred=noise_s1(t_t,cos_c=fix_cos,sin_c=fix_sin,b=noise_bias,d=noise_drift)
        else:
            noise_pred=noise_full(t_t)

        hbo_pred=hbo_neural+noise_pred
        loss_data=torch.mean((hbo_pred-h_t)**2)

        if use_ode:
            rf,rv,rq=bw_residual(t_t,f_in,f,v,q,tau,eps)
            loss_eq=torch.mean(rf**2)+torch.mean(rv**2)+torch.mean(rq**2)
        else:
            loss_eq=torch.tensor(0.0,device=device)

        t0=torch.tensor([[0.0]],device=device)
        with torch.enable_grad():
            t0.requires_grad_(True); f0,v0,q0=fvq(t0); f_in0=fin(t0)
        loss_ic=((f0-1)**2+(v0-1)**2+(q0-1)**2+f_in0**2).mean()

        if use_rest_constraint:
            f_in_rest=f_in[rest_idx.unsqueeze(1)]
            loss_f_rest=torch.mean(f_in_rest**2)*2.0
        else:
            loss_f_rest=torch.tensor(0.0,device=device)

        loss_hrf=torch.mean((f_in-hrf_t)**2)*1.0 if ep<400 else torch.tensor(0.0,device=device)

        if use_var_reg:
            hbo_neural_task=hbo_neural[task_idx.unsqueeze(1)]
            loss_var=torch.exp(-torch.var(hbo_neural_task)*5.0)*0.5
        else:
            loss_var=torch.tensor(0.0,device=device)

        loss=loss_data+loss_eq+0.5*loss_ic+loss_f_rest+loss_hrf+loss_var
        if torch.isnan(loss): opt2.zero_grad(); continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0)
        opt2.step(); sched2.step()
        if loss.item()<best_loss2:
            best_loss2=loss.item()
            best_sd={
                'fvq':{k:v.clone().detach() for k,v in fvq.state_dict().items()},
                'fin':{k:v.clone().detach() for k,v in fin.state_dict().items()},
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
# 特征提取
# ============================================================
def extract_features(sig, t):
    me_mask=(t>=8)&(t<16)
    mi_mask=(t>=24)&(t<32)
    me_sig=sig[me_mask]; mi_sig=sig[mi_mask]
    me_t=t[me_mask]; mi_t=t[mi_mask]
    me_mean=np.mean(me_sig)
    mi_mean=np.mean(mi_sig)
    me_peak=np.max(me_sig)
    me_slope=np.polyfit(me_t-np.mean(me_t), me_sig, 1)[0] if len(me_t)>1 else 0.0
    me_var=np.var(me_sig)
    mi_peak=np.max(mi_sig)
    mi_slope=np.polyfit(mi_t-np.mean(mi_t), mi_sig, 1)[0] if len(mi_t)>1 else 0.0
    mi_var=np.var(mi_sig)
    return np.array([me_mean, mi_mean, me_peak, mi_peak,
                     me_slope, mi_slope, me_var, mi_var])

# ============================================================
# 主实验
# ============================================================
print("加载v6特征数据...")
feat=np.load(FEAT_PATH,allow_pickle=True).item()
trials=feat['trials']
print(f"  总trial数: {feat['params']['n_total_trials']}")

subjects=sorted(set(t['subject'] for t in trials))
if len(subjects)>N_SUBJECTS:
    rng_select=np.random.RandomState(42)
    selected_subj=list(rng_select.choice(subjects,N_SUBJECTS,replace=False))
else:
    selected_subj=subjects
selected_trials=[t for t in trials if t['subject'] in selected_subj]
print(f"  使用 {len(selected_subj)} 个被试, {len(selected_trials)} 个trials")

t_ds=np.linspace(0,32,N_PTS)
fs_ds=1.0/(t_ds[1]-t_ds[0])

ablation_conditions = [
    {'name': 'Full PIHD',      'use_ode': True,  'use_rest_init': True,  'use_rest_constraint': True,  'use_var_reg': True},
    {'name': 'w/o L_ode',      'use_ode': False, 'use_rest_init': True,  'use_rest_constraint': True,  'use_var_reg': True},
    {'name': 'w/o Stage 1',    'use_ode': True,  'use_rest_init': False, 'use_rest_constraint': True,  'use_var_reg': True},
]

checkpoint_path = os.path.join(NPY_DIR, f'ablation_key3_48subj_seed{SEED}_checkpoint.npy')
if os.path.exists(checkpoint_path):
    checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
    if checkpoint.get('n_subjects') != len(selected_subj):
        raise ValueError(f"Checkpoint subject count does not match: {checkpoint_path}")
    all_results = checkpoint.get('results', {})
    print(f"Resuming checkpoint with completed conditions: {list(all_results)}")
else:
    all_results = {}

for cond in ablation_conditions:
    cond_name = cond['name']
    if cond_name in all_results:
        print(f"\nSkipping completed condition from checkpoint: {cond_name}")
        continue
    print(f"\n{'='*60}")
    print(f"运行消融条件: {cond_name}")
    print(f"{'='*60}")

    all_features = []
    all_labels = []
    all_subj_ids = []         # 【新增】记录每个trial的被试ID
    me_diff_list = []         # 【修改】改为存储 task-rest 配对差值
    mi_diff_list = []

    t0_cond = time.time()
    for i, tr in enumerate(selected_trials):
        ti = time.time()
        t_orig = tr['t']; hbo_orig = tr['hbo_n']
        label = 1 if tr['hand']=='right' else 0
        all_labels.append(label)
        all_subj_ids.append(tr['subject'])   # 【新增】

        hbo_ds = np.interp(t_ds, t_orig, hbo_orig)
        rest_mask  = t_ds < 8
        me_mask    = (t_ds >= 8) & (t_ds < 16)
        rest2_mask = (t_ds >= 17) & (t_ds < 23)   # 【新增】Rest2 窗口
        mi_mask    = (t_ds >= 24) & (t_ds < 32)
        rest1_mask = (t_ds >= 2) & (t_ds < 7)     # 【新增】Rest1 窗口 (避开边缘)

        bl_mean = np.mean(hbo_ds[rest_mask])
        bl_std = np.std(hbo_ds[rest_mask])
        bl_std = max(bl_std, NORM_FLOOR)
        hbo_norm = (hbo_ds - bl_mean) / bl_std

        sig = pihd_denoise_ablation(
            t_ds, hbo_norm, device=DEVICE,
            use_ode=cond['use_ode'],
            use_rest_init=cond['use_rest_init'],
            use_rest_constraint=cond['use_rest_constraint'],
            use_var_reg=cond['use_var_reg']
        )

        all_features.append(extract_features(sig, t_ds))

        # 【修改】计算 task-rest 配对差值 (与主实验 Table I 一致)
        #   ME: 8-16s vs Rest1: 2-7s
        #   MI: 24-32s vs Rest2: 17-23s
        me_diff = np.mean(sig[me_mask]) - np.mean(sig[rest1_mask])
        mi_diff = np.mean(sig[mi_mask]) - np.mean(sig[rest2_mask])
        me_diff_list.append(me_diff)
        mi_diff_list.append(mi_diff)

        dt_run = time.time() - ti
        if (i+1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(selected_trials)}] ME_diff={me_diff:+.3f} "
                  f"MI_diff={mi_diff:+.3f} ({dt_run:.1f}s)", flush=True)

    # ================================================================
    # Activation: subject-level aggregation (n=48) and one-sample t-test.
    # ================================================================
    subj_ids = np.array(all_subj_ids)
    unique_subj = sorted(set(all_subj_ids))

    # 按被试聚合: 每个被试的trial取均值 → 30个独立数据点
    me_subj = np.array([np.mean([me_diff_list[j] for j in range(len(me_diff_list))
                                  if subj_ids[j] == s]) for s in unique_subj])
    mi_subj = np.array([np.mean([mi_diff_list[j] for j in range(len(mi_diff_list))
                                  if subj_ids[j] == s]) for s in unique_subj])

    # Cohen's d_z (subject-level, ddof=1)
    me_d = np.mean(me_subj) / (np.std(me_subj, ddof=1) + 1e-10)
    mi_d = np.mean(mi_subj) / (np.std(mi_subj, ddof=1) + 1e-10)

    # 单样本t检验 (against 0)
    me_t, me_p = stats.ttest_1samp(me_subj, 0)
    mi_t, mi_p = stats.ttest_1samp(mi_subj, 0)

    # ================================================================
    # 分类: trial-level 20×5-fold CV (保留原有逻辑)
    # ================================================================
    y = np.array(all_labels)
    X = np.array(all_features)
    groups = np.array(all_subj_ids)    # 【新增】用于LOSO

    acc_list, auc_list, f1_list = [], [], []
    for rep in range(N_CV_REPEATS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=rep)
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            y_prob = clf.predict_proba(X_test_s)[:, 1]
            acc_list.append(accuracy_score(y_test, y_pred))
            auc_list.append(roc_auc_score(y_test, y_prob))
            f1_list.append(f1_score(y_test, y_pred))

    acc_cv = np.mean(acc_list)
    auc_cv = np.mean(auc_list)
    f1_cv = np.mean(f1_list)

    # LOSO: concatenate every held-out prediction, then compute pooled AUC.
    logo = LeaveOneGroupOut()
    pooled_prob = np.full(len(y), np.nan, dtype=float)
    pooled_pred = np.full(len(y), -1, dtype=int)
    fold_auc = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1]
        pooled_prob[test_idx] = y_prob
        pooled_pred[test_idx] = y_pred
        if len(set(y_test)) > 1:
            fold_auc.append(roc_auc_score(y_test, y_prob))
        else:
            fold_auc.append(np.nan)

    if np.isnan(pooled_prob).any() or (pooled_pred < 0).any():
        raise RuntimeError(f"{cond_name}: missing held-out LOSO predictions")
    acc_loso = accuracy_score(y, pooled_pred)
    auc_loso = roc_auc_score(y, pooled_prob)

    dt_cond = time.time() - t0_cond
    print(f"\n  {cond_name} 结果 ({dt_cond/60:.1f}min):")
    print(f"    [Subject-level n={len(unique_subj)}] "
          f"ME d_z = {me_d:.3f} (p={me_p:.2e}), "
          f"MI d_z = {mi_d:.3f} (p={mi_p:.2e})")
    print(f"    [Trial-level CV]  AUC = {auc_cv:.3f}, Acc = {acc_cv:.3f}, F1 = {f1_cv:.3f}")
    print(f"    [LOSO]            AUC = {auc_loso:.3f}, Acc = {acc_loso:.3f}")

    all_results[cond_name] = {
        'me_d': me_d, 'mi_d': mi_d,
        'me_t': me_t, 'mi_t': mi_t,
        'me_p': me_p, 'mi_p': mi_p,
        'me_p_bonferroni_6': min(float(me_p) * 6, 1.0),
        'mi_p_bonferroni_6': min(float(mi_p) * 6, 1.0),
        'me_subject_values': me_subj,
        'mi_subject_values': mi_subj,
        'acc_cv': acc_cv, 'auc_cv': auc_cv, 'f1_cv': f1_cv,
        'acc_loso': acc_loso, 'auc_loso': auc_loso,
        'loso_true': y.copy(),
        'loso_prob': pooled_prob,
        'loso_pred': pooled_pred,
        'loso_fold_auc': np.asarray(fold_auc, dtype=float),
        'features': X,
        'labels': y,
        'groups': groups,
    }
    np.save(checkpoint_path, {
        'seed': SEED,
        'n_subjects': len(selected_subj),
        'conditions': [c['name'] for c in ablation_conditions],
        'results': all_results,
    }, allow_pickle=True)
    print(f"  Checkpoint saved: {checkpoint_path}")

# ================================================================
# 汇总输出
# ================================================================
print("\n" + "="*95)
print(f"Key 48-subject Ablation Results (seed={SEED})")
print("="*95)
print(f"{'Method':<15} {'ME d_z':>8} {'MI d_z':>8} {'AUC_cv':>8} "
      f"{'AUC_LOSO':>10} {'Acc_LOSO':>10} {'MI p':>12}")
print("-"*95)
for cond_name, res in all_results.items():
    marker = "*" if cond_name == "Full PIHD" else " "
    print(f"{marker}{cond_name:<14} {res['me_d']:>8.3f} {res['mi_d']:>8.3f} "
          f"{res['auc_cv']:>8.3f} {res['auc_loso']:>10.3f} {res['acc_loso']:>10.3f} "
          f"{res['mi_p']:>12.2e}")
print("="*95)

# ================================================================
# Save a complete NPY plus a human-readable JSON summary. Existing 30-subject
# ablation files are not overwritten.
# ================================================================
save_payload = {
    'seed': SEED,
    'n_subjects': len(selected_subj),
    'n_trials': len(selected_trials),
    'conditions': [c['name'] for c in ablation_conditions],
    'classification_features': [
        'me_mean', 'mi_mean', 'me_peak', 'mi_peak',
        'me_slope', 'mi_slope', 'me_variance', 'mi_variance'
    ],
    'loso_statistic': 'pooled AUC from concatenated held-out predictions',
    'activation_contrast': {
        'ME': 'mean(8-16s) - mean(2-7s)',
        'MI': 'mean(24-32s) - mean(17-23s)'
    },
    'results': all_results,
}
save_path = os.path.join(NPY_DIR, f'ablation_key3_48subj_seed{SEED}.npy')
np.save(save_path, save_payload, allow_pickle=True)

json_payload = {
    'seed': SEED,
    'n_subjects': len(selected_subj),
    'n_trials': len(selected_trials),
    'loso_statistic': save_payload['loso_statistic'],
    'activation_contrast': save_payload['activation_contrast'],
    'results': {}
}
for name, result in all_results.items():
    json_payload['results'][name] = {
        'me_dz': float(result['me_d']),
        'mi_dz': float(result['mi_d']),
        'me_p_raw': float(result['me_p']),
        'mi_p_raw': float(result['mi_p']),
        'me_p_bonferroni_6': float(result['me_p_bonferroni_6']),
        'mi_p_bonferroni_6': float(result['mi_p_bonferroni_6']),
        'trial_cv_auc': float(result['auc_cv']),
        'trial_cv_accuracy': float(result['acc_cv']),
        'trial_cv_f1': float(result['f1_cv']),
        'pooled_loso_auc': float(result['auc_loso']),
        'pooled_loso_accuracy': float(result['acc_loso']),
    }
json_path = os.path.join(JSON_DIR, f'ablation_key3_48subj_seed{SEED}.json')
with open(json_path, 'w', encoding='utf-8') as stream:
    json.dump(json_payload, stream, ensure_ascii=False, indent=2)

print(f"\nComplete results saved to: {save_path}")
print(f"JSON summary saved to: {json_path}")
print("Done!")
