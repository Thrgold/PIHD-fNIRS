# -*- coding: utf-8 -*-
"""
RQ1验证v4: PINN-BW信号分解 - 真实fNIRS数据 (两阶段+强约束版)
==========================================================
48被试版本: N_SUBJECTS = 48, 路径修正至 data/eeg_features/
"""
import os, sys, random, numpy as np, torch, torch.nn as nn, time, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.special import gamma as gamma_fn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== 配置 =====
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

N_EPOCHS_STAGE1 = 400
N_EPOCHS_STAGE2 = 800
LR = 2e-3
N_SUBJECTS = 48  # 【修改1】30 → 48

# 生理噪声频率
SYS_FREQS = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 1.0]

# ===== Auto path detection =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# 【修改2】路径修正: data/eeg_features_all_v6.npy → data/eeg_features/eeg_features_all_v6.npy
FEAT_PATH = os.path.join(REPO_ROOT, 'data', 'eeg_features_all_v6.npy')
FIG_DIR = os.path.join(REPO_ROOT, 'figures')
RES_DIR = os.path.join(REPO_ROOT, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ===== 网络 =====
class FVQNet(nn.Module):
    def __init__(self, hidden=[20,40,40,20]):
        super().__init__()
        layers = []; prev = 1
        for h in hidden:
            layers.append(nn.Linear(prev, h)); prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, t):
        x = t
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net)-1: x = nn.functional.softplus(x)
        f = nn.functional.softplus(x[:,0:1])
        v = nn.functional.softplus(x[:,1:2])
        q = 1.0 + x[:,2:3]
        return f, v, q

class FinNet(nn.Module):
    def __init__(self, hidden=[16,16]):
        super().__init__()
        layers = []; prev = 1
        for h in hidden:
            layers.append(nn.Linear(prev, h)); layers.append(nn.Softplus()); prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, t):
        return nn.functional.softplus(self.net(t))

class SystemicNoise(nn.Module):
    def __init__(self, freqs, device):
        super().__init__()
        n_freq = len(freqs)
        self.cos_coeffs = nn.Parameter(torch.zeros(n_freq, device=device))
        self.sin_coeffs = nn.Parameter(torch.zeros(n_freq, device=device))
        self.bias = nn.Parameter(torch.tensor(0.0, device=device))
        self.drift = nn.Parameter(torch.tensor(0.0, device=device))
        freq_t = torch.tensor(freqs, dtype=torch.float32, device=device)
        self.register_buffer('freq_t', freq_t)
    def forward(self, t):
        angles = 2 * np.pi * t * self.freq_t.unsqueeze(0)
        cos_part = torch.cos(angles) @ self.cos_coeffs.unsqueeze(1)
        sin_part = torch.sin(angles) @ self.sin_coeffs.unsqueeze(1)
        return cos_part + sin_part + self.bias + self.drift * t

def make_hrf_prior(t_np, onset=8.0, dur=8.0):
    f_in = np.zeros_like(t_np)
    mask = (t_np >= onset) & (t_np < onset+dur)
    dt = t_np[1] - t_np[0]
    box = np.where(mask, 1.0, 0.0)
    t_krn = np.arange(0, 8, dt)
    krn = (t_krn**(2.5-1)) * (1.5**2.5) * np.exp(-1.5*t_krn) / gamma_fn(2.5)
    krn = krn / krn.sum()
    conv = np.convolve(box, krn, mode='same')
    conv = conv / (conv.max() + 1e-10)
    f_in = conv * 0.5
    return f_in

def bw_residual(t, f_in, f, v, q, tau, eps):
    ones = torch.ones_like(t)
    df = torch.autograd.grad(f, t, ones, create_graph=True)[0]
    dv = torch.autograd.grad(v, t, ones, create_graph=True)[0]
    dq = torch.autograd.grad(q, t, ones, create_graph=True)[0]
    f_s = torch.clamp(f, min=SAFE); v_s = torch.clamp(v, min=SAFE); eps_s = torch.clamp(eps, min=SAFE)
    inv_f = 1.0/f_s
    e_arg = torch.clamp(1.0-(1.0/eps_s)*(1.0-inv_f), min=-0.99)
    E_t = E0/(1.0+E0*e_arg)
    rf = df - (f_in - (f-1)/tau)
    rv = dv - (f - v_s**(1/ALPHA))/tau
    rq = dq - (f*E_t - v_s**(1/ALPHA)*q/v_s)/tau
    return rf, rv, rq

def train_decomposition_v4(t_np, hbo_obs_np, device=DEVICE):
    """两阶段训练"""
    set_seed(SEED)

    t_t = torch.tensor(t_np, dtype=torch.float32, device=device).unsqueeze(1)
    h_t = torch.tensor(hbo_obs_np, dtype=torch.float32, device=device).unsqueeze(1)

    # 时段mask
    rest_mask_np = (t_np < 7.5) | ((t_np > 16.5) & (t_np < 23.5))
    task_mask_np = ((t_np >= 8) & (t_np < 16)) | ((t_np >= 24) & (t_np < 32))
    rest_idx = torch.tensor(rest_mask_np, dtype=torch.bool, device=device)
    task_idx = torch.tensor(task_mask_np, dtype=torch.bool, device=device)

    # ===== Stage 1: 仅用静息期拟合噪声模型 =====
    noise_s1 = SystemicNoise(SYS_FREQS, device).to(device)
    opt1 = torch.optim.Adam(noise_s1.parameters(), lr=LR)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=N_EPOCHS_STAGE1, eta_min=1e-5)

    best_loss1 = float('inf'); best_noise_state = None
    for ep in range(N_EPOCHS_STAGE1):
        opt1.zero_grad()
        noise_pred = noise_s1(t_t)
        loss = torch.mean((noise_pred[rest_idx] - h_t[rest_idx])**2)
        loss += torch.mean(noise_pred**2) * 0.001
        loss += noise_s1.drift**2 * 0.01
        if torch.isnan(loss):
            opt1.zero_grad(); continue
        loss.backward()
        opt1.step(); sched1.step()
        if loss.item() < best_loss1:
            best_loss1 = loss.item()
            best_noise_state = {k:v.clone() for k,v in noise_s1.state_dict().items()}

    noise_s1.load_state_dict(best_noise_state)
    for p in noise_s1.parameters():
        p.requires_grad = False

    # ===== Stage 2: 固定噪声频率系数, 全时段拟合f_in + B-W =====
    t_t.requires_grad_(True)
    fvq = FVQNet().to(device)
    fin = FinNet().to(device)
    noise_bias = nn.Parameter(torch.tensor(best_noise_state['bias'].item(), device=device))
    noise_drift = nn.Parameter(torch.tensor(best_noise_state['drift'].item(), device=device))

    tau_raw = nn.Parameter(torch.tensor(0.0, device=device))
    eps_raw = nn.Parameter(torch.tensor(0.0, device=device))

    hrf_prior = make_hrf_prior(t_np)
    hrf_t = torch.tensor(hrf_prior, dtype=torch.float32, device=device).unsqueeze(1)

    params = (list(fvq.parameters()) + list(fin.parameters()) +
              [tau_raw, eps_raw, noise_bias, noise_drift])
    opt2 = torch.optim.Adam(params, lr=LR)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_EPOCHS_STAGE2, eta_min=1e-5)

    best_loss2 = float('inf'); best_state = None

    rest_idx_2d = rest_idx.unsqueeze(1)

    for ep in range(N_EPOCHS_STAGE2):
        opt2.zero_grad()
        tau = 0.5 + 5.5 * torch.sigmoid(tau_raw)
        eps = 0.1 + 1.9 * torch.sigmoid(eps_raw)

        f_in = fin(t_t)
        f, v, q = fvq(t_t)
        hbo_neural = v - q

        with torch.no_grad():
            angles = 2 * np.pi * t_t * noise_s1.freq_t.unsqueeze(0)
            noise_fixed = (torch.cos(angles) @ noise_s1.cos_coeffs.unsqueeze(1) +
                          torch.sin(angles) @ noise_s1.sin_coeffs.unsqueeze(1))

        # c/d 仅从静息期数据获取梯度
        noise_for_recon = noise_fixed + noise_bias.detach() + noise_drift.detach() * t_t
        hbo_pred = hbo_neural + noise_for_recon
        loss_data = torch.mean((hbo_pred - h_t)**2)

        noise_rest_cd = noise_fixed.detach()[rest_idx_2d] + noise_bias + noise_drift * t_t[rest_idx_2d]
        residual_rest = h_t[rest_idx_2d] - hbo_neural[rest_idx_2d].detach()
        loss_cd_rest = torch.mean((noise_rest_cd - residual_rest)**2)

        rf, rv, rq = bw_residual(t_t, f_in, f, v, q, tau, eps)
        loss_eq = torch.mean(rf**2) + torch.mean(rv**2) + torch.mean(rq**2)

        t0 = torch.tensor([[0.0]], device=device)
        with torch.enable_grad():
            t0.requires_grad_(True)
            f0, v0, q0 = fvq(t0)
            f_in0 = fin(t0)
        loss_ic = ((f0-1)**2 + (v0-1)**2 + (q0-1)**2 + f_in0**2).mean()

        f_in_rest = f_in[rest_idx.unsqueeze(1)]
        loss_f_rest = torch.mean(f_in_rest**2) * 2.0

        if ep < 400:
            loss_hrf = torch.mean((f_in - hrf_t)**2) * 1.0
        else:
            loss_hrf = torch.tensor(0.0, device=device)

        hbo_neural_task = hbo_neural[task_idx.unsqueeze(1)]
        neural_task_var = torch.var(hbo_neural_task)
        loss_neural_var = torch.exp(-neural_task_var * 5.0) * 0.5

        loss = (loss_data + 1.0*loss_eq + 0.5*loss_ic + loss_f_rest +
                loss_hrf + loss_neural_var + loss_cd_rest)

        if torch.isnan(loss):
            opt2.zero_grad(); continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt2.step(); sched2.step()

        if loss.item() < best_loss2:
            best_loss2 = loss.item()
            best_state = {
                'fvq': {k:v.clone() for k,v in fvq.state_dict().items()},
                'fin': {k:v.clone() for k,v in fin.state_dict().items()},
                'tau': tau.item(), 'eps': eps.item(),
                'noise_bias': noise_bias.item(), 'noise_drift': noise_drift.item()
            }

    fvq.load_state_dict(best_state['fvq']); fvq.eval()
    fin.load_state_dict(best_state['fin']); fin.eval()
    with torch.no_grad():
        t_t.requires_grad_(False)
        f, v, q = fvq(t_t)
        hbo_neural_pred = (v - q).cpu().numpy().flatten()
        f_in_pred = fin(t_t).cpu().numpy().flatten()
        angles_np = 2 * np.pi * t_ds[:,None] * np.array(SYS_FREQS)[None,:]
        cos_c = best_noise_state['cos_coeffs'].cpu().numpy()
        sin_c = best_noise_state['sin_coeffs'].cpu().numpy()
        noise_final = (np.cos(angles_np) @ cos_c + np.sin(angles_np) @ sin_c +
                      best_state['noise_bias'] + best_state['noise_drift'] * t_ds)

    return hbo_neural_pred, noise_final, f_in_pred, best_state['tau'], best_state['eps']

# ===== 加载数据 =====
print("加载v6特征数据...")
feat = np.load(FEAT_PATH, allow_pickle=True).item()
trials = feat['trials']
params_dict = feat['params']
print(f"  总trial数: {params_dict['n_total_trials']} (右手{params_dict['n_right']}, 左手{params_dict['n_left']})")

subjects = sorted(set(t['subject'] for t in trials))
if len(subjects) > N_SUBJECTS:
    selected_subj = list(np.random.choice(subjects, N_SUBJECTS, replace=False))
else:
    selected_subj = subjects
selected_trials = [t for t in trials if t['subject'] in selected_subj]
print(f"  使用 {len(selected_subj)} 个被试, {len(selected_trials)} 个trials")

N_PTS = 80
t_ds = np.linspace(0, 32, N_PTS)

# ===== 主实验 =====
print(f"\n开始训练(v4 两阶段): {len(selected_trials)} trials (seed={SEED})")
print("-"*70)

neural_me_values = []; neural_rest_values = []
noise_me_values = []; noise_rest_values = []
raw_me_values = []; raw_rest_values = []
neural_mi_values = []; neural_rest2_values = []
noise_mi_values = []; noise_rest2_values = []
tau_list = []; eps_list = []
data_fit_r2_list = []
last_result = None

t0_total = time.time()
for i, tr in enumerate(selected_trials):
    ti = time.time()
    t_orig = tr['t']
    hbo_orig = tr['hbo_n']
    hbo_ds = np.interp(t_ds, t_orig, hbo_orig)

    rest_mask = t_ds < 8
    bl_mean = np.mean(hbo_ds[rest_mask])
    bl_std = np.std(hbo_ds[rest_mask]) + 1e-10
    hbo_norm = (hbo_ds - bl_mean) / bl_std

    hbo_neural, noise_pred, f_in_pred, tau_est, eps_est = train_decomposition_v4(t_ds, hbo_norm)

    neural_bl = np.mean(hbo_neural[rest_mask])
    noise_bl = np.mean(noise_pred[rest_mask])
    neural_norm = hbo_neural - neural_bl
    noise_norm = noise_pred - noise_bl

    hbo_recon = hbo_neural + noise_pred
    fit_r2 = 1 - np.sum((hbo_norm - hbo_recon)**2) / max(np.sum((hbo_norm - np.mean(hbo_norm))**2), 1e-10)

    me_mask = (t_ds >= 8) & (t_ds < 16)
    rest1_mask = (t_ds >= 2) & (t_ds < 7)
    mi_mask = (t_ds >= 24) & (t_ds < 32)
    rest2_mask = (t_ds >= 17) & (t_ds < 23)

    neural_me_values.append(np.mean(neural_norm[me_mask]))
    neural_rest_values.append(np.mean(neural_norm[rest1_mask]))
    noise_me_values.append(np.mean(noise_norm[me_mask]))
    noise_rest_values.append(np.mean(noise_norm[rest1_mask]))
    raw_me_values.append(np.mean(hbo_norm[me_mask]))
    raw_rest_values.append(np.mean(hbo_norm[rest1_mask]))
    neural_mi_values.append(np.mean(neural_norm[mi_mask]))
    neural_rest2_values.append(np.mean(neural_norm[rest2_mask]))
    noise_mi_values.append(np.mean(noise_norm[mi_mask]))
    noise_rest2_values.append(np.mean(noise_norm[rest2_mask]))
    tau_list.append(tau_est); eps_list.append(eps_est)
    data_fit_r2_list.append(fit_r2)
    last_result = (hbo_norm, neural_norm, noise_norm, f_in_pred, tr['hand'], tr['subject'])

    dt_run = time.time() - ti
    if (i+1) % 20 == 0 or i == 0:
        print(f"  [{i+1:3d}/{len(selected_trials)}] neural_ME={neural_me_values[-1]:.3f} "
              f"noise_ME={noise_me_values[-1]:.3f} raw_ME={raw_me_values[-1]:.3f} "
              f"fitR²={fit_r2:.3f} ({dt_run:.1f}s)", flush=True)

total_time = time.time() - t0_total
print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

# ===== 统计 =====
n_me = np.array(neural_me_values); n_rest = np.array(neural_rest_values)
ns_me = np.array(noise_me_values); ns_rest = np.array(noise_rest_values)
r_me = np.array(raw_me_values); r_rest = np.array(raw_rest_values)
n_mi = np.array(neural_mi_values); n_rest2 = np.array(neural_rest2_values)
ns_mi = np.array(noise_mi_values); ns_rest2 = np.array(noise_rest2_values)

print("\n" + "="*70)
print(f"结果统计 (v4 两阶段, seed={SEED})")
print("="*70)

t_neural, p_neural = stats.ttest_rel(n_me, n_rest)
t_noise, p_noise = stats.ttest_rel(ns_me, ns_rest)
t_raw, p_raw = stats.ttest_rel(r_me, r_rest)
t_nmi, p_nmi = stats.ttest_rel(n_mi, n_rest2)
t_nsmi, p_nsmi = stats.ttest_rel(ns_mi, ns_rest2)

def cohens_d(a, b):
    diff = a - b
    return np.mean(diff) / (np.std(diff) + 1e-10)

print(f"\n--- ME期 (8-16s) vs Rest (2-7s) ---")
print(f"神经: ME={np.mean(n_me):.4f} Rest={np.mean(n_rest):.4f} "
      f"diff={np.mean(n_me-n_rest):.4f} t={t_neural:.3f} p={p_neural:.2e} d={cohens_d(n_me,n_rest):.3f}")
print(f"噪声: ME={np.mean(ns_me):.4f} Rest={np.mean(ns_rest):.4f} "
      f"diff={np.mean(ns_me-ns_rest):.4f} t={t_noise:.3f} p={p_noise:.2e}")
print(f"原始: ME={np.mean(r_me):.4f} Rest={np.mean(r_rest):.4f} "
      f"diff={np.mean(r_me-r_rest):.4f} t={t_raw:.3f} p={p_raw:.2e} d={cohens_d(r_me,r_rest):.3f}")

print(f"\n--- MI期 (24-32s) vs Rest2 (17-23s) ---")
print(f"神经: MI={np.mean(n_mi):.4f} Rest2={np.mean(n_rest2):.4f} "
      f"diff={np.mean(n_mi-n_rest2):.4f} t={t_nmi:.3f} p={p_nmi:.2e}")
print(f"噪声: MI={np.mean(ns_mi):.4f} Rest2={np.mean(ns_rest2):.4f} "
      f"diff={np.mean(ns_mi-ns_rest2):.4f} t={t_nsmi:.3f} p={p_nsmi:.2e}")

print(f"\n拟合R²: {np.mean(data_fit_r2_list):.3f} +/- {np.std(data_fit_r2_list):.3f}")
print(f"tau: {np.mean(tau_list):.2f} +/- {np.std(tau_list):.2f}")
print(f"eps: {np.mean(eps_list):.2f} +/- {np.std(eps_list):.2f}")

# ===== 判断 =====
print("\n" + "="*70)
print("综合判断")
print("="*70)

checks = []
c1 = p_neural < 0.05 and np.mean(n_me) > np.mean(n_rest)
checks.append(("神经ME期显著正偏移", c1, f"p={p_neural:.2e}, d={cohens_d(n_me,n_rest):.3f}"))
c2 = p_noise > 0.05
checks.append(("噪声不含任务信息", c2, f"p={p_noise:.2e}"))
neural_snr = abs(np.mean(n_me-n_rest)) / (np.std(n_me-n_rest)+1e-10)
raw_snr = abs(np.mean(r_me-r_rest)) / (np.std(r_me-r_rest)+1e-10)
c3 = neural_snr >= raw_snr * 0.7
checks.append(("保留激活SNR", c3, f"neural={neural_snr:.2f} vs raw={raw_snr:.2f}"))
c4 = p_nmi < 0.1
checks.append(("神经MI期显著/边缘显著", c4, f"p={p_nmi:.2e}"))

n_pass = sum(1 for c in checks if c[1])
for name, passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name} ({detail})")
print(f"\n通过: {n_pass}/4")

if n_pass >= 3:
    verdict = "方向在真实数据上成立! 分解方法有效,可以继续推进。"
elif n_pass >= 2:
    verdict = "方向有希望,但分解效果有限,需要进一步优化。"
else:
    verdict = "两阶段策略仍不理想,需要考虑方法层面的重大调整。"
print(f"\n结论: {verdict}")

# ===== 画图 =====
print("\n画图...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

if last_result:
    hbo_raw, neural, noise, f_in_pred, hand, subj = last_result
    ax = axes[0,0]
    ax.plot(t_ds, hbo_raw, 'k-', alpha=0.4, label='Raw HbO', linewidth=0.8)
    ax.plot(t_ds, neural, 'r-', linewidth=2, label='Decomposed Neural')
    ax.plot(t_ds, noise, 'b-', linewidth=1.5, alpha=0.7, label='Decomposed Noise')
    ax.axvspan(8, 16, alpha=0.1, color='red'); ax.axvspan(24, 32, alpha=0.1, color='orange')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('HbO (normalized)')
    ax.set_title(f'Single Trial (Subj {subj}, {hand})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(t_ds, f_in_pred, 'g-', linewidth=2, label='Estimated f_in')
    ax.axvspan(8, 16, alpha=0.1, color='red'); ax.axvspan(24, 32, alpha=0.1, color='orange')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('f_in')
    ax.set_title('Estimated Neural Input (f_in)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0,2]
x_pos = [0, 1, 2]
means = [np.mean(n_me-n_rest), np.mean(r_me-r_rest), np.mean(ns_me-ns_rest)]
sems = [stats.sem(n_me-n_rest), stats.sem(r_me-r_rest), stats.sem(ns_me-ns_rest)]
labels = [f'Neural\np={p_neural:.1e}', f'Raw HbO\np={p_raw:.1e}', f'Noise\np={p_noise:.1e}']
ax.bar(x_pos, means, yerr=sems, color=['#E53935','#666666','#42A5F5'], alpha=0.7, capsize=5)
ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('ME - Rest'); ax.set_title('ME Period Activation')
ax.axhline(0, color='black', linewidth=0.5); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1,0]
bp = ax.boxplot([n_rest, n_me, n_rest2, n_mi],
                labels=['Rest1\n(2-7s)','ME\n(8-16s)','Rest2\n(17-23s)','MI\n(24-32s)'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#E8F5E9','#FFCDD2','#E8F5E9','#FFE0B2']):
    patch.set_facecolor(color)
ax.set_ylabel('Neural HbO'); ax.set_title('Neural Signal by Epoch')
ax.axhline(0, color='gray', linestyle='--'); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1,1]
bp = ax.boxplot([ns_rest, ns_me, ns_rest2, ns_mi],
                labels=['Rest1\n(2-7s)','ME\n(8-16s)','Rest2\n(17-23s)','MI\n(24-32s)'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#BBDEFB')
ax.set_ylabel('Noise'); ax.set_title('Noise by Epoch')
ax.axhline(0, color='gray', linestyle='--'); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1,2]
ax.scatter(tau_list, eps_list, alpha=0.4, s=30, c='#9C27B0')
ax.set_xlabel('tau (s)'); ax.set_ylabel('eps')
ax.set_title(f'Parameters (n={len(tau_list)})'); ax.grid(True, alpha=0.3)

plt.suptitle(f'PINN-BW v4 Two-Stage Decomposition ({len(selected_trials)} trials, seed={SEED})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, f'rq1_real_data_decomposition_v4_seed{SEED}.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {fig_path}")

# ===== 保存结果到文件 =====
import json
result_data = {
    'seed': SEED,
    'n_subjects': len(selected_subj),
    'n_trials': len(selected_trials),
    'fit_r2_mean': float(np.mean(data_fit_r2_list)),
    'fit_r2_std': float(np.std(data_fit_r2_list)),
    'tau_mean': float(np.mean(tau_list)),
    'tau_std': float(np.std(tau_list)),
    'eps_mean': float(np.mean(eps_list)),
    'eps_std': float(np.std(eps_list)),
    'neural_me_d': float(cohens_d(n_me, n_rest)),
    'neural_me_p': float(p_neural),
    'neural_mi_d': float(cohens_d(n_mi, n_rest2)),
    'neural_mi_p': float(p_nmi),
}
result_path = os.path.join(RES_DIR, f'decomposition_results_seed{SEED}.json')
with open(result_path, 'w') as f:
    json.dump(result_data, f, indent=2)
print(f"结果已保存: {result_path}")

# 保存per-trial tau/epsilon参数
param_path = os.path.join(RES_DIR, f'per_trial_params_seed{SEED}.npy')
np.save(param_path, {
    'tau_list': np.array(tau_list),
    'eps_list': np.array(eps_list),
    'seed': SEED
})
print(f"Per-trial参数已保存: {param_path}")

print("\n" + "="*70)
print(verdict)
print("="*70)