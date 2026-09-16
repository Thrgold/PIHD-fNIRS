# -*- coding: utf-8 -*-
"""
Step 11 v6: 全48被试EEG特征提取 + 自动坏通道检测
==================================================
数据结构: data['trials'] 是列表, 每个trial是dict, 含:
  hand ('right'/'left'), eeg_sfreq, fnirs_sfreq,
  fnirs_hbo_all (20, T), eeg_c3, eeg_c4, trial_number
"""
import os, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d
from scipy import stats as sstats

# ===== 路径 =====

# ===== Auto path detection =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
OUT_DIR  = os.path.join(REPO_ROOT, 'data')
FIG_DIR  = os.path.join(REPO_ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ===== 参数 =====
EEG_SFREQ   = 1000.0
FNIRS_SFREQ = 11.0
TRIAL_LEN_S = 32
REST_S      = 8
ALPHA_BAND  = [8, 13]
BETA_BAND   = [13, 30]
SMOOTH_SIGMA_S = 2.0
LP_CUTOFF_HZ   = 0.5
BAD_CH_STD_THRESH = 5.0       # 基线期std>5μM -> 坏通道
BAD_CH_FULL_STD_THRESH = 8.0  # 全trial std>8μM -> 坏通道
MIN_GOOD_CHS    = 3
DELAY_CANDIDATES = np.arange(0, 10.1, 0.5)
TRIALS_PER_HAND = 5           # 每手最多取5个trial

# 通道分组: 0-9 左半球(右手运动对侧), 10-19 右半球(左手运动对侧)
CH_L = list(range(10))
CH_R = list(range(10, 20))

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
    'axes.unicode_minus': False,
})

print("=" * 60)
print("Step 11 v6: 全被试EEG特征提取")
print("=" * 60)

def extract_eeg_envelope(eeg_raw, sfreq, band):
    nyq = sfreq / 2.0
    b, a = butter(4, [band[0]/nyq, band[1]/nyq], btype='band')
    filtered = filtfilt(b, a, eeg_raw)
    analytic = hilbert(filtered)
    env = np.abs(analytic)
    env = gaussian_filter1d(env, sigma=SMOOTH_SIGMA_S * sfreq)
    if LP_CUTOFF_HZ < nyq:
        bl, al = butter(4, LP_CUTOFF_HZ/nyq, btype='low')
        env = filtfilt(bl, al, env)
    return env

def detect_good_channels(hbo_all, n_rest, ch_indices):
    good = []; bad_info = {}
    for ch in ch_indices:
        sig = hbo_all[ch]
        rest_std = np.std(sig[:n_rest])
        full_std = np.std(sig)
        is_bad = False; reason = ''
        if rest_std > BAD_CH_STD_THRESH:
            is_bad = True; reason = f'rest_std={rest_std:.2f}'
        elif full_std > BAD_CH_FULL_STD_THRESH:
            is_bad = True; reason = f'full_std={full_std:.2f}'
        if not is_bad:
            good.append(ch)
        else:
            bad_info[ch] = reason
    return good, bad_info

def process_one_trial(trial):
    hand = trial['hand']
    hbo_all = trial['fnirs_hbo_all']
    if hand == 'right':
        ch_pool = CH_L
        eeg_raw = trial['eeg_c3']
    else:
        ch_pool = CH_R
        eeg_raw = trial['eeg_c4']
    n_rest = int(REST_S * FNIRS_SFREQ)
    n_fnirs = hbo_all.shape[1]
    good_chs, bad_info = detect_good_channels(hbo_all, n_rest, ch_pool)
    if len(good_chs) < MIN_GOOD_CHS:
        return None
    hbo = np.mean(hbo_all[good_chs], axis=0)
    hbo_bl = np.mean(hbo[:n_rest])
    hbo_c = hbo - hbo_bl
    alpha_env = extract_eeg_envelope(eeg_raw, EEG_SFREQ, ALPHA_BAND)
    beta_env  = extract_eeg_envelope(eeg_raw, EEG_SFREQ, BETA_BAND)
    eeg_env   = (alpha_env + beta_env) / 2.0
    n_eeg = len(eeg_env)
    t_eeg = np.linspace(0, TRIAL_LEN_S, n_eeg)
    t_fnirs = np.linspace(0, TRIAL_LEN_S, n_fnirs)
    eeg_ds = np.interp(t_fnirs, t_eeg, eeg_env)
    # 延迟搜索: 用不同延迟量平移EEG,找与HbO的最大负相关
    best_delay = 0; best_corr = -999
    hbo_task = hbo_c[int(10*FNIRS_SFREQ):int(22*FNIRS_SFREQ)]
    for d in DELAY_CANDIDATES:
        start = int((10-d)*FNIRS_SFREQ)
        end   = int((22-d)*FNIRS_SFREQ)
        if start < 0 or end > n_fnirs: continue
        eeg_seg = eeg_ds[start:end]
        if len(eeg_seg) != len(hbo_task): continue
        corr = -np.corrcoef(eeg_seg, hbo_task)[0,1]
        if corr > best_corr:
            best_corr = corr; best_delay = d
    return {
        'eeg_env_raw': eeg_ds, 'hbo': hbo_c, 'hand': hand,
        'delay': best_delay, 'corr': best_corr,
        'good_chs': good_chs, 'bad_chs': bad_info,
        'n_good': len(good_chs), 't': t_fnirs,
    }

# ===== 遍历所有被试 =====
subj_files = sorted(glob.glob(os.path.join(DATA_DIR, 'subject_*.npy')))
print(f"发现 {len(subj_files)} 个被试文件")

all_trials = []
quality_log = []

for fi, fpath in enumerate(subj_files):
    subj_id = os.path.basename(fpath).replace('.npy','').replace('subject_','')
    try:
        subj_data = np.load(fpath, allow_pickle=True).item()
    except Exception as e:
        print(f"  [跳过] {subj_id}: 加载失败 {e}"); continue
    if 'trials' not in subj_data:
        print(f"  [跳过] {subj_id}: 无trials字段"); continue
    trials = subj_data['trials']
    # 按手筛选,每手最多取TRIALS_PER_HAND个
    rt = [t for t in trials if t.get('hand')=='right'][:TRIALS_PER_HAND]
    lt = [t for t in trials if t.get('hand')=='left'][:TRIALS_PER_HAND]
    n_right_ok = 0; n_left_ok = 0; n_dropped = 0
    for tr in rt:
        try:
            res = process_one_trial(tr)
            if res is not None:
                res['subject'] = subj_id
                res['trial_number'] = tr.get('trial_number', -1)
                all_trials.append(res); n_right_ok += 1
            else:
                n_dropped += 1
        except Exception as e:
            print(f"  [ERROR] {subj_id} right trial: {e}"); n_dropped += 1
    for tr in lt:
        try:
            res = process_one_trial(tr)
            if res is not None:
                res['subject'] = subj_id
                res['trial_number'] = tr.get('trial_number', -1)
                all_trials.append(res); n_left_ok += 1
            else:
                n_dropped += 1
        except Exception as e:
            print(f"  [ERROR] {subj_id} left trial: {e}"); n_dropped += 1
    quality_log.append({'subject': subj_id, 'n_trials': len(trials),
        'n_right_ok': n_right_ok, 'n_left_ok': n_left_ok, 'n_dropped': n_dropped})
    if (fi+1) % 10 == 0 or fi == len(subj_files)-1:
        print(f"  [{fi+1}/{len(subj_files)}] {subj_id}: 右{n_right_ok} 左{n_left_ok} 丢{n_dropped}")

n_right = sum(1 for t in all_trials if t['hand']=='right')
n_left  = sum(1 for t in all_trials if t['hand']=='left')
print(f"\n总有效trial数: {len(all_trials)} (右手:{n_right}, 左手:{n_left})")

if len(all_trials) == 0:
    print("ERROR: 没有有效trial! 退出。")
    exit(1)

# ===== 全局归一化 =====
print("\n全局归一化...")
all_eeg = np.array([t['eeg_env_raw'] for t in all_trials])
all_hbo = np.array([t['hbo'] for t in all_trials])
ri = slice(0, int(REST_S*FNIRS_SFREQ))
grm = np.mean(all_eeg[:, ri]); grs = np.std(all_eeg[:, ri]) + 1e-10
eeg_z = (all_eeg - grm) / grs
eeg_inv = -eeg_z
g5, g95 = np.percentile(eeg_inv, 5), np.percentile(eeg_inv, 95)
all_eeg_n = np.clip((eeg_inv - g5) / (g95 - g5 + 1e-10), 0, 1)
hbo_abs_max = max(abs(np.percentile(all_hbo, 5)), abs(np.percentile(all_hbo, 95)))
all_hbo_n = np.clip(all_hbo / (hbo_abs_max + 1e-10), -1, 1)
for i, t in enumerate(all_trials):
    t['eeg_n'] = all_eeg_n[i]; t['hbo_n'] = all_hbo_n[i]

# ===== 统计 =====
delays_r = [t['delay'] for t in all_trials if t['hand']=='right']
delays_l = [t['delay'] for t in all_trials if t['hand']=='left']
all_delays = [t['delay'] for t in all_trials]
print(f"延迟统计 (秒):")
print(f"  右手: mean={np.mean(delays_r):.2f}, std={np.std(delays_r):.2f}, median={np.median(delays_r):.1f}")
print(f"  左手: mean={np.mean(delays_l):.2f}, std={np.std(delays_l):.2f}, median={np.median(delays_l):.1f}")
print(f"  总体: mean={np.mean(all_delays):.2f}, std={np.std(all_delays):.2f}")

eeg_rest = np.mean(all_eeg_n[:, int(0*FNIRS_SFREQ):int(7*FNIRS_SFREQ)], axis=1)
eeg_task = np.mean(all_eeg_n[:, int(8*FNIRS_SFREQ):int(16*FNIRS_SFREQ)], axis=1)
t_erd, p_erd = sstats.ttest_rel(eeg_task, eeg_rest)
n_pos = np.sum(eeg_task > eeg_rest)
print(f"\nEEG ERD检验: task-rest 差={np.mean(eeg_task-eeg_rest):.4f}, t={t_erd:.3f}, p={p_erd:.6f}")
print(f"  {n_pos}/{len(all_trials)} trials执行期EEG(归一化后)高于静息期")

hbo_rest_m = np.mean(all_hbo_n[:, int(0*FNIRS_SFREQ):int(7*FNIRS_SFREQ)], axis=1)
hbo_peak = np.max(all_hbo_n[:, int(10*FNIRS_SFREQ):int(22*FNIRS_SFREQ)], axis=1)
t_hbo, p_hbo = sstats.ttest_rel(hbo_peak, hbo_rest_m)
n_hbo_pos = np.sum(hbo_peak > hbo_rest_m)
print(f"HbO响应检验: peak-rest 差={np.mean(hbo_peak-hbo_rest_m):.4f}, t={t_hbo:.3f}, p={p_hbo:.8f}")
print(f"  {n_hbo_pos}/{len(all_trials)} trials HbO峰值高于静息期")

# ===== 保存 =====
save_path = os.path.join(OUT_DIR, 'eeg_features_all_v6.npy')
np.save(save_path, {
    'trials': all_trials,
    'params': {
        'eeg_sfreq': EEG_SFREQ, 'fnirs_sfreq': FNIRS_SFREQ,
        'alpha_band': ALPHA_BAND, 'beta_band': BETA_BAND,
        'smooth_sigma_s': SMOOTH_SIGMA_S, 'lp_cutoff': LP_CUTOFF_HZ,
        'bad_ch_std_thresh': BAD_CH_STD_THRESH,
        'bad_ch_full_std_thresh': BAD_CH_FULL_STD_THRESH,
        'n_total_subjects': len(subj_files),
        'n_total_trials': len(all_trials),
        'n_right': n_right, 'n_left': n_left,
        'delay_mean': float(np.mean(all_delays)),
        'delay_std': float(np.std(all_delays)),
        'erd_p': float(p_erd), 'hbo_p': float(p_hbo),
    },
    'quality_log': quality_log,
})
print(f"\n已保存: {save_path}")

# ===== 画图 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
t_axis = np.linspace(0, TRIAL_LEN_S, all_eeg_n.shape[1])
periods = [(0,8,'#E8E8E8','静息'), (8,16,'#FFE0B2','运动执行'),
           (16,24,'#E8E8E8','静息'), (24,32,'#C8E6C9','运动想象')]

def plot_ga(ax, data_r, data_l, title, ylabel, ylim=None):
    for s,e,c,_ in periods:
        ax.axvspan(s, e, color=c, alpha=0.5, zorder=0)
    mean_r = np.mean(data_r, axis=0); std_r = np.std(data_r, axis=0)/np.sqrt(data_r.shape[0])
    mean_l = np.mean(data_l, axis=0); std_l = np.std(data_l, axis=0)/np.sqrt(data_l.shape[0])
    ax.plot(t_axis, mean_r, color='#E53935', label='右手(C3/L)', linewidth=2)
    ax.fill_between(t_axis, mean_r-std_r, mean_r+std_r, color='#E53935', alpha=0.2)
    ax.plot(t_axis, mean_l, color='#1E88E5', label='左手(C4/R)', linewidth=2)
    ax.fill_between(t_axis, mean_l-std_l, mean_l+std_l, color='#1E88E5', alpha=0.2)
    ax.axvline(8, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(16, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(24, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc='upper right'); ax.set_xlim(0,32)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

eeg_r = np.array([t['eeg_n'] for t in all_trials if t['hand']=='right'])
eeg_l = np.array([t['eeg_n'] for t in all_trials if t['hand']=='left'])
hbo_r = np.array([t['hbo_n'] for t in all_trials if t['hand']=='right'])
hbo_l = np.array([t['hbo_n'] for t in all_trials if t['hand']=='left'])
hbo_raw_r = np.array([t['hbo'] for t in all_trials if t['hand']=='right'])
hbo_raw_l = np.array([t['hbo'] for t in all_trials if t['hand']=='left'])

plot_ga(axes[0,0], eeg_r, eeg_l, f'EEG包络(归一化, 反相) - 全{len(all_trials)} trials', 'EEG (a.u.)', (-0.1,1.1))
plot_ga(axes[0,1], hbo_r, hbo_l, f'HbO(归一化) - 全{len(all_trials)} trials', 'HbO (a.u.)')
plot_ga(axes[1,0], hbo_raw_r, hbo_raw_l, f'HbO(原始μM) - 全{len(all_trials)} trials', 'HbO (μM)')

ax_d = axes[1,1]
bins = np.arange(-0.25, 10.5, 0.5)
ax_d.hist(delays_r, bins=bins, alpha=0.6, color='#E53935', label=f'右手(mean={np.mean(delays_r):.1f}s)')
ax_d.hist(delays_l, bins=bins, alpha=0.6, color='#1E88E5', label=f'左手(mean={np.mean(delays_l):.1f}s)')
ax_d.axvline(np.mean(delays_r), color='#E53935', linestyle='--')
ax_d.axvline(np.mean(delays_l), color='#1E88E5', linestyle='--')
ax_d.set_xlabel('EEG->HbO延迟 (s)'); ax_d.set_ylabel('Trial数')
ax_d.set_title('最优延迟分布'); ax_d.legend(); ax_d.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'step11_v6_grand_average.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path}")

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
n_good_r = [t['n_good'] for t in all_trials if t['hand']=='right']
n_good_l = [t['n_good'] for t in all_trials if t['hand']=='left']
ax2[0].hist(n_good_r, bins=range(0,12), alpha=0.6, color='#E53935', label='右手(对侧L)')
ax2[0].hist(n_good_l, bins=range(0,12), alpha=0.6, color='#1E88E5', label='左手(对侧R)')
ax2[0].set_xlabel('好通道数'); ax2[0].set_ylabel('Trial数')
ax2[0].set_title('每trial保留的好通道数分布'); ax2[0].legend(); ax2[0].grid(True, alpha=0.3)
subj_n = [q['n_right_ok']+q['n_left_ok'] for q in quality_log]
ax2[1].bar(range(len(subj_n)), subj_n, color='#7E57C2')
ax2[1].set_xlabel('被试序号'); ax2[1].set_ylabel('有效trial数')
ax2[1].set_title(f'各被试有效trial数(共{len(subj_files)}人, 平均{np.mean(subj_n):.1f}trials/人)')
ax2[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, 'step11_v6_quality.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {fig2_path}")

print("\n" + "="*60)
print("Step11 v6 完成!")
print("="*60)
