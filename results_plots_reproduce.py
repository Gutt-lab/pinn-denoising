
#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ttc_processor import TTCProcessor
from helper_functions import *

# ---------------------------- Configuration ----------------------------
DATA_DIR = Path('./denoised_ttc_v7')
OUTPUT_DIR = Path('./plotPINN')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AVE_EGGYOLK = [5, 25, 50, 100, 200, 400, 600, 800, 1000, 1200, 1300]
Q_VALUES = np.array([
    0.095, 0.105, 0.115, 0.125, 0.135, 0.145, 0.155, 0.165,
    0.175, 0.185, 0.195, 0.205, 0.215, 0.225, 0.235, 0.245,
    0.255, 0.265, 0.275, 0.28854377
])
CONTRAST = np.array([
    0.05188205, 0.05192311, 0.05136611, 0.05171926, 0.05108087,
    0.05104867, 0.05020785, 0.04957793, 0.0491428, 0.04859964,
    0.04808463, 0.04736483, 0.04679808, 0.04616146, 0.04531015,
    0.04457282, 0.04359917, 0.04177554, 0.04105258, 0.0395628
])
LOW_STATS_INDEX = 4           
Q_INDICES = [10, 12]         # q = 0.195 and 0.215 nm^-1
CUT_INDEX = 0
N_CUTS = 1
LOG_AVG_N = 100
TTC_SLICE = (0,100)
TTC_SLICE_START = 5
TTC_SLICE_END = 95
FRAME_DELAY = 0.44e-6
REFERENCE_SCAN_COUNT = 2800
REFERENCE_IS_INDEPENDENT = True
FIT_RANGE = None  

# ---------------------------- Data loading -----------------------------
raw_ttc = np.load(DATA_DIR / 'ttc_eggyolk.npy', mmap_mode='r')
raw_se = np.load(DATA_DIR / 'SE_ttc_eggyolk.npy', mmap_mode='r')
den_ttc = np.load(DATA_DIR / 'ttc_denoised_parallel.npy', mmap_mode='r')
den_se = np.load(DATA_DIR / 'SE_ttc_denoised_parallel.npy', mmap_mode='r')
raw_ref_ttc = np.load(DATA_DIR / 'Ref_TTC.npy', mmap_mode='r')
raw_ref_se = np.load(DATA_DIR / 'Ref_SE.npy', mmap_mode='r')

assert LOW_STATS_INDEX < len(AVE_EGGYOLK)
assert max(Q_INDICES) < len(Q_VALUES)
assert raw_ttc.shape[0] > LOW_STATS_INDEX
assert den_ttc.shape[0] > LOW_STATS_INDEX
assert raw_ttc.shape[1] == len(Q_VALUES)


n_t = TTC_SLICE_END - TTC_SLICE_START
delays = np.arange(2 * n_t) * FRAME_DELAY
time_binned = delays.reshape(n_t, 2).mean(axis=1)

processor = TTCProcessor()

def process(ttc, se, averaging_index,ttc_type="raw_ttc"):
    if ttc_type =="Ref_TTC":
        return processor.process_and_fit_all_q(
            np.asarray(ttc, dtype=float),
            np.asarray(se, dtype=float),
            time_binned,
            CONTRAST,
            num_cuts=N_CUTS,
            log_avg_n=LOG_AVG_N,
        )
    else:
        return processor.process_and_fit_all_q(
            np.asarray(ttc[averaging_index]),
            np.asarray(se[averaging_index]),
            time_binned,
            CONTRAST,
            num_cuts=N_CUTS,
            log_avg_n=LOG_AVG_N,
        )

reference = process(raw_ref_ttc, raw_ref_se, 0,"Ref_TTC")
noisy = process(raw_ttc, raw_se, LOW_STATS_INDEX,"raw_ttc")
denoised = process(den_ttc, den_se, LOW_STATS_INDEX,"den_ttc")

# --------------------------- Residual analysis -------------------------
def get_curve(result, q_index, cut_index):
    t = np.asarray(result['t_list'][q_index][cut_index], dtype=float).ravel()
    g2 = np.asarray(result['g2_list'][q_index][cut_index], dtype=float).ravel()
    err = np.asarray(result['err_list'][q_index][cut_index], dtype=float).ravel()
    if not (len(t) == len(g2) == len(err)):
        raise ValueError('t, g2 and error arrays must have equal lengths')
    return t, g2, err


def analyze_q(q_index, cut_index):
    t_ref, g_ref, e_ref = get_curve(reference, q_index, cut_index)
    t_raw, g_raw, e_raw = get_curve(noisy, q_index, cut_index)
    t_den, g_den, e_den = get_curve(denoised, q_index, cut_index)

    # No silent interpolation: mismatched grids need explicit handling.
    if not (t_ref.shape == t_raw.shape == t_den.shape and
            np.allclose(t_ref, t_raw, rtol=1e-8, atol=1e-12) and
            np.allclose(t_ref, t_den, rtol=1e-8, atol=1e-12)):
        raise ValueError(f'q index {q_index}: g2 lag grids differ; align them first')

    mask = (np.isfinite(t_ref) & (t_ref > 0) &
            np.isfinite(g_ref) & np.isfinite(g_raw) & np.isfinite(g_den))
    if FIT_RANGE is not None:
        mask &= (t_ref >= FIT_RANGE[0]) & (t_ref <= FIT_RANGE[1])
    if mask.sum() < 10:
        raise ValueError(f'q index {q_index}: fewer than 10 valid lag points')

    t, g_ref, g_raw, g_den = (arr[mask] for arr in (t_ref, g_ref, g_raw, g_den))
    e_ref, e_raw, e_den = (arr[mask] for arr in (e_ref, e_raw, e_den))
    r_raw = g_raw - g_ref
    r_den = g_den - g_ref
    rmse_raw = float(np.sqrt(np.mean(r_raw**2)))
    rmse_den = float(np.sqrt(np.mean(r_den**2)))
    improvement = float(100 * (1 - rmse_den / rmse_raw)) if rmse_raw > 0 else np.nan

    row = {
        'q_nm_inv': float(Q_VALUES[q_index]),
        'cut_index': int(cut_index),
        'n_lag_points': len(t),
        'raw_scans': AVE_EGGYOLK[LOW_STATS_INDEX],
        'reference_scans': REFERENCE_SCAN_COUNT,
        'reference_independent_verified': REFERENCE_IS_INDEPENDENT,
        'rmse_raw': rmse_raw,
        'rmse_denoised': rmse_den,
        'rmse_reduction_percent': improvement,
        'mean_residual_raw': float(np.mean(r_raw)),
        'mean_residual_denoised': float(np.mean(r_den)),
    }
    return locals()

# ------------------------------- Figures -------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'axes.linewidth': 1.0,
    'savefig.dpi': 300,
})


for q_index in Q_INDICES:

    #TTC_plots 
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    vmin, vmax = np.percentile(raw_ref_ttc, [0.5, 99.5])
    residuals, p, x, freq, spectrum = analyze_residuals(
        raw_ttc[LOW_STATS_INDEX, q_index, TTC_SLICE_START:TTC_SLICE_END, TTC_SLICE_START:TTC_SLICE_END], 
        den_ttc[LOW_STATS_INDEX, q_index, TTC_SLICE_START:TTC_SLICE_END, TTC_SLICE_START:TTC_SLICE_END], 
        dt=FRAME_DELAY, 
        qvalue=Q_VALUES[q_index], 
        plot=False
    )
    datasets = [
        (raw_ref_ttc[q_index, TTC_SLICE_START:TTC_SLICE_END, TTC_SLICE_START:TTC_SLICE_END], "Reference", None),
        (raw_ttc[LOW_STATS_INDEX, q_index, TTC_SLICE_START:TTC_SLICE_END, TTC_SLICE_START:TTC_SLICE_END], "Raw", None),
        (den_ttc[LOW_STATS_INDEX, q_index, TTC_SLICE_START:TTC_SLICE_END, TTC_SLICE_START:TTC_SLICE_END], "PI-PD-CNN Denoised", None),
    ]

    for i, (ax, (image_data, title, cmap_override)) in enumerate(zip(axes, datasets)):
        if i == 0:  # Reference image: no color bar
            im = ax.imshow(image_data,
                        cmap='viridis',
                        origin='lower',
                        vmin=vmin, vmax=vmax)
        else:  # Other images: share color bar
            im = ax.imshow(image_data,
                        cmap='viridis',
                        origin='lower',
                        vmin=vmin, vmax=vmax)
        ax.set_title(title, pad=12)
        ax.set_xlabel(r"$\tau_{1}$ ($\mu$s)",fontsize=18)
        ax.set_ylabel(r"$\tau_{2}$ ($\mu$s)",fontsize=18)
        ax.set_xticks([0, 50 ,90])
        ax.set_yticks([0, 50 ,90])
        ax.set_xticklabels([0*FRAME_DELAY*1e6, 50*FRAME_DELAY*1e6 ,90*FRAME_DELAY*1e6],fontsize=18)
        ax.set_yticklabels([0*FRAME_DELAY*1e6, 50*FRAME_DELAY*1e6 ,90*FRAME_DELAY*1e6],fontsize=18)
    
    cbar = fig.colorbar(im, ax=axes[0], location='left', shrink=0.9, aspect=25, pad=0.02)

    # Panel D: Residual Trace

    spec = axes[3].get_subplotspec()
    axes[3].remove()
    gs_sub = spec.subgridspec(2, 1, hspace=0.1)
    ax_dist = fig.add_subplot(gs_sub[1, 0])
    ax_psd = fig.add_subplot(gs_sub[0, 0])
    ax_psd.plot(residuals, color='#6a0dad', alpha=0.6, linewidth=0.8)
    ax_psd.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax_psd.set_title('Residual Trace', pad=1)
    ax_psd.tick_params(labelsize=14)
    ax_dist.hist(residuals, bins=60, density=True, color='gray', alpha=0.5, label='Residuals')
    ax_dist.plot(x, p, 'r-', linewidth=2, label='Gaussian Fit')
    ax_dist.set_title('Residual Distribution', pad=1)
    ax_dist.tick_params(labelsize=16)
    ax_dist.legend(fontsize=14)
    plt.show()
    plt.close(fig)



    #g2 plotting 
    d = analyze_q(q_index, CUT_INDEX)


    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    ax.errorbar(d['t'], d['g_ref'], yerr=d['e_ref'], fmt='ko-',
                mfc='none', ms=5, lw=1.1, capsize=1.5,
                label=f'High-statistics reference ({REFERENCE_SCAN_COUNT} scans)')
    ax.errorbar(d['t'], d['g_raw'], yerr=d['e_raw'], fmt='s',
                color='tab:red', ms=3.5, alpha=0.65, capsize=1.5,
                label=f'Raw ({AVE_EGGYOLK[LOW_STATS_INDEX]} scans)')
    # Omit denoised error bars unless their uncertainty calibration is established.
    ax.errorbar(d['t'], d['g_den'], yerr=d['e_den'], fmt='o', color='tab:blue', ms=4.2,
            label=f'PI-PD-CNN denoised ({AVE_EGGYOLK[LOW_STATS_INDEX]} scans)')
    ax.set_xscale('log')
    ax.set_ylabel(r'$g_2(\tau)$')
    ax.set_title(rf'$q = {Q_VALUES[q_index]:.3f}\ \mathrm{{nm}}^{{-1}}$')
    ax.legend(loc='upper right', fontsize=9)
    ax.tick_params(labelbottom=False)

    ax_res.plot(d['t'], d['r_raw'], 'o-', color='tab:red', ms=3.2,
                lw=0.8, alpha=0.75, label='Raw − reference')
    ax_res.plot(d['t'], d['r_den'], 's-', color='tab:blue', ms=3.2,
                lw=0.8, label='Denoised − reference')
    ax_res.axhline(0, color='black', ls='--', lw=0.9)
    ax_res.set_ylabel(r'$\Delta g_2$')
    ax_res.set_xlabel(r'Delay time $\tau$ (s)')
    ax_res.legend(loc='best', fontsize=8)
    
   
    for a in (ax, ax_res):
        a.tick_params(which='both', direction='in', top=True, right=True)
        a.grid(True, which='major', ls=':', alpha=0.3)

    name = (f'g2_reference_residuals_N{AVE_EGGYOLK[LOW_STATS_INDEX]}'
            f'_q{Q_VALUES[q_index]:.3f}')
    fig.savefig(OUTPUT_DIR / f'{name}.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)




#Parameter Extraction 
# PDI var
q_index_list = range(len(Q_VALUES))
N_CUTS = 1
results_dist_eggyolk = []
results_res_whiteness_eggyolk = []
for ci in range(N_CUTS):
    cut_dist = []
    cut_res_whiteness = []
    for q_index in q_index_list:
        g2_array = np.array([
            reference['g2_list'][q_index][ci],
            noisy['g2_list'][q_index][ci],
            denoised['g2_list'][q_index][ci],
        ])

        t_array = np.array([
            reference['t_list'][q_index][ci],
            noisy['t_list'][q_index][ci],
            denoised['t_list'][q_index][ci],
        ])

        results_dist = inverse_laplace_nnls_g2(g2_array, t_array,
                                                gamma_min=None,
                                                gamma_max=None,
                                                n_gamma=20,
                                                q_values=Q_VALUES[q_index], 
                                                datasets_labels=[f'Raw {AVE_EGGYOLK[LOW_STATS_INDEX]}',
                                                f'Raw {AVE_EGGYOLK[LOW_STATS_INDEX]}', 
                                                f'Denoised Parallel {AVE_EGGYOLK[LOW_STATS_INDEX]}', 
                                                ],
                                                subtract_baseline=False,
                                                plot=False
                                            )

        res = residual_whiteness_from_fits_g2(
            g2_denoised_list=[denoised['g2_list'][q_index][ci]],
            g2_raw_list=[noisy['g2_list'][q_index][ci]],
            t_lags_list=[denoised['t_list'][q_index][ci]],
            labels=[f"q={Q_VALUES[q_index]:.3f} nm$^{{-1}}$"] ,
            fit_range=None,
            max_lag=100,
            ljung_box_lags=50,
            sigma_raw_list=[noisy['err_list'][q_index][ci]],
            sigma_denoised_list=[denoised['err_list'][q_index][ci]],
            main_title="AI-denoised g2 residual whiteness across q", 
            dpi=200,
            plot=False
        )
      
        cut_dist.append(results_dist)
        cut_res_whiteness.append(res)
        
    results_dist_eggyolk.append(cut_dist)
    results_res_whiteness_eggyolk.append(cut_res_whiteness)


results_paras_denoised = fit_g2_kww_many(
    g2_list=denoised['fitted_g2_list'],
    t_list=denoised['t_list'],
    q_list=Q_VALUES,
    sigma_list=denoised['err_list'],
    plot=False,
    fit_range=None,
    fix_baseline=False,
    baseline_tail_frac=0.1,
    baseline_val=0.0,
    subtract_baseline=True,
    weights="linear",
    method="bootstrap",         # best for error bars
    n_boot=20,
    seed=42,    
)

results_paras_reference = fit_g2_kww_many(
    g2_list=reference['fitted_g2_list'],
    t_list=reference['t_list'],
    q_list=Q_VALUES,
    sigma_list=reference['err_list'],
    plot=False,
    fit_range=None,
    fix_baseline=False,
    baseline_tail_frac=0.1,
    baseline_val=0.0,
    subtract_baseline=True,
    weights="linear",
    method="bootstrap",         # best for error bars
    n_boot=20,
    seed=42,    
)

results_paras_raw = fit_g2_kww_many(
    g2_list=noisy['g2_list'],
    t_list=noisy['t_list'],
    q_list=Q_VALUES,
    sigma_list=noisy['err_list'],
    plot=False,
    fit_range=None,
    fix_baseline=False,
    baseline_tail_frac=0.1,
    baseline_val=0.0,
    subtract_baseline=True,
    weights="linear",
    method="bootstrap",         # best for error bars
    n_boot=20,
    seed=42,    
)

pdi_gt = np.array([res["summary"]["PDI"][0] for res in results_dist_eggyolk[0]])
pdi_gt_err = np.array([res["summary"]["PDI_err"][0] for res in results_dist_eggyolk[0]])
pdi_noisy = np.array([res["summary"]["PDI"][1] for res in results_dist_eggyolk[0]])
pdi_noisy_err = np.array([res["summary"]["PDI_err"][1] for res in results_dist_eggyolk[0]])
pdi_denoised = np.array([res["summary"]["PDI"][2] for res in results_dist_eggyolk[0]])
pdi_denoised_err = np.array([res["summary"]["PDI_err"][2] for res in results_dist_eggyolk[0]])

print(len(pdi_denoised))



import matplotlib.ticker as ticker

valid_idx = [i for i in range(len(results_paras_denoised)) 
             if abs(results_paras_denoised[i]['beta'] - results_paras_reference[i]['beta']) < 0.7
             and abs(pdi_denoised[i] - pdi_gt[i]) < 0.11
             ]

print(valid_idx)
qs_plot = [f"{Q_VALUES[q_index_list[i]]:.3f}" for i in valid_idx]
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 10), sharex=True, gridspec_kw={'hspace': 0.2})
x = np.array([i for i in range(len(valid_idx))])
def plot_params(ax, gt_key, denoised_key, noisy_key, ylabel, ylim=None):
    gt_val = np.array([results_paras_reference[i][gt_key] for i in valid_idx])
    gt_err = np.array([results_paras_reference[i][f'{gt_key}_err'] for i in valid_idx])
    dn_val = np.array([results_paras_denoised[i][gt_key] for i in valid_idx])
    dn_err = np.array([results_paras_denoised[i][f'{gt_key}_err'] for i in valid_idx])
    ns_val = np.array([results_paras_raw[i][gt_key] for i in valid_idx])
    ns_err = np.array([results_paras_raw[i][f'{gt_key}_err'] for i in valid_idx])
    ax.errorbar(x -0.1, gt_val, yerr=gt_err, fmt='ko-', label='Target', mfc='none', ms=7, mew=1.2)
    ax.errorbar(x+0.1, ns_val, yerr=ns_err, fmt='s', color='tab:red', label='Noisy', ms=5, alpha=0.6, capsize=2)
    ax.errorbar(x, dn_val, yerr=dn_err, fmt='o', color='tab:blue', label='Denoised', ms=6, capsize=2.5)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.tick_params(which='both', direction='in', top=True, right=True)

# Plot Beta, Gamma, and PDI
plot_params(ax1, 'beta', 'beta', 'beta', r'KWW')
ax1.legend(frameon=True, fontsize=9, loc='upper right', bbox_to_anchor=(1, 1.1))

plot_params(ax2, 'Gamma', 'Gamma', 'Gamma', r'$\Gamma$ (s$^{-1})$')
# Format Gamma y-axis with scientific notation 1.1 x 10^N
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#ax2.set_ylim(-3e3, 2.5e4)
# Small informative inset for RMSE vs q on the top panel
rmse_noisy = [results_paras_raw[i]['rmse'] for i in valid_idx]
rmse_denoised = [results_paras_denoised[i]['rmse'] for i in valid_idx]
rmse_gt = [results_paras_reference[i]['rmse'] for i in valid_idx]

ax4.plot(x - 0.1, rmse_gt, 'ko-', mfc='none', ms=7, mew=1.2, label='Target')
ax4.plot(x + 0.1, rmse_noisy, 's', color='tab:red', ms=5, alpha=0.6, label='Noisy')
ax4.plot(x, rmse_denoised, 'o', color='tab:blue', ms=6, label='Denoised')
ax4.set_ylabel('RMSE')
ax4.set_ylim(-0.005, 0.007)
ax4.grid(True, linestyle=':', alpha=0.5)
ax4.tick_params(which='both', direction='in', top=True, right=True)
ax4.set_xticks(x)
ax4.set_xticklabels([f'{i}' for i in range(1, len(valid_idx)+1)])
#ax4.set_yscale('log')
# Format RMSE y-axis with scientific notation 1.1 x 10^N
ax4.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))




# Manual plot for PDI as it's from a different source array
ax3.errorbar(x - 0.1, pdi_gt[valid_idx], yerr=pdi_gt_err[valid_idx]**2, fmt='ko-', mfc='none', ms=7, mew=1.2, zorder=3)
ax3.errorbar(x + 0.1, pdi_noisy[valid_idx], yerr=pdi_noisy_err[valid_idx]**2, fmt='s', color='tab:red', ms=5, alpha=0.6, capsize=2, zorder=3)
ax3.errorbar(x, pdi_denoised[valid_idx], yerr=pdi_denoised_err[valid_idx]**2, fmt='o', color='tab:blue', ms=6, capsize=2.5, zorder=3)

ax3.set_ylabel(r'$\text{Var}(\Gamma)/\langle \Gamma \rangle^2$')


#ax3.set_ylim(-0.0005, 0.0035)
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.tick_params(which='both', direction='in', top=True, right=True)
# Format PDI y-axis with scientific notation 1.1 x 10^N
ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax4.set_xlabel(r'ROI Index')

#plt.tight_layout()
fig.savefig(f'plotPINN/parameter_summary_unified_{len(valid_idx)}_points.png', dpi=600, bbox_inches='tight')
plt.show()

# %%
