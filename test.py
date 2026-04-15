"""
Unified Normalizing Flow model testing code
Supports configuration via JSON config file or command-line arguments

Features:
- P-P Plot: Evaluate posterior distribution calibration
- Scatter Plot: True vs Predicted comparison
- Corner Plot: Single-sample posterior distribution visualization
- Model Comparison: Multi-model performance comparison

Usage examples:
  # Using a config file (recommended; auto-infers model/data/mode/band)
  python test.py --config config_lisa_physical.json
  python test.py --config config_all_physical.json --params all

  # Command-line mode (manual specification)
  python test.py --model models/flow_concat_slope.pt --data dataset/joint/Dataset_PTA_LISA_LIGO.json

  # Single-band model testing
  python test.py --model models/single/pta_flow.pt --data dataset/single/PTA_fixed_14pts.json --mode single --band PTA

  # Config file + override selected parameters
  python test.py --config config_lisa_physical.json --num-samples 10000 --no-corner
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
import json
import argparse


# ==========================================
# Utility: tee output to both stdout and a file (used for package eval log)
# ==========================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()

# ==========================================
# 0. Style configuration
# ==========================================
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 24
})

# Parameter labels
PARAM_LABELS = [
    r"$\log_{10} r$",           # 0
    r"$n_\mathrm{t}$",          # 1
    r"$\log_{10} \kappa_{10}$", # 2
    r"$\log_{10} T_{re}$",      # 3
    r"$\Delta N_{re}$",         # 4
    r"$\Omega_b h^2$",          # 5
    r"$\Omega_c h^2$",          # 6
    r"$H_0$",                   # 7
    r"$\log(10^{10} A_s)$"      # 8
]


# ==========================================
# 1. Plotting functions
# ==========================================

def draw_pp_plot(all_true, all_pred_samples, full_labels, active_indices=None,
                 out_dir="./figures/cache", title_suffix="", max_cols=5):
    """Draw P-P plots

    Args:
        out_dir: Direct output directory (absolute or relative path).
    """
    print("Plotting P-P Plots...")

    if active_indices is None:
        active_indices = list(range(len(full_labels)))

    num_plots = len(active_indices)
    num_cols = min(max_cols, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5.5 * num_rows))
    fig.subplots_adjust(wspace=0.3, hspace=0.35)

    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]

    for plot_idx, param_idx in enumerate(active_indices):
        row = plot_idx // num_cols
        col = plot_idx % num_cols
        ax = axes[row][col]
        label = full_labels[param_idx]

        # Compute ranks
        ranks = np.mean(all_pred_samples[:, :, param_idx] < all_true[:, param_idx, None], axis=1)

        # Sort and compute theoretical CDF
        sorted_ranks = np.sort(ranks)
        N = len(sorted_ranks)
        theoretical_cdf = np.linspace(0, 1, N)

        # KS statistic
        ks_stat, _ = stats.kstest(ranks, 'uniform')

        # Confidence intervals
        ci_alphas = [0.7, 0.4, 0.2]
        ci_color = '#708090'

        for i, sigma in enumerate([1, 2, 3]):
            confidence = 0.6827 ** sigma
            lower = stats.beta.ppf((1 - confidence) / 2, theoretical_cdf * N + 1, (1 - theoretical_cdf) * N + 1)
            upper = stats.beta.ppf(1 - (1 - confidence) / 2, theoretical_cdf * N + 1, (1 - theoretical_cdf) * N + 1)
            label_ci = f'{sigma}$\\sigma$ CI' if plot_idx == 0 else None
            ax.fill_between(theoretical_cdf, lower, upper, color=ci_color, alpha=ci_alphas[i],
                           label=label_ci, zorder=1, rasterized=True)

        # Ideal line
        ax.plot([0, 1], [0, 1], linestyle='--', color='#e74c3c', linewidth=1.5,
                label='Ideal' if plot_idx == 0 else None, zorder=5)

        # Model line
        ax.plot(theoretical_cdf, sorted_ranks, color='#1f77b4', linewidth=2.0, alpha=0.85,
                label='Model' if plot_idx == 0 else None, zorder=10, rasterized=True)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"{label}", fontweight='bold', pad=10)
        ax.set_xlabel("Theoretical Probability")
        if col == 0:
            ax.set_ylabel("Empirical Probability")

        ax.text(0.05, 0.95, f"KS = {ks_stat:.3f}",
                transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9, lw=1.5),
                zorder=15)

        ax.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=0)

        if plot_idx == 0:
            ax.legend(loc='lower right', frameon=True, fontsize=12, edgecolor='#cccccc',
                     facecolor='white', framealpha=0.9)

    # Hide unused subplots
    for idx in range(num_plots, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row][col].set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"PP_Plot{title_suffix.replace(' ', '_')}.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"P-P plot saved to {save_path}")
    plt.close(fig)


def draw_scatter_plot(all_true, all_pred_samples, full_labels, active_indices=None,
                      out_dir="./figures/cache", title_suffix="", max_cols=5):
    """Draw True vs Predicted scatter plots"""
    print("Plotting Scatter Plots...")

    if active_indices is None:
        active_indices = list(range(len(full_labels)))

    # Use posterior median as the point estimate, consistent with evaluate_model's metrics
    # (previously scatter plots used mean while metrics used median, causing R² inconsistency on skewed posteriors)
    pred_median = np.median(all_pred_samples, axis=1)
    num_plots = len(active_indices)
    num_cols = min(max_cols, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.2 * num_cols, 5.5 * num_rows))
    fig.subplots_adjust(wspace=0.35, hspace=0.35)

    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]

    for plot_idx, param_idx in enumerate(active_indices):
        row = plot_idx // num_cols
        col = plot_idx % num_cols
        ax = axes[row][col]

        y_true = all_true[:, param_idx]
        y_pred = pred_median[:, param_idx]
        label = full_labels[param_idx]

        # Metrics
        score = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(np.abs(y_pred - y_true))

        abs_error = np.abs(y_pred - y_true)

        # Diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        span = max_val - min_val
        plot_min = min_val - 0.05 * span
        plot_max = max_val + 0.05 * span

        ax.plot([plot_min, plot_max], [plot_min, plot_max], color='#e74c3c',
                linestyle='--', linewidth=2, label='Ideal', zorder=10)

        sc = ax.scatter(y_true, y_pred, c=abs_error, cmap='viridis', alpha=0.6,
                       s=15, edgecolors='none', label='Prediction', zorder=5, rasterized=True)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_title(f"{label}", fontweight='bold', pad=10)
        ax.set_xlabel("True Value")
        if col == 0:
            ax.set_ylabel("Predicted Median")

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(r'$|\Delta|$', fontsize=14)

        metrics_text = f"$R^2 = {score:.3f}$\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9, lw=1.5))

        ax.grid(True, linestyle='--', color='gray', alpha=0.3)

    # Hide unused subplots
    for idx in range(num_plots, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row][col].set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"Scatter_Plot{title_suffix.replace(' ', '_')}.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved to {save_path}")
    plt.close(fig)


def draw_corner_plot(true_params, pred_samples, full_labels, sample_idx=0,
                     out_dir="./figures/cache", title_suffix="", active_indices=None):
    """Draw a corner plot for a single sample"""
    try:
        import corner
    except ImportError:
        print("corner package not installed, skipping corner plot")
        return

    print(f"Plotting Corner Plot for sample {sample_idx}...")

    if active_indices is None:
        active_indices = [0, 1, 2, 3, 4]

    active_labels = [full_labels[i] for i in active_indices]

    samples = pred_samples[sample_idx, :, active_indices]
    true_vals = true_params[sample_idx, active_indices]

    fig = corner.corner(
        samples,
        labels=active_labels,
        truths=true_vals,
        truth_color='#e74c3c',
        color='#1f77b4',
        hist_kwargs={'density': True, 'alpha': 0.7},
        smooth=1.0,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={'fontsize': 12}
    )

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"Corner_Plot_sample{sample_idx}{title_suffix.replace(' ', '_')}.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Corner plot saved to {save_path}")
    plt.close(fig)


# ==========================================
# 2. Metric computation and printing
# ==========================================

def _get_model_hparams(checkpoint):
    """Read model hyperparameters from checkpoint, compatible with legacy format (hardcoded defaults)."""
    cfg = checkpoint.get('config', {}).get('model', {})
    return {
        'num_inputs':  cfg.get('num_inputs', 9),
        'num_hidden':  cfg.get('num_hidden', 128),
        'num_layers':  cfg.get('num_layers', 10),
        'context_dim': cfg.get('context_dim', 64),
    }

def print_metrics_table(all_true, all_pred_samples, full_labels, active_indices=None):
    """Print per-parameter evaluation metrics table, return full metrics dict."""
    if active_indices is None:
        active_indices = list(range(len(full_labels)))

    pred_median = np.median(all_pred_samples, axis=1)
    pred_std = np.std(all_pred_samples, axis=1)

    # Compute data range
    data_min = np.min(all_true, axis=0)
    data_max = np.max(all_true, axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1e-8

    # Compute 90% credible interval width
    lower_bound = np.percentile(all_pred_samples, 5, axis=1)
    upper_bound = np.percentile(all_pred_samples, 95, axis=1)
    ci_width = np.mean(upper_bound - lower_bound, axis=0)

    # KS statistic
    from scipy import stats as _stats
    ks_stats = {}
    for idx in active_indices:
        ranks = np.mean(all_pred_samples[:, :, idx] < all_true[:, idx, None], axis=1)
        ks_stat, _ = _stats.kstest(ranks, 'uniform')
        ks_stats[idx] = ks_stat

    print("\n" + "="*110)
    print("Parameter Prediction Performance")
    print("="*110)
    print(f"{'Parameter':<30} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'NMAE (%)':>12} {'Rel. CI Width':>15} {'Mean Std':>12}")
    print("-"*110)

    per_param = {}
    for idx in active_indices:
        y_true = all_true[:, idx]
        y_pred = pred_median[:, idx]

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(np.abs(y_pred - y_true))
        nmae = mae / data_range[idx] * 100
        rel_ci = ci_width[idx] / data_range[idx]
        mean_std = np.mean(pred_std[:, idx])

        label_raw = full_labels[idx].replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        label_display = label_raw[:28]

        print(f"{label_display:<30} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f} {nmae:>12.2f} {rel_ci:>15.4f} {mean_std:>12.4f}")

        from train import PARAM_CONFIG
        param_name = PARAM_CONFIG['names'][idx] if idx < len(PARAM_CONFIG['names']) else label_raw
        per_param[param_name] = {
            'r2': round(float(r2), 6),
            'rmse': round(float(rmse), 6),
            'mae': round(float(mae), 6),
            'nmae_pct': round(float(nmae), 4),
            'rel_ci_width': round(float(rel_ci), 6),
            'mean_std': round(float(mean_std), 6),
            'ks_stat': round(float(ks_stats.get(idx, -1)), 6),
        }

    print("="*110)

    # Print averages
    avg_r2 = np.mean([per_param[k]['r2'] for k in per_param])
    avg_nmae = np.mean([per_param[k]['nmae_pct'] for k in per_param])
    avg_rel_ci = np.mean([per_param[k]['rel_ci_width'] for k in per_param])
    print(f"Average: R² = {avg_r2:.4f}, NMAE = {avg_nmae:.2f}%, Rel. CI Width = {avg_rel_ci:.4f}")

    return {
        'avg_r2': round(float(avg_r2), 6),
        'avg_nmae_pct': round(float(avg_nmae), 4),
        'avg_rel_ci_width': round(float(avg_rel_ci), 6),
        'per_param': per_param,
        'n_samples': int(all_true.shape[0]),
        'n_posterior': int(all_pred_samples.shape[1]),
    }


def save_metrics(metrics, out_dir, filename="metrics.json", extra_info=None):
    """Save metrics to a JSON file.

    Args:
        metrics: Dict returned by print_metrics_table
        out_dir: Output directory (used directly, no ./figures/ prefix added)
        filename: JSON filename
        extra_info: Extra info dict (model name, config, etc.), merged into the output
    """
    import datetime
    out = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if extra_info:
        out.update(extra_info)
    out['metrics'] = metrics

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {save_path}")
    return save_path


# ==========================================
# 3. Evaluation functions
# ==========================================

def evaluate_model(model_path, data_path, mode='multiband', band_name=None,
                   out_dir="./figures/eval", device='cuda', num_samples=5000,
                   active_indices=None, draw_corner=True,
                   return_only=False):
    """
    General model evaluation function

    Args:
        model_path: Path to model weights
        data_path: Path to JSON data
        mode: 'multiband' or 'single'
        band_name: Band name in single-band mode
        out_dir: Output directory (used directly, absolute or relative path)
        device: cuda / cpu
        num_samples: Number of posterior samples per data point
        active_indices: List of parameter indices to evaluate
        draw_corner: Whether to draw corner plots
        return_only: If True, skip plotting and only return true_params/pred_samples/metrics
                     (used by package mode for comparison aggregation)
    """
    # Import training components
    from train import (
        MultiBandDataset, SingleBandDataset,
        MultiBandFlow, SingleBandFlow,
        BAND_CONFIG_CONCAT, BAND_CONFIG_CONTI,
        LIGONoiseModel, PTANoiseModel, LISANoiseModel
    )

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    # Split validation set
    train_idx, val_idx = train_test_split(np.arange(len(raw_data)), test_size=0.2, random_state=42)
    val_data = [raw_data[i] for i in val_idx]

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get configuration from checkpoint
    use_slope = checkpoint.get('use_slope', True)
    use_curvature = checkpoint.get('use_curvature', True)
    num_channels = checkpoint.get('num_channels', 4)
    dataset_type = checkpoint.get('dataset_type', 'concat')
    saved_band_config = checkpoint.get('band_config', None)

    # Select band config: distinguish single-band (flat dict) from multiband (nested dict)
    if dataset_type == 'conti':
        band_config = BAND_CONFIG_CONTI
    else:
        band_config = BAND_CONFIG_CONCAT

    # Build noise_config (preserve physical noise)
    noise_config = {'USE_COMPLEX_NOISE': False, 'noise_level': 0.0}

    # PTA noise
    pta_noise_cfg = checkpoint.get('config', {}).get('noise', {}).get('pta', None)
    if pta_noise_cfg is not None:
        noise_config['pta'] = pta_noise_cfg
        print(f"[Test] PTA physical noise enabled: mode={pta_noise_cfg.get('mode', 'white')}, "
      f"per_sample={pta_noise_cfg.get('per_sample_noise', True)}")

    # LISA noise
    lisa_noise_cfg = checkpoint.get('config', {}).get('noise', {}).get('lisa', None)
    if lisa_noise_cfg is not None:
        noise_config['lisa'] = lisa_noise_cfg
        print(f"[Test] LISA physical noise enabled: mode={lisa_noise_cfg.get('injection_mode', 'N/A')}")

    # LIGO noise
    ligo_noise_cfg = checkpoint.get('config', {}).get('noise', {}).get('ligo', None)
    if ligo_noise_cfg is not None:
        noise_config['ligo'] = ligo_noise_cfg
        print(f"[Test] LIGO physical noise enabled: {ligo_noise_cfg.get('noise_file', 'N/A')}")

    if mode == 'single':
        if band_name is None:
            band_name = checkpoint.get('band_name', 'LISA')

        # Single-band checkpoint's band_config is a flat dict {'len': N, 'freq_range': (...)},
        # multiband checkpoint's is a nested dict {'PTA': {...}, 'LISA': {...}, ...}
        if saved_band_config is not None and 'len' in saved_band_config:
            # flat dict — use directly
            band_config_single = saved_band_config
        elif saved_band_config is not None and band_name in saved_band_config:
            # nested dict — extract the corresponding band
            band_config_single = saved_band_config[band_name]
        else:
            # Fall back to default config
            band_config_single = band_config.get(band_name, {'len': 256, 'freq_range': (-3.0, -1.0)})

        # Ensure freq_range exists
        if 'freq_range' not in band_config_single:
            band_config_single = dict(band_config_single)
            band_config_single['freq_range'] = band_config.get(band_name, {}).get('freq_range', (-3.0, -1.0))

        print(f"[Test] Single-band {band_name}: len={band_config_single['len']}, "
              f"freq_range={band_config_single.get('freq_range', 'N/A')}")

        val_dataset = SingleBandDataset(val_data, band_name, band_config_single, noise_config,
                                        use_slope, use_curvature)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        hp = _get_model_hparams(checkpoint)
        model = SingleBandFlow(band_name, band_config_single['len'],
                                **hp, in_channels=num_channels).to(device)
    else:
        val_dataset = MultiBandDataset(val_data, band_config, noise_config, use_slope, use_curvature)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        hp = _get_model_hparams(checkpoint)
        model = MultiBandFlow(band_config, **hp, in_channels=num_channels).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Use param_scaler saved in checkpoint (consistent with training)
    param_scaler = checkpoint.get('param_scaler', val_dataset.param_scaler)

    # Default: evaluate the 5 primary parameters
    if active_indices is None:
        active_indices = [0, 1, 2, 3, 4]

    # Inference
    print("Generating samples...")
    true_params_list = []
    pred_samples_list = []

    with torch.no_grad():
        if mode == 'single':
            for curve, params in val_loader:
                curve = curve.to(device)
                samples = model.sample(curve, num_samples=num_samples)

                samples = samples.cpu().numpy()
                B_s, N_samp, D = samples.shape
                samples_flat = samples.reshape(-1, D)
                samples_phys = param_scaler.inverse_transform(samples_flat).reshape(B_s, N_samp, D)
                true_phys = param_scaler.inverse_transform(params.numpy())

                true_params_list.append(true_phys)
                pred_samples_list.append(samples_phys)
        else:
            for pta, lisa, ligo, params in val_loader:
                pta, lisa, ligo = pta.to(device), lisa.to(device), ligo.to(device)
                samples = model.sample(pta, lisa, ligo, num_samples=num_samples)

                samples = samples.cpu().numpy()
                B_s, N_samp, D = samples.shape
                samples_flat = samples.reshape(-1, D)
                samples_phys = param_scaler.inverse_transform(samples_flat).reshape(B_s, N_samp, D)
                true_phys = param_scaler.inverse_transform(params.numpy())

                true_params_list.append(true_phys)
                pred_samples_list.append(samples_phys)

    all_true = np.concatenate(true_params_list, axis=0)
    all_pred_samples = np.concatenate(pred_samples_list, axis=0)

    print(f"Evaluated {len(all_true)} samples with {num_samples} posterior samples each")

    # Print metrics table
    metrics = print_metrics_table(all_true, all_pred_samples, PARAM_LABELS, active_indices)

    # Save metrics to JSON
    train_cfg = checkpoint.get('config', {})
    extra_info = {
        'model_path': model_path,
        'data_path': data_path,
        'mode': mode,
        'band_name': band_name if mode == 'single' else 'joint',
        'experiment_name': train_cfg.get('name', 'unknown'),
        'use_slope': use_slope,
        'use_curvature': use_curvature,
        'dataset_type': dataset_type,
    }

    # return_only fast path: package mode aggregation does not need full plots for each model
    # (package mode decides separately in evaluate_package whether to plot)
    result = {
        'true_params': all_true,
        'pred_samples': all_pred_samples,
        'metrics': metrics,
        'checkpoint': checkpoint,
        'extra_info': extra_info,
    }
    if return_only:
        return result

    save_metrics(metrics, out_dir, filename="metrics.json", extra_info=extra_info)

    # Plotting
    suffix = f" ({dataset_type}, slope={use_slope}, curv={use_curvature})"
    if mode == 'single':
        suffix = f" ({band_name}, slope={use_slope}, curv={use_curvature})"

    draw_pp_plot(all_true, all_pred_samples, PARAM_LABELS, active_indices,
                 out_dir=out_dir, title_suffix=suffix)
    draw_scatter_plot(all_true, all_pred_samples, PARAM_LABELS, active_indices,
                      out_dir=out_dir, title_suffix=suffix)

    if draw_corner:
        draw_corner_plot(all_true, all_pred_samples, PARAM_LABELS, sample_idx=0,
                         out_dir=out_dir, title_suffix=suffix, active_indices=active_indices)

    return result


def compare_models(model_configs, out_dir="./figures/model_comparison",
                   device='cuda', num_samples=5000, active_indices=None):
    """
    Compare performance of multiple models

    Args:
        model_configs: list of dict, each containing {'path': model_path, 'data': data_path, 'label': label, 'mode': mode}
        out_dir: Output directory (used directly, absolute or relative path)
    """
    from train import (
        MultiBandDataset, SingleBandDataset,
        MultiBandFlow, SingleBandFlow,
        BAND_CONFIG_CONCAT, BAND_CONFIG_CONTI,
        LIGONoiseModel, PTANoiseModel, LISANoiseModel
    )

    if active_indices is None:
        active_indices = [0, 1, 2, 3, 4]

    results = []

    for config in model_configs:
        model_path = config['path']
        data_path = config['data']
        label = config['label']
        mode = config.get('mode', 'multiband')

        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print('='*60)

        # Load data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        train_idx, val_idx = train_test_split(np.arange(len(raw_data)), test_size=0.2, random_state=42)
        val_data = [raw_data[i] for i in val_idx]

        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        use_slope = checkpoint.get('use_slope', True)
        use_curvature = checkpoint.get('use_curvature', True)
        num_channels = checkpoint.get('num_channels', 4)
        dataset_type = checkpoint.get('dataset_type', 'concat')
        saved_band_config = checkpoint.get('band_config', None)

        if dataset_type == 'conti':
            band_config = BAND_CONFIG_CONTI
        else:
            band_config = BAND_CONFIG_CONCAT

        # Build noise_config (preserve physical noise)
        noise_config = {'USE_COMPLEX_NOISE': False, 'noise_level': 0.0}

        # PTA noise
        pta_noise_cfg = checkpoint.get('config', {}).get('noise', {}).get('pta', None)
        if pta_noise_cfg is not None:
            noise_config['pta'] = pta_noise_cfg

        # LISA noise
        lisa_noise_cfg = checkpoint.get('config', {}).get('noise', {}).get('lisa', None)
        if lisa_noise_cfg is not None:
            noise_config['lisa'] = lisa_noise_cfg

        # LIGO noise
        ligo_noise_cfg = checkpoint.get('config', {}).get('noise', {}).get('ligo', None)
        if ligo_noise_cfg is not None:
            noise_config['ligo'] = ligo_noise_cfg

        if mode == 'single':
            band_name = checkpoint.get('band_name', 'LISA')

            # Support both flat dict (single-band) and nested dict (multiband) formats
            if saved_band_config is not None and 'len' in saved_band_config:
                band_config_single = saved_band_config
            elif saved_band_config is not None and band_name in saved_band_config:
                band_config_single = saved_band_config[band_name]
            else:
                band_config_single = band_config.get(band_name, {'len': 256, 'freq_range': (-3.0, -1.0)})

            if 'freq_range' not in band_config_single:
                band_config_single = dict(band_config_single)
                band_config_single['freq_range'] = band_config.get(band_name, {}).get('freq_range', (-3.0, -1.0))

            val_dataset = SingleBandDataset(val_data, band_name, band_config_single, noise_config,
                                            use_slope, use_curvature)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            hp = _get_model_hparams(checkpoint)
            model = SingleBandFlow(band_name, band_config_single['len'],
                                    **hp, in_channels=num_channels).to(device)
        else:
            val_dataset = MultiBandDataset(val_data, band_config, noise_config, use_slope, use_curvature)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            hp = _get_model_hparams(checkpoint)
            model = MultiBandFlow(band_config, **hp, in_channels=num_channels).to(device)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()

        # Use param_scaler saved in checkpoint (consistent with training)
        param_scaler = checkpoint.get('param_scaler', val_dataset.param_scaler)

        # Inference
        true_params_list = []
        pred_samples_list = []

        with torch.no_grad():
            if mode == 'single':
                for curve, params in val_loader:
                    curve = curve.to(device)
                    samples = model.sample(curve, num_samples=num_samples)

                    samples = samples.cpu().numpy()
                    B_s, N_samp, D = samples.shape
                    samples_flat = samples.reshape(-1, D)
                    samples_phys = param_scaler.inverse_transform(samples_flat).reshape(B_s, N_samp, D)
                    true_phys = param_scaler.inverse_transform(params.numpy())

                    true_params_list.append(true_phys)
                    pred_samples_list.append(samples_phys)
            else:
                for pta, lisa, ligo, params in val_loader:
                    pta, lisa, ligo = pta.to(device), lisa.to(device), ligo.to(device)
                    samples = model.sample(pta, lisa, ligo, num_samples=num_samples)

                    samples = samples.cpu().numpy()
                    B_s, N_samp, D = samples.shape
                    samples_flat = samples.reshape(-1, D)
                    samples_phys = param_scaler.inverse_transform(samples_flat).reshape(B_s, N_samp, D)
                    true_phys = param_scaler.inverse_transform(params.numpy())

                    true_params_list.append(true_phys)
                    pred_samples_list.append(samples_phys)

        all_true = np.concatenate(true_params_list, axis=0)
        all_pred_samples = np.concatenate(pred_samples_list, axis=0)

        # Compute R² (using posterior median, consistent with evaluate_model)
        pred_median = np.median(all_pred_samples, axis=1)
        r2_scores = {}
        for idx in active_indices:
            r2_scores[PARAM_LABELS[idx]] = r2_score(all_true[:, idx], pred_median[:, idx])

        results.append({
            'label': label,
            'r2_scores': r2_scores,
            'all_true': all_true,
            'all_pred_samples': all_pred_samples
        })

        print(f"\n{label} R² scores:")
        for param, score in r2_scores.items():
            print(f"  {param}: {score:.4f}")

    # Draw comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(active_indices))
    width = 0.8 / len(results)

    colors = ['#1f77b4', '#ff7f0e', '#2ecc71', '#e74c3c', '#9b59b6', '#3498db']

    for i, result in enumerate(results):
        scores = [result['r2_scores'][PARAM_LABELS[idx]] for idx in active_indices]
        ax.bar(x + i * width, scores, width, label=result['label'],
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_ylabel('R² Score', fontsize=18)
    ax.set_title('Model Comparison: Parameter Prediction Performance', fontsize=20, fontweight='bold')
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([PARAM_LABELS[idx].replace('$', '') for idx in active_indices],
                       rotation=15, fontsize=14)
    ax.legend(fontsize=14)
    ax.set_ylim(-0.5, 1)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    comparison_path = os.path.join(out_dir, "model_comparison.pdf")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {comparison_path}")
    plt.close(fig)

    # Print comparison table
    print("\n" + "="*100)
    header = f"{'Parameter':<25} " + " ".join([f"{r['label']:>15}" for r in results])
    print(header)
    print("-"*100)

    for idx in active_indices:
        label = PARAM_LABELS[idx].replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        scores = [r['r2_scores'][PARAM_LABELS[idx]] for r in results]
        scores_str = " ".join([f"{s:>15.4f}" for s in scores])
        print(f"{label:<25} {scores_str}")

    print("="*100)

    # Save comparison metrics
    compare_out = {
        'models': []
    }
    from train import PARAM_CONFIG
    for r in results:
        model_metrics = {'label': r['label'], 'r2_scores': {}}
        for idx in active_indices:
            param_name = PARAM_CONFIG['names'][idx] if idx < len(PARAM_CONFIG['names']) else str(idx)
            model_metrics['r2_scores'][param_name] = round(float(r['r2_scores'][PARAM_LABELS[idx]]), 6)
        compare_out['models'].append(model_metrics)

    save_metrics(compare_out, out_dir, filename="comparison_metrics.json")

    return results


# ==========================================
# 3.5 Experiment package evaluation: 1 joint + 3 single, one-click run + automatic comparison
# ==========================================

def _draw_joint_vs_single_comparison(results, out_dir, active_indices, package_name):
    """Draw joint vs single R² comparison bar chart + text table.

    Args:
        results: dict {task_name: {'metrics': ..., 'extra_info': ...}}
        out_dir: Output directory
        active_indices: Parameter indices to evaluate
        package_name: Package name, used in titles
    """
    os.makedirs(out_dir, exist_ok=True)

    # Maintain order: joint on the far left for quick comparison, followed by 3 single bands
    task_order = ['joint', 'PTA', 'LISA', 'LIGO']
    tasks_present = [t for t in task_order if t in results]
    if len(tasks_present) < 2:
        print(f"[comparison] only {len(tasks_present)} task(s), skipping comparison plot")
        return

    from train import PARAM_CONFIG
    param_names = PARAM_CONFIG['names']

    # R² matrix: rows = tasks, cols = params
    r2_matrix = []
    for task in tasks_present:
        per_param = results[task]['metrics'].get('per_param', {})
        row = []
        for i in active_indices:
            pname = param_names[i] if i < len(param_names) else str(i)
            row.append(per_param.get(pname, {}).get('r2', float('nan')))
        r2_matrix.append(row)
    r2_matrix = np.array(r2_matrix)

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(active_indices))
    n_tasks = len(tasks_present)
    width = 0.8 / n_tasks
    color_map = {
        'joint': '#e74c3c',
        'PTA':   '#1f77b4',
        'LISA':  '#2ecc71',
        'LIGO':  '#ff7f0e',
    }

    for i, task in enumerate(tasks_present):
        offset = (i - (n_tasks - 1) / 2) * width
        ax.bar(x + offset, r2_matrix[i], width,
               label=task, color=color_map.get(task, '#888888'), alpha=0.85,
               edgecolor='black', linewidth=0.5)

    ax.set_ylabel('$R^2$', fontsize=18)
    ax.set_title(f'Joint vs Single — {package_name}',
                 fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([PARAM_LABELS[i] for i in active_indices], fontsize=14)
    ax.legend(fontsize=14, loc='lower right')
    ymin = min(-0.5, np.nanmin(r2_matrix) - 0.1) if r2_matrix.size else -0.5
    ax.set_ylim(ymin, 1.05)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    pdf_path = os.path.join(out_dir, 'joint_vs_single.pdf')
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[comparison plot]  {pdf_path}")

    # ---- Text table (also written to stdout) ----
    print("\n" + "=" * (25 + 12 * n_tasks))
    print(f"{'Parameter':<25}" + "".join([f"{t:>12}" for t in tasks_present]))
    print("-" * (25 + 12 * n_tasks))
    for j, idx in enumerate(active_indices):
        pname = param_names[idx] if idx < len(param_names) else str(idx)
        print(f"{pname:<25}" + "".join([f"{r2_matrix[i, j]:>12.4f}"
                                          for i in range(n_tasks)]))
    print("=" * (25 + 12 * n_tasks))

    # ---- Also write to a text file ----
    table_path = os.path.join(out_dir, 'joint_vs_single_R2.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(f"{'Parameter':<25}" + "".join([f"{t:>12}" for t in tasks_present]) + "\n")
        f.write("-" * (25 + 12 * n_tasks) + "\n")
        for j, idx in enumerate(active_indices):
            pname = param_names[idx] if idx < len(param_names) else str(idx)
            f.write(f"{pname:<25}"
                    + "".join([f"{r2_matrix[i, j]:>12.4f}" for i in range(n_tasks)])
                    + "\n")
    print(f"[comparison table] {table_path}")


def evaluate_package(cfg, device='cuda', num_samples=5000,
                     active_indices=None, draw_corner=True):
    """Evaluate a training experiment package (1 joint + 3 single), output to {package_dir}/eval/.

    Directory structure:
        {package_dir}/
        ├── joint/flow_joint.pt          <-- training output
        ├── single/{PTA,LISA,LIGO}_flow.pt
        └── eval/                        <-- output of this function
            ├── eval_log.txt
            ├── summary_metrics.json
            ├── joint/                   (PP, scatter, corner, metrics.json)
            ├── single/
            │   ├── PTA/
            │   ├── LISA/
            │   └── LIGO/
            └── comparison/
                ├── joint_vs_single.pdf
                └── joint_vs_single_R2.txt

    Args:
        cfg: Full package config dict (same format as in train.py)
    """
    package_dir = cfg['output']['package_dir']
    eval_dir = os.path.join(package_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Resolve data paths
    joint_path = cfg['dataset']['joint_path']
    single_dir = cfg['dataset']['single_dir']
    single_files = cfg['dataset']['single_files']

    # Default to training order (LIGO→LISA→PTA→joint), but respect custom order in config
    tasks = cfg.get('tasks', ['LIGO', 'LISA', 'PTA', 'joint'])

    # Tee stdout → write to both terminal and eval_log.txt
    log_path = os.path.join(eval_dir, 'eval_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    results = {}  # task_name -> {'metrics': ..., 'extra_info': ...}

    try:
        print("\n" + "#" * 70)
        print(f"# EVALUATION PACKAGE: {cfg.get('name', 'unnamed')}")
        print(f"# Description:  {cfg.get('description', 'N/A')}")
        print(f"# Package dir:  {package_dir}")
        print(f"# Eval dir:     {eval_dir}")
        print(f"# Tasks:        {tasks}")
        print(f"# num_samples:  {num_samples}")
        print(f"# active_idx:   {active_indices}")
        print("#" * 70)

        for task in tasks:
            print("\n" + "#" * 70)
            print(f"# [EVAL TASK]  {task}")
            print("#" * 70)

            if task == 'joint':
                model_path = os.path.join(package_dir, 'joint', 'flow_joint.pt')
                data_path = joint_path
                task_out = os.path.join(eval_dir, 'joint')
                mode = 'multiband'
                band_name = None
            elif task in ('PTA', 'LISA', 'LIGO'):
                model_path = os.path.join(package_dir, 'single', f'{task}_flow.pt')
                if task not in single_files:
                    print(f"[WARNING] single_files missing entry for '{task}', skipping")
                    continue
                data_path = os.path.join(single_dir, single_files[task])
                task_out = os.path.join(eval_dir, 'single', task)
                mode = 'single'
                band_name = task
            else:
                print(f"[WARNING] Unknown task '{task}', skipping")
                continue

            if not os.path.isfile(model_path):
                print(f"[WARNING] Model not found: {model_path}")
                print(f"          (training probably hasn't finished this task yet)")
                continue
            if not os.path.isfile(data_path):
                print(f"[WARNING] Data not found: {data_path}, skipping")
                continue

            res = evaluate_model(
                model_path, data_path,
                mode=mode, band_name=band_name,
                out_dir=task_out, device=device,
                num_samples=num_samples, active_indices=active_indices,
                draw_corner=draw_corner,
            )
            results[task] = {
                'metrics': res['metrics'],
                'extra_info': res['extra_info'],
            }

        # ---- Comparison plot + summary JSON ----
        if len(results) >= 2:
            print("\n" + "#" * 70)
            print("# [EVAL] Generating joint vs single comparison")
            print("#" * 70)
            _draw_joint_vs_single_comparison(
                results,
                out_dir=os.path.join(eval_dir, 'comparison'),
                active_indices=active_indices or [0, 1, 2, 3, 4],
                package_name=cfg.get('name', 'unnamed'),
            )

        # Summary JSON: convenient for cross-package comparison later (mock vs observed)
        import datetime
        summary = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'package_name': cfg.get('name', 'unnamed'),
            'package_dir': package_dir,
            'tasks_evaluated': list(results.keys()),
            'num_samples': num_samples,
            'active_indices': active_indices,
            'per_task_metrics': {task: r['metrics'] for task, r in results.items()},
        }
        summary_path = os.path.join(eval_dir, 'summary_metrics.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[summary] {summary_path}")

        print("\n" + "#" * 70)
        print(f"# PACKAGE EVAL COMPLETE: {eval_dir}")
        print("#" * 70)
    finally:
        sys.stdout = original_stdout
        log_file.close()

    return results


# ==========================================
# 4. Main function
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Unified Normalizing Flow Model Testing')

    # Config file mode (highest priority, aligned with train.py)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training config JSON. '
                             'If it contains output.package_dir, runs package eval.')

    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset (.json file)')
    parser.add_argument('--mode', type=str, default='multiband',
                        choices=['multiband', 'single'],
                        help='Evaluation mode')
    parser.add_argument('--band', type=str, default=None,
                        choices=['PTA', 'LISA', 'LIGO'],
                        help='Band name for single mode')
    parser.add_argument('--figpath', type=str, default=None,
                        help='Legacy: subdirectory name under ./figures/ for outputs. '
                             'Ignored in package mode (outputs go to {package_dir}/eval/).')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Direct output directory (overrides --figpath). '
                             'Ignored in package mode.')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of posterior samples per data point')
    parser.add_argument('--params', type=str, default='primary',
                        choices=['primary', 'all', 'secondary'],
                        help='Which parameters to evaluate')
    parser.add_argument('--no-corner', action='store_true',
                        help='Skip corner plot')

    # Model comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Enable legacy model comparison mode')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Paths to models for comparison')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Labels for each model in comparison')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine parameter indices
    if args.params == 'primary':
        active_indices = [0, 1, 2, 3, 4]
    elif args.params == 'secondary':
        active_indices = [5, 6, 7, 8]
    else:
        active_indices = list(range(9))

    # ==========================================
    # Mode 1: Package evaluation (highest priority)
    # ==========================================
    if args.config:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        print(f"Loaded config from: {args.config}")

        if 'package_dir' in cfg.get('output', {}):
            # Package mode: auto-evaluate 1 joint + 3 single + generate comparison plots
            print(f"[mode] package evaluation")
            print(f"[device] {device}")
            evaluate_package(
                cfg, device=device,
                num_samples=args.num_samples,
                active_indices=active_indices,
                draw_corner=not args.no_corner,
            )
            return

        # ==========================================
        # Mode 2: Legacy single-model (infer paths from config)
        # ==========================================
        print(f"[mode] legacy single-model evaluation from config")

        # model path
        if args.model is None:
            save_dir = cfg['output']['save_dir']
            save_name = cfg['output'].get('save_name', 'flow.pt')
            args.model = os.path.join(save_dir, save_name)

        # mode & band & data
        ds_type = cfg['dataset']['type']
        if ds_type == 'single':
            args.mode = 'single'
            bands = cfg['dataset']['bands']
            if args.band is None:
                args.band = bands[0]
            if args.data is None:
                data_dir = cfg['dataset']['data_dir']
                band_files = cfg['dataset']['band_files']
                args.data = os.path.join(data_dir, band_files[args.band])
        else:
            args.mode = 'multiband'
            if args.data is None:
                args.data = cfg['dataset']['path']

        # legacy figpath default from config name
        if args.figpath is None and args.out_dir is None:
            args.figpath = cfg.get('name', 'eval')

    # ==========================================
    # Parameter validation
    # ==========================================
    if not args.compare and (args.model is None or args.data is None):
        parser.error("--config or both --model and --data are required")

    # Resolve final output directory (priority: --out-dir > --figpath > default 'eval')
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        figpath = args.figpath or 'eval'
        out_dir = os.path.join('./figures', figpath)

    print(f"Using device: {device}")
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data}")
    print(f"Mode:    {args.mode}" + (f" ({args.band})" if args.band else ""))
    print(f"Output:  {out_dir}")

    if args.compare:
        # ==========================================
        # Mode 3: Legacy model comparison
        # ==========================================
        if not args.models or not args.labels:
            print("Error: --models and --labels are required for comparison mode")
            return

        if len(args.models) != len(args.labels):
            print("Error: Number of models must match number of labels")
            return

        model_configs = [
            {'path': m, 'data': args.data, 'label': l, 'mode': args.mode}
            for m, l in zip(args.models, args.labels)
        ]

        compare_models(model_configs, out_dir=out_dir, device=device,
                       num_samples=args.num_samples, active_indices=active_indices)

    else:
        # ==========================================
        # Mode 4: Single-model evaluation
        # ==========================================
        evaluate_model(
            args.model, args.data,
            mode=args.mode,
            band_name=args.band,
            out_dir=out_dir,
            device=device,
            num_samples=args.num_samples,
            active_indices=active_indices,
            draw_corner=not args.no_corner,
        )


if __name__ == "__main__":
    main()
