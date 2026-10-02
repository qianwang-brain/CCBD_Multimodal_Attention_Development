from __future__ import annotations

from pathlib import Path
import sys

PYDEPS = Path(__file__).resolve().parent / 'pydeps'
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

BASE = Path(r'/path/to/pku_prediction')
PREDICTIONS_CSV = BASE / 'analysis_outputs_two_model_multi_family_allsubjects_svr_lgbm_tuned_dqia' / 'two_model_multi_family_predictions_all.csv'
RESULTS_CSV = BASE / 'analysis_outputs_score_permutation_lightgbm_sigtargets_tuned_dqia' / 'score_permutation_lightgbm_sigtargets_summary.csv'
OUTDIR = BASE / 'analysis_outputs_score_permutation_lightgbm_sigtargets_tuned_dqia' / 'scatter_plots'
OUTDIR.mkdir(exist_ok=True)

TARGETS = {
    'data_final_used_HX_predict': ['DQ_IA', 'DQ_HI', 'DQ_TO', 'M工作记忆'],
    'data_final_used_Tt_predict': ['DQ_IA', 'DQ_TO', 'M情感控制', 'M行为管理指数'],
}

LABELS = {
    'data_final_used_HX_predict': 'HX',
    'data_final_used_Tt_predict': 'Tt',
}

DISPLAY_NAMES = {
    'DQ_IA': 'DQ_IA',
    'DQ_HI': 'DQ_HI',
    'DQ_TO': 'DQ_TO',
    'M工作记忆': 'M_WorkingMemory',
    'M情感控制': 'M_EmotionControl',
    'M行为管理指数': 'M_BehaviorRegIndex',
}

STYLE = {
    'HX': {
        'bg': '#F0A9AF',
        'fill': '#B93649',
        'edge': '#111111',
        'line': '#0C234B',
        'band': '#F0B7BE',
    },
    'Tt': {
        'bg': '#AFC2FF',
        'fill': '#355FD4',
        'edge': '#111111',
        'line': '#0C234B',
        'band': '#BFD0FF',
    },
}


def apply_publication_style() -> None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['legend.frameon'] = False


def add_regression_with_ci(ax, x: np.ndarray, y: np.ndarray, line_color: str, band_color: str) -> None:
    if len(x) < 3:
        return
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0:
        return

    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = y_mean - slope * x_mean
    y_fit = intercept + slope * x
    residual = y - y_fit
    dof = n - 2
    if dof <= 0:
        return
    s_err = float(np.sqrt(np.sum(residual ** 2) / dof))

    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_line = intercept + slope * x_line
    t_crit = float(student_t.ppf(0.975, dof))
    se_line = s_err * np.sqrt((1.0 / n) + ((x_line - x_mean) ** 2 / sxx))
    y_low = y_line - t_crit * se_line
    y_high = y_line + t_crit * se_line

    ax.fill_between(x_line, y_low, y_high, color=band_color, alpha=0.24, linewidth=0, zorder=2)
    ax.plot(x_line, y_line, color=line_color, linewidth=1.5, zorder=4)


def layered_scatter(ax, x: np.ndarray, y: np.ndarray, dataset_label: str) -> None:
    style = STYLE[dataset_label]

    ax.scatter(
        x,
        y,
        s=42,
        facecolors=style['fill'],
        edgecolors='none',
        alpha=0.06,
        zorder=1,
        rasterized=True,
    )

    ax.scatter(
        x,
        y,
        s=34,
        facecolors=style['fill'],
        edgecolors=style['edge'],
        linewidths=0.55,
        alpha=0.78,
        zorder=3,
        rasterized=True,
    )


def make_one_panel(ax, sub: pd.DataFrame, title: str, stat_text: str, dataset_label: str) -> None:
    x = sub['y_true_eval'].to_numpy(float)
    y = sub['y_pred'].to_numpy(float)
    layered_scatter(ax, x, y, dataset_label=dataset_label)
    add_regression_with_ci(ax, x, y, STYLE[dataset_label]['line'], STYLE[dataset_label]['band'])
    ax.set_title(title, fontsize=10.5, pad=5)
    ax.set_xlabel('Observed data', fontsize=9.5)
    ax.set_ylabel('Predicted score', fontsize=9.5)
    ax.grid(False)
    ax.tick_params(labelsize=8.5, width=0.8, length=3)
    ax.text(
        0.97,
        0.96,
        stat_text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=8.5,
        color='#102347',
    )


def main() -> None:
    apply_publication_style()
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    res_df = pd.read_csv(RESULTS_CSV)

    pred_df = pred_df[
        (pred_df['model_family'] == 'LightGBM')
        & (pred_df['two_model_role'] == 'model1_age_sex_brain_raw_y')
        & (pred_df['label_filter'] == 'all')
    ].copy()

    saved_files = []
    for dataset, target_list in TARGETS.items():
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4), constrained_layout=True)
        axes = axes.ravel()
        dataset_label = LABELS[dataset]

        for ax, target in zip(axes, target_list):
            sub = pred_df[(pred_df['dataset'] == dataset) & (pred_df['target'] == target)].copy()
            if sub.empty:
                raise ValueError(f'Missing predictions for {dataset} - {target}')

            row = res_df[(res_df['dataset'] == dataset) & (res_df['target'] == target)]
            if row.empty:
                raise ValueError(f'Missing permutation summary for {dataset} - {target}')
            row = row.iloc[0]
            stat_text = (
                f"r = {row['real_model1_pearson_r']:.4f}\n"
                f"p = {row['real_model1_pearson_p']:.4g}\n"
                f"p(perm) = {row['empirical_p_signed_tail']:.6g}"
            )
            display = DISPLAY_NAMES.get(target, target)
            make_one_panel(ax, sub, f'{dataset_label} | {display}', stat_text, dataset_label)

        for ax in axes[len(target_list):]:
            ax.axis('off')

        fig.suptitle(f'{dataset_label} significant targets under score permutation', fontsize=12)
        png_path = OUTDIR / f'{dataset_label}_score_permutation_significant_scatter.png'
        pdf_path = OUTDIR / f'{dataset_label}_score_permutation_significant_scatter.pdf'
        svg_path = OUTDIR / f'{dataset_label}_score_permutation_significant_scatter.svg'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        plt.close(fig)
        saved_files.extend([png_path, pdf_path, svg_path])

    print('Saved files:')
    for path in saved_files:
        print(path)


if __name__ == '__main__':
    main()




