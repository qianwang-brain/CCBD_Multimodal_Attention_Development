from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYDEPS = ROOT / "pydeps"
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages


INPUTS = {
    "HX": {
        "mat": ROOT / "data_follow_finall_HX.mat",
        "hits": ROOT / "analysis_outputs_followup_med1_partialcorr" / "HX" / "fdr_hits.csv",
    },
    "Tt": {
        "mat": ROOT / "data_follow_finall_Tt.mat",
        "hits": ROOT / "analysis_outputs_followup_med1_partialcorr" / "Tt" / "fdr_hits.csv",
    },
}
OUTPUT_ROOT = ROOT / "analysis_outputs_followup_med1_partialcorr_trend_plots_pdf"
MEDICATION_TYPE = 1
STYLE = {
    "HX": {
        "fill": "#C43E52",
        "edge": "#111111",
        "line": "#0C234B",
        "band": "#F0B7BE",
        "stat": "#102347",
    },
    "Tt": {
        "fill": "#3F68D8",
        "edge": "#111111",
        "line": "#0C234B",
        "band": "#BFD0FF",
        "stat": "#102347",
    },
}


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["legend.frameon"] = False


def decode_mat_string(value) -> str:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S"):
            return "".join(value.tolist()).strip()
        if value.size == 1:
            return decode_mat_string(value.item())
        return str(value)
    return str(value).strip()


def residualize(y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    if cov.size == 0:
        return y - np.nanmean(y)
    design = np.column_stack([np.ones(len(y)), cov])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[<>:\"/\\\\|?*]+", "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def load_dataset(dataset_name: str) -> dict[str, object]:
    mat = sio.loadmat(INPUTS[dataset_name]["mat"])
    subject_key = "sub_final2" if "sub_final2" in mat else "sub_f_final"
    subject_ids = [decode_mat_string(x) for x in mat[subject_key].reshape(-1)]
    data1 = mat["data1"]
    data2 = mat["data2"]
    symptom_names = [decode_mat_string(x) for x in mat["data1_name"].reshape(-1)][1:]

    keep_mask = data1[:, 0] == float(MEDICATION_TYPE)
    keep_ids = [subject_ids[idx] for idx, keep in enumerate(keep_mask) if keep]
    improvement = pd.DataFrame(data1[keep_mask, 1:] - data2[keep_mask, 1:], columns=symptom_names, index=keep_ids)
    cov_base = pd.DataFrame(mat["inf"][keep_mask, :], columns=["sex", "age"], index=keep_ids)
    mfd = pd.Series(mat["mfd"][keep_mask].reshape(-1), index=keep_ids, name="mfd")
    features = {
        "gmv": pd.DataFrame(mat["gmv2"][keep_mask, :] - mat["gmv1"][keep_mask, :], index=keep_ids),
        "fcs": pd.DataFrame(mat["fcs2"][keep_mask, :] - mat["fcs1"][keep_mask, :], index=keep_ids),
        "eeg": pd.DataFrame(mat["eeg2"][keep_mask, :] - mat["eeg1"][keep_mask, :], index=keep_ids),
    }
    return {"improvement": improvement, "cov_base": cov_base, "mfd": mfd, "features": features}


def get_covariates(dataset: dict[str, object], modality: str) -> pd.DataFrame:
    cov = dataset["cov_base"].copy()
    if modality == "fcs":
        cov["mfd"] = dataset["mfd"]
    return cov


def prepare_plot_frame(dataset: dict[str, object], row: pd.Series) -> pd.DataFrame:
    modality = row["modality"].lower()
    feature_index = int(row["feature_index"]) - 1
    symptom = row["symptom"]
    y = dataset["improvement"][symptom].rename("symptom_improvement")
    x = dataset["features"][modality].iloc[:, feature_index].rename("feature_change")
    cov = get_covariates(dataset, modality)
    frame = pd.concat([x, y, cov], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    cov_values = frame[cov.columns].to_numpy(dtype=float) if len(cov.columns) else np.empty((len(frame), 0))
    frame["x_resid"] = residualize(frame["feature_change"].to_numpy(dtype=float), cov_values)
    frame["y_resid"] = residualize(frame["symptom_improvement"].to_numpy(dtype=float), cov_values)
    return frame


def layered_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, style: dict[str, str]) -> None:
    ax.scatter(
        x,
        y,
        s=180,
        facecolors=style["fill"],
        edgecolors=style["edge"],
        linewidths=0.80,
        alpha=0.72,
        zorder=3,
        rasterized=True,
    )


def draw_plot(dataset_name: str, row: pd.Series, frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.1, 4.2))
    style = STYLE[dataset_name]
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = frame["x_resid"].to_numpy(dtype=float)
    y = frame["y_resid"].to_numpy(dtype=float)
    layered_scatter(ax, x, y, style)
    coeff = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = coeff[0] * x_line + coeff[1]

    n = len(x)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    residuals = y - (coeff[0] * x + coeff[1])
    dof = max(n - 2, 1)
    s_err = np.sqrt(np.sum(residuals**2) / dof) if n > 2 else 0.0
    if n > 2 and sxx > 0:
        t_crit = stats.t.ppf(0.975, dof)
        se_fit = s_err * np.sqrt((1.0 / n) + ((x_line - x_mean) ** 2) / sxx)
        ci_lower = y_line - t_crit * se_fit
        ci_upper = y_line + t_crit * se_fit
        ax.fill_between(x_line, ci_lower, ci_upper, color=style["band"], alpha=0.24, linewidth=0, zorder=1)

    ax.plot(x_line, y_line, color=style["line"], linewidth=1.5, zorder=4)

    modality = row["modality"].upper()
    feature_index = int(row["feature_index"])
    symptom = row["symptom"]
    ax.set_xlabel("Feature-change residual", fontsize=9.5)
    ax.set_ylabel("Symptom-improvement residual", fontsize=9.5)
    ax.set_title(
        f"{dataset_name} | {modality} feature {feature_index} vs {symptom}",
        fontsize=10.5,
        pad=5,
    )
    ax.grid(False)
    ax.tick_params(labelsize=8.5, width=0.8, length=3)
    stat_text = (
        f"partial r = {float(row['partial_r']):.3f}\n"
        f"p = {float(row['p_value']):.4g}\n"
        f"q = {float(row['p_fdr_within_symptom']):.4g}\n"
        f"n = {int(row['n'])}"
    )
    ax.text(
        0.97,
        0.96,
        stat_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=style["stat"],
    )
    fig.tight_layout()
    return fig


def main() -> None:
    apply_publication_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_frames = []
    for dataset_name, cfg in INPUTS.items():
      dataset = load_dataset(dataset_name)
      hits = pd.read_csv(cfg["hits"], encoding="utf-8-sig")
      dataset_dir = OUTPUT_ROOT / dataset_name
      dataset_dir.mkdir(parents=True, exist_ok=True)
      indiv_dir = dataset_dir / "individual_pdfs"
      indiv_dir.mkdir(parents=True, exist_ok=True)
      combined_pdf = dataset_dir / f"{dataset_name}_significant_partialcorr_trends.pdf"

      summary_rows = []
      with PdfPages(combined_pdf) as pdf:
          for _, row in hits.iterrows():
              frame = prepare_plot_frame(dataset, row)
              if frame.empty:
                  continue
              fig = draw_plot(dataset_name, row, frame)
              pdf.savefig(fig)
              filename = sanitize_filename(f"{dataset_name}_{row['modality']}_feature_{int(row['feature_index'])}_{row['symptom']}.pdf")
              individual_pdf = indiv_dir / filename
              try:
                  fig.savefig(individual_pdf, bbox_inches="tight")
              except PermissionError:
                  fallback_pdf = indiv_dir / f"{individual_pdf.stem}_updated.pdf"
                  fig.savefig(fallback_pdf, bbox_inches="tight")
                  individual_pdf = fallback_pdf
              plt.close(fig)
              summary_rows.append(
                  {
                      "dataset": dataset_name,
                      "modality": row["modality"],
                      "feature_index": int(row["feature_index"]),
                      "symptom": row["symptom"],
                      "n": int(row["n"]),
                      "partial_r": float(row["partial_r"]),
                      "p_value": float(row["p_value"]),
                      "p_fdr_within_symptom": float(row["p_fdr_within_symptom"]),
                      "combined_pdf": str(combined_pdf),
                      "individual_pdf": str(individual_pdf),
                  }
              )
      summary_df = pd.DataFrame(summary_rows)
      summary_df.to_csv(dataset_dir / "trend_plot_summary.csv", index=False, encoding="utf-8-sig")
      summary_frames.append(summary_df)

    pd.concat(summary_frames, ignore_index=True).to_csv(OUTPUT_ROOT / "trend_plot_summary_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
