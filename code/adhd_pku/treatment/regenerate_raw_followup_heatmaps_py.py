from __future__ import annotations

import csv
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
from matplotlib.colors import LinearSegmentedColormap


RESULTS_ROOT = ROOT / "analysis_outputs_followup_med1_partialcorr"
SYMPTOM_ORDER = [
    "V0注意缺陷",
    "V0多动冲动",
    "0总分",
    "品行",
    "学习",
    "心身",
    "冲动多动",
    "焦虑",
    "多动指数",
    "EL",
    "M抑制",
    "M转换",
    "M情感控制",
    "M启动",
    "M工作记忆",
    "M计划",
    "M组织",
    "M监控",
    "M行为管理指数",
    "M元认知指数",
    "M总分",
]
MODALITY_ORDER = {"eeg": 0, "gmv": 1, "fcs": 2}


plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}


def read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def order_symptoms(rows: list[dict[str, str]]) -> list[str]:
    symptoms = []
    for row in rows:
        symptom = row["symptom"]
        if symptom not in symptoms:
            symptoms.append(symptom)
    ordered = [s for s in SYMPTOM_ORDER if s in symptoms]
    ordered.extend([s for s in symptoms if s not in ordered])
    return ordered


def order_features(rows: list[dict[str, str]]) -> list[str]:
    seen: dict[str, tuple[int, float, str]] = {}
    for row in rows:
        if not parse_bool(row["significant_fdr"]):
            continue
        modality = row["modality"].lower()
        feature_index = row["feature_index"]
        label = f"{modality}\n{feature_index}"
        feature_num = float(feature_index)
        if label not in seen:
            seen[label] = (MODALITY_ORDER.get(modality, 99), feature_num, feature_index)
    return [label for label, _ in sorted(seen.items(), key=lambda x: x[1])]


def build_matrices(
    rows: list[dict[str, str]],
    symptom_levels: list[str],
    feature_levels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    value_mat = np.full((len(symptom_levels), len(feature_levels)), np.nan, dtype=float)
    sig_mat = np.zeros((len(symptom_levels), len(feature_levels)), dtype=bool)
    symptom_idx = {s: i for i, s in enumerate(symptom_levels)}
    feature_idx = {f: i for i, f in enumerate(feature_levels)}

    for row in rows:
        feature_label = f"{row['modality'].lower()}\n{row['feature_index']}"
        symptom = row["symptom"]
        if feature_label not in feature_idx or symptom not in symptom_idx:
            continue
        r = symptom_idx[symptom]
        c = feature_idx[feature_label]
        value = float(row["partial_r"])
        is_sig = parse_bool(row["significant_fdr"])
        current = value_mat[r, c]
        current_sig = sig_mat[r, c]
        if np.isnan(current) or (is_sig and not current_sig) or (is_sig == current_sig and abs(value) > abs(current)):
            value_mat[r, c] = value
            sig_mat[r, c] = is_sig
        elif is_sig:
            sig_mat[r, c] = True

    return value_mat, sig_mat


def save_heatmap(dataset: str, rows: list[dict[str, str]]) -> dict[str, object]:
    symptom_levels = order_symptoms(rows)
    feature_levels = order_features(rows)
    output_pdf = RESULTS_ROOT / dataset / f"{dataset}_significant_fdr_heatmap.pdf"

    if not feature_levels:
        fig, ax = plt.subplots(figsize=(7, 11))
        ax.axis("off")
        ax.text(0.5, 0.56, "No rows with significant_fdr == TRUE", ha="center", va="center", fontsize=14)
        ax.text(0.5, 0.48, f"Symptoms in file: {len(symptom_levels)}", ha="center", va="center", fontsize=11, color="#555555")
        ax.set_title(f"{dataset} significant_fdr heatmap", fontsize=13, pad=16)
        fig.savefig(output_pdf, bbox_inches="tight")
        plt.close(fig)
        return {"csv_file": str(RESULTS_ROOT / dataset / "all_results.csv"), "output_file": str(output_pdf), "significant_hits": 0}

    value_mat, sig_mat = build_matrices(rows, symptom_levels, feature_levels)
    max_abs = float(np.nanmax(np.abs(value_mat)))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    cmap = LinearSegmentedColormap.from_list("rb_custom", ["#2166AC", "#F7F7F7", "#B2182B"], N=256)
    width = max(8.5, 0.7 * len(feature_levels) + 4)
    height = max(11, 0.45 * len(symptom_levels) + 4)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(value_mat, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(np.arange(len(feature_levels)))
    ax.set_xticklabels(feature_levels, rotation=90, fontsize=9)
    ax.set_yticks(np.arange(len(symptom_levels)))
    ax.set_yticklabels(symptom_levels, fontsize=10)
    ax.set_xlabel("Feature (modality + feature_index)")
    ax.set_ylabel("Symptom")
    ax.set_title(f"{dataset} significant_fdr heatmap", fontsize=13, pad=12)

    ax.set_xticks(np.arange(-0.5, len(feature_levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(symptom_levels), 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(sig_mat.shape[0]):
        for c in range(sig_mat.shape[1]):
            if sig_mat[r, c]:
                ax.text(c, r, "*", ha="center", va="center", color="#111111", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("partial_r")
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "csv_file": str(RESULTS_ROOT / dataset / "all_results.csv"),
        "output_file": str(output_pdf),
        "significant_hits": int(sum(parse_bool(row["significant_fdr"]) for row in rows)),
    }


def main() -> None:
    summary_rows = []
    for dataset in ["HX", "Tt"]:
        rows = read_results_csv(RESULTS_ROOT / dataset / "all_results.csv")
        summary_rows.append(save_heatmap(dataset, rows))

    summary_path = RESULTS_ROOT / "heatmap_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["csv_file", "output_file", "significant_hits"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
