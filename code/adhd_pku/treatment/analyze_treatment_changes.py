from __future__ import annotations

import sys
import argparse
import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYDEPS = ROOT / "pydeps"
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.stats import false_discovery_control


INPUT_FILES = [
    ("HX", ROOT / "data_follow_finall_HX.mat"),
    ("Tt", ROOT / "data_follow_finall_Tt.mat"),
]
OUTPUT_ROOT = ROOT / "treatment_change_results"
MEDICATION_TYPE = 1


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


def fisher_ci(r_value: float, n_eff: int, alpha: float = 0.05) -> tuple[float, float]:
    if n_eff <= 3 or abs(r_value) >= 1:
        return np.nan, np.nan
    z_value = np.arctanh(r_value)
    se = 1.0 / np.sqrt(n_eff - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z_value - z_crit * se)), float(np.tanh(z_value + z_crit * se))


def build_delta_frame(values1: np.ndarray, values2: np.ndarray, prefix: str, index: list[str]) -> pd.DataFrame:
    if values1.shape != values2.shape or values1.ndim != 2 or values1.shape[0] != len(index):
        raise ValueError("Baseline/follow-up feature matrices must have matching rows and columns.")
    delta = values2 - values1
    columns = [f"{prefix}_{idx + 1}" for idx in range(delta.shape[1])]
    return pd.DataFrame(delta, columns=columns, index=index)


def fit_partial_corr(
    symptom_improvement: pd.Series,
    feature_delta: pd.Series,
    covariates: pd.DataFrame,
    modality: str,
    feature_index: int,
    symptom: str,
) -> dict[str, float] | None:
    data = pd.concat(
        [symptom_improvement.rename("symptom_improvement"), feature_delta.rename("feature_delta"), covariates],
        axis=1,
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < len(covariates.columns) + 6:
        return None
    if data["symptom_improvement"].std(ddof=0) == 0 or data["feature_delta"].std(ddof=0) == 0:
        return None

    cov = data[covariates.columns].to_numpy(dtype=float) if len(covariates.columns) else np.empty((len(data), 0))
    x_res = residualize(data["feature_delta"].to_numpy(dtype=float), cov)
    y_res = residualize(data["symptom_improvement"].to_numpy(dtype=float), cov)

    r_value, p_value = stats.pearsonr(x_res, y_res)
    if np.isnan(r_value):
        return None

    cov_design = np.column_stack([np.ones(len(data)), cov])
    n_covariates_effective = int(np.linalg.matrix_rank(cov_design) - 1)
    df = len(data) - n_covariates_effective - 2
    if df <= 0 or np.std(x_res) < 1e-12 or np.std(y_res) < 1e-12:
        return None
    t_value = r_value * np.sqrt(df / max(1e-12, 1 - r_value**2))
    p_value = 2 * stats.t.sf(abs(t_value), df)
    ci_low, ci_high = fisher_ci(float(r_value), len(data) - n_covariates_effective)

    return {
        "modality": modality,
        "feature_index": feature_index,
        "symptom": symptom,
        "n": int(len(data)),
        "n_covariates": int(len(covariates.columns)),
        "n_covariates_effective": n_covariates_effective,
        "df": df,
        "partial_r": float(r_value),
        "t_value": float(t_value),
        "p_value": float(p_value),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def analyze_symptoms(symptom_df: pd.DataFrame, feature_df: pd.DataFrame, covariates: pd.DataFrame, modality: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for feature_col in feature_df.columns:
        feature_index = int(feature_col.split("_")[-1])
        for symptom in symptom_df.columns:
            result = fit_partial_corr(
                symptom_df[symptom],
                feature_df[feature_col],
                covariates,
                modality,
                feature_index,
                symptom,
            )
            if result is not None:
                rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_fdr_within_symptom"] = np.nan
    for symptom, idx in out.groupby("symptom").groups.items():
        qvals = false_discovery_control(out.loc[idx, "p_value"].to_numpy(float))
        out.loc[idx, "p_fdr_within_symptom"] = qvals

    out["significant_fdr"] = out["p_fdr_within_symptom"] < 0.05
    return out.sort_values(["symptom", "p_fdr_within_symptom", "p_value"]).reset_index(drop=True)


def load_subject_ids(mat: dict) -> list[str]:
    subject_key = "sub_final2" if "sub_final2" in mat else "sub_f_final"
    return [decode_mat_string(x) for x in mat[subject_key].reshape(-1)]


def summarize_results(dataset_name: str, all_results: pd.DataFrame, symptom_names: list[str], n_subjects: int) -> str:
    lines = [
        f"Dataset: {dataset_name}",
        f"Medication filter: data1(:,1) == {MEDICATION_TYPE}",
        "Symptom improvement: data1 - data2",
        "Feature change: modality2 - modality1",
        "Covariates: GMV/EEG -> age + sex; FCS -> age + sex + mfd",
        f"Symptoms analyzed: {len(symptom_names)}",
        f"Subjects analyzed: {n_subjects}",
        "",
    ]
    for modality in ["gmv", "fcs", "eeg"]:
        part = all_results[all_results["modality"] == modality]
        if part.empty:
            continue
        lines.append(
            f"{modality}: FDR hits={int(part['significant_fdr'].sum())}, nominal hits={int((part['p_value'] < 0.05).sum())}"
        )
        best = part.sort_values(["p_fdr_within_symptom", "p_value"]).head(10)
        for _, row in best.iterrows():
            lines.append(
                f"  - {row['symptom']} / feature {int(row['feature_index'])}: "
                f"partial_r={row['partial_r']:.3f}, p={row['p_value']:.4f}, q={row['p_fdr_within_symptom']:.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def run_one_dataset(dataset_name: str, mat_path: Path) -> pd.DataFrame:
    out_dir = OUTPUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = sio.loadmat(mat_path)
    subject_ids = load_subject_ids(mat)
    all_names = [decode_mat_string(x) for x in mat["data1_name"].reshape(-1)]
    symptom_names = all_names[1:]

    data1 = mat["data1"]
    data2 = mat["data2"]
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError('Duplicate participant IDs.')
    if data1.shape != data2.shape or data1.shape[0] != len(subject_ids) or data1.shape[1] != len(all_names):
        raise ValueError('Baseline/follow-up symptom dimensions and labels must match.')
    if mat['inf'].shape != (len(subject_ids), 2):
        raise ValueError('Expected inf columns sex, age and aligned participant rows.')
    keep_mask = data1[:, 0] == float(MEDICATION_TYPE)

    keep_ids = [subject_ids[idx] for idx, keep in enumerate(keep_mask) if keep]
    improvement = pd.DataFrame(data1[keep_mask, 1:] - data2[keep_mask, 1:], columns=symptom_names, index=keep_ids)
    cov_base = pd.DataFrame(mat["inf"][keep_mask, :], columns=["sex", "age"], index=keep_ids)

    gmv_features = build_delta_frame(mat["gmv1"][keep_mask, :], mat["gmv2"][keep_mask, :], "gmv", keep_ids)
    fcs_features = build_delta_frame(mat["fcs1"][keep_mask, :], mat["fcs2"][keep_mask, :], "fcs", keep_ids)
    eeg_features = build_delta_frame(mat["eeg1"][keep_mask, :], mat["eeg2"][keep_mask, :], "eeg", keep_ids)

    mfd_series = pd.Series(mat["mfd"][keep_mask].reshape(-1), index=keep_ids, name="mfd")

    gmv_cov = cov_base.copy()
    eeg_cov = cov_base.copy()
    fcs_cov = cov_base.copy()
    fcs_cov["mfd"] = mfd_series

    results_frames = [
        analyze_symptoms(improvement, gmv_features, gmv_cov, "gmv"),
        analyze_symptoms(improvement, fcs_features, fcs_cov, "fcs"),
        analyze_symptoms(improvement, eeg_features, eeg_cov, "eeg"),
    ]
    all_results = pd.concat(results_frames, ignore_index=True)
    if all_results.empty:
        raise ValueError(f"No estimable tests after medication filtering for {dataset_name}.")
    all_results.insert(0, "dataset", dataset_name)

    fdr_hits = all_results[all_results["significant_fdr"]].copy()
    nominal_hits = all_results[all_results["p_value"] < 0.05].copy()
    top_hits = (
        all_results.sort_values(["modality", "symptom", "p_fdr_within_symptom", "p_value"])
        .groupby(["modality", "symptom"], as_index=False)
        .head(5)
        .reset_index(drop=True)
    )

    cov_summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "modality": "gmv",
                "covariates": "age, sex",
                "n_features": int(gmv_features.shape[1]),
                "n_symptoms": len(symptom_names),
                "n_subjects": int(improvement.shape[0]),
            },
            {
                "dataset": dataset_name,
                "modality": "fcs",
                "covariates": "age, sex, mfd",
                "n_features": int(fcs_features.shape[1]),
                "n_symptoms": len(symptom_names),
                "n_subjects": int(improvement.shape[0]),
            },
            {
                "dataset": dataset_name,
                "modality": "eeg",
                "covariates": "age, sex",
                "n_features": int(eeg_features.shape[1]),
                "n_symptoms": len(symptom_names),
                "n_subjects": int(improvement.shape[0]),
            },
        ]
    )

    all_results.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8-sig")
    fdr_hits.to_csv(out_dir / "fdr_hits.csv", index=False, encoding="utf-8-sig")
    nominal_hits.to_csv(out_dir / "nominal_hits.csv", index=False, encoding="utf-8-sig")
    top_hits.to_csv(out_dir / "top_hits.csv", index=False, encoding="utf-8-sig")
    cov_summary.to_csv(out_dir / "covariate_summary.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_dir / "results.xlsx", engine="openpyxl") as writer:
        all_results.to_excel(writer, sheet_name="all_results", index=False)
        fdr_hits.to_excel(writer, sheet_name="fdr_hits", index=False)
        nominal_hits.to_excel(writer, sheet_name="nominal_hits", index=False)
        top_hits.to_excel(writer, sheet_name="top_hits", index=False)
        cov_summary.to_excel(writer, sheet_name="covariates", index=False)

    (out_dir / "summary.txt").write_text(
        summarize_results(dataset_name, all_results, symptom_names, int(improvement.shape[0])),
        encoding="utf-8-sig",
    )

    return all_results


def main() -> None:
    global INPUT_FILES, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description='Treatment brain-feature change versus symptom improvement: partial correlations and within-symptom/modality BH-FDR.')
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    parser.add_argument('--outdir', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    INPUT_FILES = [(name, args.data_dir / path.name) for name, path in INPUT_FILES]
    OUTPUT_ROOT = args.outdir
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_frames = [run_one_dataset(dataset_name, mat_path) for dataset_name, mat_path in INPUT_FILES]
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTPUT_ROOT / "combined_all_results.csv", index=False, encoding="utf-8-sig")
    manifest = {'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for _, p in INPUT_FILES},
                'medication_code': MEDICATION_TYPE, 'symptom_change': 'baseline minus follow-up',
                'brain_change': 'follow-up minus baseline', 'p_value_df': 'n minus effective covariate rank minus 2',
                'fdr_family': 'features within each dataset x modality x symptom', 'python': sys.version}
    (OUTPUT_ROOT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == "__main__":
    main()
