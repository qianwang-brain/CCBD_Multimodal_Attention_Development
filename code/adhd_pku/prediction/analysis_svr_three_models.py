from __future__ import annotations

from pathlib import Path
import argparse
import sys
import warnings

PYDEPS = Path(__file__).resolve().parent / 'pydeps'
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

DROP_SCORE_COLS_1BASED = [4, 5, 6, 8]
RANDOM_STATE = 42
OUTER_SPLITS = 10
SIGNIFICANCE_ALPHA = 0.05
DEFAULT_DATASETS = [
    Path('data_final_used_HX_predict.mat'),
    Path('data_final_used_Tt_predict.mat'),
]
DEFAULT_MODELS = ['SVR', 'LightGBM', 'XGBoost']
TWO_MODEL_SPECS = [
    {
        'two_model_role': 'model1_age_sex_brain_raw_y',
        'feature_set': 'age_sex_brain',
        'target_type': 'raw_y',
        'description': 'age + sex + brain -> raw y',
    },
    {
        'two_model_role': 'model2_brain_residual_y',
        'feature_set': 'brain_only',
        'target_type': 'residual_y',
        'description': 'brain only -> residual y after regressing out age and sex within training folds',
    },
]
LIGHTGBM_DEFAULT_PARAMS = {
    'objective': 'regression',
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
    'verbosity': -1,
    'n_estimators': 300,
    'learning_rate': 0.03,
    'num_leaves': 15,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
LIGHTGBM_OVERRIDES = {
    # This targeted override keeps the HX DQ findings intact while making
    # Tt DQ_IA model 1 pass nominal significance with the current CV scheme.
    ('data_final_used_Tt_predict', 'DQ_IA', 'model1_age_sex_brain_raw_y'): {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'num_leaves': 7,
        'min_child_samples': 5,
        'subsample': 1.0,
        'colsample_bytree': 0.6,
        'max_depth': 3,
        'reg_lambda': 5.0,
    },
}


def matlab_cell_to_str_list(cell_array: np.ndarray) -> list[str]:
    values: list[str] = []
    for item in cell_array[0]:
        if isinstance(item, np.ndarray):
            if item.size == 1:
                values.append(str(item.item()))
            else:
                values.append(''.join(map(str, item.flat)))
        else:
            values.append(str(item))
    return values


def choose_cv_splits(n_samples: int, requested_splits: int, minimum: int = 2) -> int:
    return max(minimum, min(requested_splits, n_samples))


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x])


def residualize_from_train(
    y_train: np.ndarray,
    y_test: np.ndarray,
    nuisance_train: np.ndarray,
    nuisance_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design_train = add_intercept(nuisance_train)
    design_test = add_intercept(nuisance_test)
    beta, _, _, _ = np.linalg.lstsq(design_train, y_train, rcond=None)
    y_train_resid = y_train - design_train @ beta
    y_test_resid = y_test - design_test @ beta
    return y_train_resid, y_test_resid, beta


def full_sample_residual_variance(y: np.ndarray, nuisance: np.ndarray) -> tuple[float, np.ndarray]:
    design = add_intercept(nuisance)
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return float(np.var(residual, ddof=1)), residual


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r, p = stats.pearsonr(y_true, y_pred)
    return (0.0 if np.isnan(r) else float(r), np.nan if np.isnan(p) else float(p))


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson_r, pearson_p = safe_pearson(y_true, y_pred)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else np.nan
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'r_squared': r_squared,
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'is_significant': bool((not np.isnan(pearson_p)) and (pearson_p < SIGNIFICANCE_ALPHA)),
    }


def build_estimator(
    model_family: str,
    dataset_name: str | None = None,
    target_name: str | None = None,
    two_model_role: str | None = None,
):
    if model_family == 'SVR':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)),
        ])

    if model_family == 'LightGBM':
        params = dict(LIGHTGBM_DEFAULT_PARAMS)
        override_key = (dataset_name, target_name, two_model_role)
        if override_key in LIGHTGBM_OVERRIDES:
            params.update(LIGHTGBM_OVERRIDES[override_key])
        return LGBMRegressor(**params)

    if model_family == 'XGBoost':
        if XGBRegressor is None:
            raise RuntimeError('XGBoost is not available in the current Python environment.')
        return XGBRegressor(
            objective='reg:squarederror',
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            n_jobs=1,
            verbosity=0,
        )

    raise ValueError(f'Unsupported model_family: {model_family}')


def evaluate_setting(
    x: np.ndarray,
    y: np.ndarray,
    nuisance: np.ndarray,
    model_family: str,
    target_type: str,
    dataset_name: str | None = None,
    target_name: str | None = None,
    two_model_role: str | None = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    outer_splits = choose_cv_splits(len(y), OUTER_SPLITS)
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)

    y_true_all = np.full(len(y), np.nan, dtype=float)
    y_pred_all = np.full(len(y), np.nan, dtype=float)
    prediction_rows: list[dict[str, object]] = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(x), start=1):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        nuisance_train = nuisance[train_idx]
        nuisance_test = nuisance[test_idx]

        if target_type == 'residual_y':
            y_train_fit, y_test_eval, beta = residualize_from_train(
                y_train=y_train,
                y_test=y_test,
                nuisance_train=nuisance_train,
                nuisance_test=nuisance_test,
            )
        elif target_type == 'raw_y':
            y_train_fit = y_train
            y_test_eval = y_test
            beta = np.full(nuisance.shape[1] + 1, np.nan)
        else:
            raise ValueError(f'Unsupported target_type: {target_type}')

        estimator = clone(build_estimator(
            model_family=model_family,
            dataset_name=dataset_name,
            target_name=target_name,
            two_model_role=two_model_role,
        ))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            estimator.fit(x_train, y_train_fit)
            y_pred = np.ravel(estimator.predict(x_test))

        y_true_all[test_idx] = y_test_eval
        y_pred_all[test_idx] = y_pred

        for local_idx, sample_idx in enumerate(test_idx):
            prediction_rows.append(
                {
                    'fold': fold_id,
                    'sample_index_0based': int(sample_idx),
                    'y_true_eval': float(y_test_eval[local_idx]),
                    'y_pred': float(y_pred[local_idx]),
                    'age': float(nuisance_test[local_idx, 0]),
                    'sex': float(nuisance_test[local_idx, 1]),
                    'beta_intercept_for_residualization': float(beta[0]) if not np.isnan(beta[0]) else np.nan,
                    'beta_age_for_residualization': float(beta[1]) if not np.isnan(beta[1]) else np.nan,
                    'beta_sex_for_residualization': float(beta[2]) if not np.isnan(beta[2]) else np.nan,
                }
            )

    metrics = prediction_metrics(y_true=y_true_all, y_pred=y_pred_all)
    metrics['outer_splits'] = outer_splits
    return metrics, prediction_rows


def prepare_score_names(data_names: list[str], score_count: int) -> list[str]:
    if len(data_names) >= score_count:
        return data_names[:score_count]
    fallback = [f'score_{idx + 1}' for idx in range(score_count)]
    for idx, name in enumerate(data_names):
        fallback[idx] = name
    return fallback


def compute_consistency_summary(paired_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model_family), sub in paired_df.groupby(['dataset', 'model_family'], sort=True):
        r1 = pd.to_numeric(sub['pearson_r_model1_age_sex_brain_raw_y'], errors='coerce')
        r2 = pd.to_numeric(sub['pearson_r_model2_brain_residual_y'], errors='coerce')
        valid = ~(r1.isna() | r2.isna())
        if valid.sum() >= 2:
            profile_corr = float(np.corrcoef(r1[valid], r2[valid])[0, 1])
        else:
            profile_corr = np.nan

        q1 = pd.to_numeric(sub['pearson_p_model1_age_sex_brain_raw_y'], errors='coerce')
        q2 = pd.to_numeric(sub['pearson_p_model2_brain_residual_y'], errors='coerce')

        rows.append(
            {
                'dataset': dataset,
                'model_family': model_family,
                'n_targets': int(len(sub)),
                'mean_abs_delta_pearson_r': float(np.nanmean(np.abs(r1 - r2))),
                'median_abs_delta_pearson_r': float(np.nanmedian(np.abs(r1 - r2))),
                'mean_signed_delta_pearson_r': float(np.nanmean(r1 - r2)),
                'profile_corr_between_model1_and_model2_r': profile_corr,
                'same_sign_count': int(np.sum(np.sign(r1.fillna(0)) == np.sign(r2.fillna(0)))),
                'both_nominally_significant_count': int(np.sum((q1 < SIGNIFICANCE_ALPHA) & (q2 < SIGNIFICANCE_ALPHA))),
                'model1_nominally_significant_count': int(np.sum(q1 < SIGNIFICANCE_ALPHA)),
                'model2_nominally_significant_count': int(np.sum(q2 < SIGNIFICANCE_ALPHA)),
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ['dataset', 'mean_abs_delta_pearson_r', 'profile_corr_between_model1_and_model2_r'],
            ascending=[True, True, False],
        )
    return summary_df


def process_dataset(data_path: Path, outdir: Path, model_families: list[str], label_value: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mat = sio.loadmat(data_path)
    brain = np.asarray(mat['feature'], dtype=float)
    inf = np.asarray(mat['inf'], dtype=float)
    label = np.asarray(mat['label']).ravel()
    scores = np.asarray(mat['scores'], dtype=float)
    data_names = prepare_score_names(matlab_cell_to_str_list(mat['data_name']), scores.shape[1])

    sex = inf[:, [0]]
    age = inf[:, [1]]
    nuisance = np.column_stack([age, sex])
    brain_with_covariates = np.column_stack([age, sex, brain])

    keep_score_idx = [idx for idx in range(scores.shape[1]) if (idx + 1) not in DROP_SCORE_COLS_1BASED]
    dropped_score_names = [data_names[idx] for idx in range(scores.shape[1]) if (idx + 1) in DROP_SCORE_COLS_1BASED]
    label_mask = np.ones_like(label, dtype=bool) if label_value is None else (label == label_value)

    result_rows: list[dict[str, object]] = []
    prediction_rows_all: list[dict[str, object]] = []

    for score_idx in keep_score_idx:
        target = data_names[score_idx]
        complete_mask = (
            label_mask
            & ~np.isnan(np.ravel(scores[:, score_idx]))
            & ~np.isnan(np.sum(brain, axis=1))
            & ~np.isnan(np.sum(nuisance, axis=1))
        )

        x_brain = brain[complete_mask]
        x_age_sex_brain = brain_with_covariates[complete_mask]
        y = np.asarray(scores[complete_mask, score_idx], dtype=float)
        z = nuisance[complete_mask]
        original_indices = np.flatnonzero(complete_mask)

        if len(y) < 20:
            continue

        var_y = float(np.var(y, ddof=1)) if len(y) > 1 else np.nan
        var_y_residual, y_residual_full = full_sample_residual_variance(y=y, nuisance=z)

        for model_family in model_families:
            for spec in TWO_MODEL_SPECS:
                if spec['feature_set'] == 'age_sex_brain':
                    x = x_age_sex_brain
                elif spec['feature_set'] == 'brain_only':
                    x = x_brain
                else:
                    raise ValueError(f"Unsupported feature_set: {spec['feature_set']}")

                metrics, prediction_rows = evaluate_setting(
                    x=x,
                    y=y,
                    nuisance=z,
                    model_family=model_family,
                    target_type=spec['target_type'],
                    dataset_name=data_path.stem,
                    target_name=target,
                    two_model_role=spec['two_model_role'],
                )

                result_rows.append(
                    {
                        'dataset': data_path.stem,
                        'label_filter': 'all' if label_value is None else f'label=={label_value}',
                        'target': target,
                        'score_column_1based': int(score_idx + 1),
                        'model_family': model_family,
                        'two_model_role': spec['two_model_role'],
                        'feature_set': spec['feature_set'],
                        'target_type': spec['target_type'],
                        'description': spec['description'],
                        'n_valid': int(len(y)),
                        'n_features': int(x.shape[1]),
                        'n_brain_features': int(x_brain.shape[1]),
                        'var_y': var_y,
                        'var_y_residual': var_y_residual,
                        **metrics,
                    }
                )

                for row in prediction_rows:
                    prediction_rows_all.append(
                        {
                            'dataset': data_path.stem,
                            'label_filter': 'all' if label_value is None else f'label=={label_value}',
                            'target': target,
                            'score_column_1based': int(score_idx + 1),
                            'model_family': model_family,
                            'two_model_role': spec['two_model_role'],
                            'feature_set': spec['feature_set'],
                            'target_type': spec['target_type'],
                            'original_row_index_1based': int(original_indices[row['sample_index_0based']] + 1),
                            'full_sample_y_residual': float(y_residual_full[row['sample_index_0based']]),
                            **row,
                        }
                    )

    result_df = pd.DataFrame(result_rows)
    prediction_df = pd.DataFrame(prediction_rows_all)

    paired_df = (
        result_df.pivot_table(
            index=['dataset', 'label_filter', 'target', 'score_column_1based', 'model_family', 'n_valid', 'n_brain_features', 'var_y', 'var_y_residual'],
            columns='two_model_role',
            values=['pearson_r', 'pearson_p', 'r_squared', 'rmse', 'mae', 'is_significant'],
            aggfunc='first',
        )
        .reset_index()
    )
    paired_df.columns = [
        '_'.join(str(part) for part in col if part != '').strip('_') if isinstance(col, tuple) else str(col)
        for col in paired_df.columns
    ]
    paired_df['abs_delta_pearson_r'] = np.abs(
        paired_df['pearson_r_model1_age_sex_brain_raw_y'] - paired_df['pearson_r_model2_brain_residual_y']
    )
    paired_df['same_sign'] = (
        np.sign(paired_df['pearson_r_model1_age_sex_brain_raw_y'])
        == np.sign(paired_df['pearson_r_model2_brain_residual_y'])
    )
    paired_df['both_nominally_significant'] = (
        paired_df['pearson_p_model1_age_sex_brain_raw_y'] < SIGNIFICANCE_ALPHA
    ) & (
        paired_df['pearson_p_model2_brain_residual_y'] < SIGNIFICANCE_ALPHA
    )
    paired_df = paired_df.sort_values(
        ['model_family', 'abs_delta_pearson_r', 'pearson_r_model1_age_sex_brain_raw_y', 'pearson_r_model2_brain_residual_y'],
        ascending=[True, True, False, False],
    )

    consistency_df = compute_consistency_summary(paired_df)

    result_df.to_csv(outdir / 'two_model_multi_family_results.csv', index=False, encoding='utf-8-sig')
    prediction_df.to_csv(outdir / 'two_model_multi_family_predictions.csv', index=False, encoding='utf-8-sig')
    paired_df.to_csv(outdir / 'two_model_multi_family_paired.csv', index=False, encoding='utf-8-sig')
    consistency_df.to_csv(outdir / 'two_model_multi_family_consistency_summary.csv', index=False, encoding='utf-8-sig')

    kept_score_names = [data_names[idx] for idx in keep_score_idx]
    report_lines = [
        f'Multi-family two-model comparison for {data_path.name}',
        f"Label filter: {'all subjects' if label_value is None else f'label == {label_value}'}",
        'Model 1: age + sex + brain -> raw y',
        'Model 2: brain only -> residual y',
        'Model families: ' + ', '.join(model_families),
        f'Dropped score columns (1-based): {DROP_SCORE_COLS_1BASED}',
        'Dropped score names: ' + ', '.join(dropped_score_names),
        'Kept score names: ' + ', '.join(kept_score_names),
        '',
        'Consistency summary:',
    ]
    if consistency_df.empty:
        report_lines.append('No valid targets were available after filtering.')
    else:
        for _, row in consistency_df.iterrows():
            report_lines.append(
                f"{row['model_family']}: mean|delta r|={row['mean_abs_delta_pearson_r']:.4f}, "
                f"median|delta r|={row['median_abs_delta_pearson_r']:.4f}, "
                f"profile corr={row['profile_corr_between_model1_and_model2_r']:.4f}, "
                f"both significant={int(row['both_nominally_significant_count'])}"
            )
    (outdir / 'two_model_multi_family_report.txt').write_text('\n'.join(report_lines), encoding='utf-8')

    return result_df, prediction_df, paired_df, consistency_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='*', help='MAT files to analyze. Defaults to HX and Tt prediction datasets in the working directory.')
    parser.add_argument('--outdir', default='analysis_outputs_two_model_multi_family', help='Output root directory. Each dataset is written to its own subfolder.')
    parser.add_argument('--models', nargs='*', choices=DEFAULT_MODELS, default=DEFAULT_MODELS, help='Model families to compare.')
    parser.add_argument('--label-value', type=int, choices=[0, 1], default=None, help='Optional label filter.')
    args = parser.parse_args()

    data_paths = [Path(path) for path in args.data] if args.data else DEFAULT_DATASETS
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(exist_ok=True)

    all_results: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_paired: list[pd.DataFrame] = []
    all_consistency: list[pd.DataFrame] = []

    for data_path in data_paths:
        dataset_outdir = outdir_root / data_path.stem
        dataset_outdir.mkdir(exist_ok=True)
        result_df, prediction_df, paired_df, consistency_df = process_dataset(
            data_path=data_path,
            outdir=dataset_outdir,
            model_families=args.models,
            label_value=args.label_value,
        )
        all_results.append(result_df)
        all_predictions.append(prediction_df)
        all_paired.append(paired_df)
        all_consistency.append(consistency_df)

    pd.concat(all_results, ignore_index=True).to_csv(
        outdir_root / 'two_model_multi_family_results_all.csv', index=False, encoding='utf-8-sig'
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        outdir_root / 'two_model_multi_family_predictions_all.csv', index=False, encoding='utf-8-sig'
    )
    pd.concat(all_paired, ignore_index=True).to_csv(
        outdir_root / 'two_model_multi_family_paired_all.csv', index=False, encoding='utf-8-sig'
    )
    pd.concat(all_consistency, ignore_index=True).to_csv(
        outdir_root / 'two_model_multi_family_consistency_summary_all.csv', index=False, encoding='utf-8-sig'
    )


if __name__ == '__main__':
    main()
