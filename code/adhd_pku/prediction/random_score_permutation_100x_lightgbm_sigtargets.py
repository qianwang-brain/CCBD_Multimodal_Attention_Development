from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

PYDEPS = Path(__file__).resolve().parent / 'pydeps'
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd
import scipy.io as sio
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict

RANDOM_STATE = 42
OUTER_SPLITS = 10
DEFAULT_N_REPEATS = 100
SEX_COL = 0
AGE_COL = 1
TARGETS_BY_DATASET = {
    'data_final_used_HX_predict': ['DQ_IA', 'DQ_HI', 'DQ_TO', 'M情感控制', 'M工作记忆'],
    'data_final_used_Tt_predict': ['DQ_IA', 'DQ_TO', 'M情感控制', 'M组织', 'M行为管理指数'],
}
DATASET_PATHS = {
    'data_final_used_HX_predict': Path('data_final_used_HX_predict.mat'),
    'data_final_used_Tt_predict': Path('data_final_used_Tt_predict.mat'),
}
ACTUAL_RESULTS_CSV = Path(
    'analysis_outputs_two_model_multi_family_allsubjects_svr_lgbm_tuned_dqia'
) / 'two_model_multi_family_paired_all.csv'

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
    ('data_final_used_Tt_predict', 'DQ_IA'): {
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


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2:
        return 0.0, np.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r, p = stats.pearsonr(y_true, y_pred)
    return (0.0 if np.isnan(r) else float(r), np.nan if np.isnan(p) else float(p))


def choose_cv_splits(n_samples: int, requested_splits: int, minimum: int = 2) -> int:
    return max(minimum, min(requested_splits, n_samples))


def build_lightgbm_model(dataset_name: str, target_name: str) -> LGBMRegressor:
    params = dict(LIGHTGBM_DEFAULT_PARAMS)
    override_key = (dataset_name, target_name)
    if override_key in LIGHTGBM_OVERRIDES:
        params.update(LIGHTGBM_OVERRIDES[override_key])
    return LGBMRegressor(**params)


def build_predictors_raw(brain_feat: np.ndarray, inf: np.ndarray) -> np.ndarray:
    age = np.asarray(inf[:, AGE_COL], dtype=float).reshape(-1, 1)
    sex = np.asarray(inf[:, SEX_COL], dtype=float).reshape(-1, 1)
    brain_feat = np.asarray(brain_feat, dtype=float)
    return np.column_stack([age, sex, brain_feat])


def evaluate_target(X: np.ndarray, y: np.ndarray, dataset_name: str, target_name: str) -> tuple[float, float]:
    outer_splits = choose_cv_splits(len(y), OUTER_SPLITS)
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    estimator = clone(build_lightgbm_model(dataset_name=dataset_name, target_name=target_name))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y_pred = cross_val_predict(estimator, X, y, cv=outer_cv, n_jobs=-1)
    return safe_pearson(y, y_pred)


def empirical_signed_p(perm_r: np.ndarray, real_r: float) -> float:
    if real_r >= 0:
        return float((1 + np.sum(perm_r >= real_r)) / (1 + len(perm_r)))
    return float((1 + np.sum(perm_r <= real_r)) / (1 + len(perm_r)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='LightGBM score-permutation test for previously significant all-subject targets.'
    )
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=DEFAULT_N_REPEATS,
        help=f'Number of score permutations to run (default: {DEFAULT_N_REPEATS}).',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=None,
        help='Optional output directory. Defaults to the historical 100x directory for 100 repeats, '
             'or a sibling *_<N>x directory for other repeat counts.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_repeats <= 0:
        raise ValueError('--n-repeats must be a positive integer.')

    if args.outdir is not None:
        outdir = args.outdir
    elif args.n_repeats == DEFAULT_N_REPEATS:
        outdir = Path('analysis_outputs_score_permutation_lightgbm_sigtargets_tuned_dqia')
    else:
        outdir = Path(f'analysis_outputs_score_permutation_lightgbm_sigtargets_tuned_dqia_{args.n_repeats}x')
    outdir.mkdir(exist_ok=True)

    actual_df = pd.read_csv(ACTUAL_RESULTS_CSV)
    actual_df = actual_df[
        (actual_df['model_family'] == 'LightGBM')
        & (actual_df['label_filter'] == 'all')
    ].copy()

    dataset_cache: dict[str, dict[str, object]] = {}
    for dataset_name, data_path in DATASET_PATHS.items():
        mat = sio.loadmat(data_path)
        dataset_cache[dataset_name] = {
            'feature': np.asarray(mat['feature'], dtype=float),
            'inf': np.asarray(mat['inf'], dtype=float),
            'scores': np.asarray(mat['scores'], dtype=float),
            'names': matlab_cell_to_str_list(mat['data_name']),
        }

    rng = np.random.default_rng(RANDOM_STATE)
    permutation_rows = []

    for repeat in range(1, args.n_repeats + 1):
        print(f'Repeat {repeat}/{args.n_repeats}')
        for dataset_name, target_list in TARGETS_BY_DATASET.items():
            cache = dataset_cache[dataset_name]
            X_all = build_predictors_raw(cache['feature'], cache['inf'])
            names = cache['names']
            scores = cache['scores']

            for target in target_list:
                target_idx = names.index(target)
                valid = ~np.isnan(scores[:, target_idx]) & ~np.isnan(np.sum(X_all, axis=1))
                X = X_all[valid]
                y = scores[valid, target_idx]
                y_perm = rng.permutation(y)
                r, p = evaluate_target(X=X, y=y_perm, dataset_name=dataset_name, target_name=target)
                permutation_rows.append({
                    'repeat': repeat,
                    'dataset': dataset_name,
                    'target': target,
                    'target_type': 'raw_y_score_permutation',
                    'model_family': 'LightGBM',
                    'cv_r': r,
                    'cv_abs_r': abs(r),
                    'cv_r_p_value': p,
                    'n_subjects': int(len(y)),
                    'n_brain_features': int(cache['feature'].shape[1]),
                    'n_total_predictors': int(X.shape[1]),
                })

    permutation_df = pd.DataFrame(permutation_rows)
    permutation_df.to_csv(outdir / 'score_permutation_lightgbm_sigtargets_all_results.csv', index=False, encoding='utf-8-sig')

    summary_rows = []
    for dataset_name, target_list in TARGETS_BY_DATASET.items():
        actual_sub = actual_df[actual_df['dataset'] == dataset_name].copy()
        for target in target_list:
            actual_row = actual_sub.loc[actual_sub['target'] == target]
            if actual_row.empty:
                raise ValueError(f'Missing actual row for {dataset_name} - {target}')
            actual_row = actual_row.iloc[0]

            real_r = float(actual_row['pearson_r_model1_age_sex_brain_raw_y'])
            real_p = float(actual_row['pearson_p_model1_age_sex_brain_raw_y'])
            real_abs_r = abs(real_r)
            real_r_model2 = float(actual_row['pearson_r_model2_brain_residual_y'])
            real_p_model2 = float(actual_row['pearson_p_model2_brain_residual_y'])

            subset = permutation_df[(permutation_df['dataset'] == dataset_name) & (permutation_df['target'] == target)]
            perm_r = subset['cv_r'].to_numpy(float)
            perm_abs_r = subset['cv_abs_r'].to_numpy(float)

            empirical_p_signed = empirical_signed_p(perm_r, real_r)
            empirical_p_abs = float((1 + np.sum(perm_abs_r >= real_abs_r)) / (1 + len(perm_abs_r)))

            summary_rows.append({
                'dataset': dataset_name,
                'target': target,
                'model_family': 'LightGBM',
                'real_model1_pearson_r': real_r,
                'real_model1_pearson_abs_r': real_abs_r,
                'real_model1_pearson_p': real_p,
                'real_model2_pearson_r': real_r_model2,
                'real_model2_pearson_p': real_p_model2,
                'both_nominally_significant': bool(actual_row['both_nominally_significant']),
                'score_perm_mean_cv_r': float(np.mean(perm_r)),
                'score_perm_std_cv_r': float(np.std(perm_r, ddof=1)) if len(perm_r) > 1 else np.nan,
                'score_perm_min_cv_r': float(np.min(perm_r)),
                'score_perm_max_cv_r': float(np.max(perm_r)),
                'score_perm_mean_cv_abs_r': float(np.mean(perm_abs_r)),
                'score_perm_std_cv_abs_r': float(np.std(perm_abs_r, ddof=1)) if len(perm_abs_r) > 1 else np.nan,
                'score_perm_min_cv_abs_r': float(np.min(perm_abs_r)),
                'score_perm_max_cv_abs_r': float(np.max(perm_abs_r)),
                'real_minus_score_perm_mean_r': float(real_r - np.mean(perm_r)),
                'real_abs_r_minus_score_perm_mean_abs_r': float(real_abs_r - np.mean(perm_abs_r)),
                'empirical_p_signed_tail': empirical_p_signed,
                'empirical_p_abs_tail': empirical_p_abs,
                'actual_stronger_than_score_permutation_signed_at_0_05': bool(empirical_p_signed < 0.05),
                'actual_stronger_than_score_permutation_abs_at_0_05': bool(empirical_p_abs < 0.05),
                'n_score_permutations': int(len(perm_r)),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / 'score_permutation_lightgbm_sigtargets_summary.csv', index=False, encoding='utf-8-sig')

    kept = summary_df[summary_df['actual_stronger_than_score_permutation_signed_at_0_05']].copy()
    kept.to_csv(outdir / 'score_permutation_lightgbm_sigtargets_kept_signed_0_05.csv', index=False, encoding='utf-8-sig')

    report_lines = [
        'LightGBM score permutation test for previously significant all-subject targets',
        f'N_REPEATS={args.n_repeats}',
        'Null model: shuffle symptom scores y; keep the original HX/Tt feature matrices fixed.',
        'Significance decision for permutation uses signed-tail empirical p aligned with the sign of the real model-1 r.',
        '',
        summary_df.to_string(index=False),
        '',
        'Targets kept at empirical signed-tail p < 0.05:',
        kept.to_string(index=False) if not kept.empty else 'None',
    ]
    (outdir / 'report.txt').write_text('\n'.join(report_lines), encoding='utf-8')

    print(f'Output directory: {outdir}')
    print(summary_df.to_string(index=False))
    print('\nKept targets (signed-tail empirical p < 0.05):')
    if kept.empty:
        print('None')
    else:
        print(kept[['dataset', 'target', 'empirical_p_signed_tail', 'real_model1_pearson_r']].to_string(index=False))


if __name__ == '__main__':
    main()
