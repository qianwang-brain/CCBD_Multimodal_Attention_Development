"""Permutation validation of FDR-screened LightGBM raw-score predictions.

Uses the same nested-CV evaluation and shared hyperparameter search protocol as
predict_clinical_scores.py. Every permutation repeats inner-fold tuning.
Regenerate predictions before using this new analysis protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import predict_clinical_scores as prediction
import numpy as np
import pandas as pd
import scipy.io as sio


def empirical_p(null_r: np.ndarray, observed_r: float, alternative: str) -> float:
    null_r = np.asarray(null_r, dtype=float)
    if not len(null_r) or not np.all(np.isfinite(null_r)) or not np.isfinite(observed_r):
        raise ValueError('Observed and permutation correlations must all be finite.')
    if alternative == 'two-sided':
        extreme = np.abs(null_r) >= abs(observed_r)
    elif alternative == 'greater':
        extreme = null_r >= observed_r
    elif alternative == 'less':
        extreme = null_r <= observed_r
    elif alternative == 'legacy-signed':
        extreme = null_r >= observed_r if observed_r >= 0 else null_r <= observed_r
    else:
        raise ValueError(f'Unknown alternative: {alternative}')
    return float((1 + np.count_nonzero(extreme)) / (1 + len(null_r)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('prediction_results'))
    parser.add_argument('--data', type=Path, nargs='+', default=prediction.DEFAULT_DATASETS)
    parser.add_argument('--n-repeats', type=int, default=1000)
    parser.add_argument('--alternative', choices=['two-sided', 'greater', 'less', 'legacy-signed'], default='legacy-signed',
                        help='Default legacy-signed uses the tail determined by the observed r sign. Two-sided uses |r|; greater/less require a direction chosen in advance.')
    parser.add_argument('--outdir', type=Path, default=None)
    args = parser.parse_args()
    if args.n_repeats <= 0:
        parser.error('--n-repeats must be positive.')
    if args.alternative == 'legacy-signed':
        warnings.warn('Legacy mode chooses the tail from the observed sign; interpret as historical output reproduction, not a prespecified directional test.')
    outdir = args.outdir or Path(f'prediction_permutation_{args.n_repeats}x_{args.alternative}')
    outdir.mkdir(parents=True, exist_ok=True)
    actual = pd.read_csv(args.results_dir / 'clinical_prediction_results_all.csv')
    required = {'dataset', 'target', 'label_filter', 'model_family', 'two_model_role', 'pearson_r', 'pearson_q', 'n_valid'}
    if not required.issubset(actual.columns):
        raise ValueError('Run predict_clinical_scores.py first; results must include FDR q values.')
    actual = actual[(actual.model_family == 'LightGBM') &
                    (actual.two_model_role == 'model1_age_sex_brain_raw_y') &
                    (actual.label_filter == 'all')].copy()
    if actual.duplicated(['dataset', 'target']).any():
        raise ValueError('Duplicate prediction results.')
    data_paths = {p.stem: p for p in args.data}
    manifest = json.loads((args.results_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    old_hashes = set(manifest['input_sha256'].values())
    for path in data_paths.values():
        if hashlib.sha256(path.read_bytes()).hexdigest() not in old_hashes:
            raise ValueError(f'Input {path.name} differs from the prediction run.')
    if (manifest['lightgbm_default_parameters'] != prediction.LIGHTGBM_DEFAULT_PARAMS
            or manifest.get('historical_target_overrides') != {}
            or manifest.get('parameter_policy') != 'nested_inner_cv_lightgbm'
            or manifest.get('lightgbm_search') != prediction.lightgbm_search_spec()):
        raise ValueError('Model parameters differ from the prediction run; regenerate predictions.')
    candidates = actual[actual.pearson_q < prediction.SIGNIFICANCE_ALPHA]
    rng = np.random.default_rng(prediction.RANDOM_STATE)
    null_rows, summary_rows = [], []
    for _, row in candidates.iterrows():
        if row.dataset not in data_paths:
            raise ValueError(f'Missing input for {row.dataset}.')
        mat = sio.loadmat(data_paths[row.dataset])
        brain = np.asarray(mat['feature'], dtype=float)
        inf = np.asarray(mat['inf'], dtype=float)
        scores = np.asarray(mat['scores'], dtype=float)
        names = prediction.matlab_cell_to_str_list(mat['data_name'])
        target_idx = names.index(row.target)
        nuisance = np.column_stack([inf[:, 1], inf[:, 0]])
        x_all = np.column_stack([nuisance, brain])
        valid = np.isfinite(scores[:, target_idx]) & np.all(np.isfinite(x_all), axis=1)
        x, y, cov = x_all[valid], scores[valid, target_idx], nuisance[valid]
        if len(y) != int(row.n_valid):
            raise ValueError('Prediction and permutation complete-case samples differ.')

        def evaluate(outcome):
            metrics, _ = prediction.evaluate_setting(x, outcome, cov, 'LightGBM', 'raw_y',
                                                      row.dataset, row.target, 'model1_age_sex_brain_raw_y')
            return metrics['pearson_r']

        observed = evaluate(y)
        if not np.isclose(observed, row.pearson_r, atol=1e-9, rtol=1e-7):
            raise ValueError('Recomputed observed correlation differs; check data, dependencies and model settings.')
        null_r = np.empty(args.n_repeats)
        for repeat in range(args.n_repeats):
            null_r[repeat] = evaluate(rng.permutation(y))
            null_rows.append({'dataset': row.dataset, 'target': row.target, 'repeat': repeat + 1, 'cv_r': null_r[repeat]})
            if (repeat + 1) % 50 == 0:
                print(f'{row.dataset} / {row.target}: {repeat + 1}/{args.n_repeats}', flush=True)
        p = empirical_p(null_r, observed, args.alternative)
        summary_rows.append({'dataset': row.dataset, 'target': row.target, 'n_valid': len(y),
                             'pearson_r': observed, 'pearson_q': row.pearson_q, 'permutation_p': p,
                             'permutation_p_abs': empirical_p(null_r, observed, 'two-sided'),
                             'alternative': args.alternative, 'n_permutations': args.n_repeats,
                             'significant_fdr_and_permutation': bool(row.pearson_q < .05 and p < .05)})
    columns = ['dataset', 'target', 'n_valid', 'pearson_r', 'pearson_q', 'permutation_p',
               'permutation_p_abs', 'alternative', 'n_permutations', 'significant_fdr_and_permutation']
    summary = pd.DataFrame(summary_rows, columns=columns)
    summary.to_csv(outdir / 'permutation_summary.csv', index=False, encoding='utf-8-sig')
    summary[summary.significant_fdr_and_permutation.astype(bool)].to_csv(outdir / 'significant_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(null_rows, columns=['dataset', 'target', 'repeat', 'cv_r']).to_csv(outdir / 'permutation_null_results.csv', index=False, encoding='utf-8-sig')
    (outdir / 'run_manifest.json').write_text(json.dumps({
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data},
        'prediction_manifest': manifest, 'n_permutations': args.n_repeats,
        'alternative': args.alternative, 'seed': prediction.RANDOM_STATE,
        'tail_selection': 'right if observed r>=0, left otherwise' if args.alternative == 'legacy-signed' else args.alternative,
        'candidate_selection': 'all raw-score LightGBM targets with BH q<0.05',
        'null': 'shuffle y, keep X and fold splits fixed; repeat inner tuning and outer fitting for every permutation',
        'tuning_scope': 'inner folds of each outer training set only; same search protocol as observed data',
        'python': sys.version}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
