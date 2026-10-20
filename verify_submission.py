"""Independent integrity and numerical checks for the EC521 submission.

This verifies saved outputs without rerunning a model-selection search. It
never edits original data, scripts, or experiment results. See --help.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKS = []


def check(condition, name, **details):
    if not bool(condition):
        raise AssertionError(f'{name}: {details}')
    CHECKS.append(dict(check=name, status='passed', **details))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def close(actual, expected, name, atol=1e-7, rtol=1e-7):
    check(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True),
          name, max_absolute_difference=float(np.nanmax(
              np.abs(np.asarray(actual) - np.asarray(expected)))))


def verify_originals(root):
    hashes = read_json(HERE / 'checks/original_hashes.json')
    for name, expected in hashes.items():
        path = root / name
        check(path.is_file(), f'original present: {name}')
        check(sha256(path) == expected, f'original SHA256 unchanged: {name}')
    return len(hashes)


def verify_data(path):
    """Compare the production loader with a separately implemented parser."""
    common = import_script(HERE / 'common.py')
    actual = common.load_data(path)
    raw = pd.read_csv(path)
    expected = raw.copy()
    for col in expected:
        expected[col] = expected[col].map(
            lambda x: x.strip().lower() if isinstance(x, str) else x)
        expected[col] = expected[col].map(
            lambda x: 1 if x == 'yes' else 0 if x == 'no' else x)
        expected[col] = pd.to_numeric(expected[col], errors='raise')
    close(actual.to_numpy(float), expected.to_numpy(float), 'independent data parsing')
    check(actual[['pov_exp', 'rspcexp23']].notna().all().all(),
          'targets complete and unaltered')
    check(set(actual.pov_exp.unique()) == {0, 1}, 'target classes valid')
    check((actual.rspcexp23 > 0).all(), 'regression target strictly positive')
    check(actual.electcon.dropna().eq(1).all(), 'electricity observed values converted')
    # Tuple-based grouping is deliberately independent of common.feature_groups.
    feature_names = [x for x in common.FEATURES if x != 'electcon']
    keys = list(actual[feature_names].itertuples(index=False, name=None))
    group_lookup = {}
    groups = np.array([group_lookup.setdefault(k, len(group_lookup)) for k in keys])
    production_groups = common.feature_groups(actual)
    equivalent = all(len(np.unique(production_groups[ids.to_numpy()])) == 1
                     for _, ids in pd.Series(np.arange(len(actual))).groupby(groups))
    check(equivalent, 'independent profile equivalence')
    check(len(np.unique(production_groups)) == len(np.unique(groups)),
          'independent profile-group count', count=len(group_lookup))
    return actual, groups


def verify_manifest(path, frame, groups):
    m = read_json(path)
    n = len(frame)
    check(m['data_sha256'] == frame.attrs['data_sha256'],
          f'dataset digest matches manifest: {path.name}')
    row_ids = np.arange(n)
    _, old_c = train_test_split(row_ids, test_size=.2, random_state=1,
                                stratify=frame.pov_exp)
    _, old_r = train_test_split(row_ids, test_size=.2, random_state=1)
    legacy = set(old_c) | set(old_r)
    holdout = set(m['holdout'])
    dev = set(m['dev'])
    check(len(legacy) == m['historical_union_before_group_extension'],
          'historical union independently reconstructed', count=len(legacy))
    check(legacy <= holdout, 'all historical test rows quarantined')
    expected_holdout = set(np.flatnonzero(np.isin(groups, groups[list(legacy)])))
    check(holdout == expected_holdout, 'holdout profile extension exact')
    check(not holdout & dev and holdout | dev == set(row_ids),
          'development/holdout row partition complete')
    check(not set(groups[list(holdout)]) & set(groups[list(dev)]),
          'development/holdout predictor profiles disjoint')
    check(len(holdout) == len(m['holdout']) == m['historical_holdout_rows'],
          'holdout count and uniqueness')
    check(len(dev) == len(m['dev']) == m['development_rows'],
          'development count and uniqueness')
    evaluations = []
    for number, fold in enumerate(m['folds'], 1):
        check(set(fold) == {'fit', 'stop', 'calibration', 'evaluation'},
              f'fold {number} roles present')
        parts = {name: set(values) for name, values in fold.items()}
        check(set.union(*parts.values()) == dev,
              f'fold {number} partitions cover development rows')
        for name, values in parts.items():
            check(len(values) == len(fold[name]), f'fold {number} {name} unique rows')
            check(frame.pov_exp.iloc[list(values)].nunique() == 2,
                  f'fold {number} {name} has both classes', count=len(values),
                  positives=int(frame.pov_exp.iloc[list(values)].sum()))
        for a, b in itertools.combinations(parts, 2):
            check(not parts[a] & parts[b], f'fold {number}: {a}/{b} row isolation')
            check(not set(groups[list(parts[a])]) & set(groups[list(parts[b])]),
                  f'fold {number}: {a}/{b} profile isolation')
        evaluations.extend(fold['evaluation'])
    check(len(evaluations) == len(set(evaluations)) == len(dev)
          and set(evaluations) == dev, 'each development row evaluated exactly once')
    check(len(m['folds']) == m['n_splits'], 'stored fold count matches configuration')
    return m


def classification_metrics(y, scores, threshold):
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    pred = scores >= threshold
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                average_precision=average_precision_score(y, scores),
                accuracy=accuracy_score(y, pred), n=len(y), positives=int(y.sum()),
                tn=int(((y == 0) & ~pred).sum()), fp=int(((y == 0) & pred).sum()),
                fn=int(((y == 1) & ~pred).sum()), tp=int(((y == 1) & pred).sum()))


def regression_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    residual = pred - y
    return dict(n=len(y), rmse=float(np.sqrt(mean_squared_error(y, pred))),
                mae=mean_absolute_error(y, pred), r2=r2_score(y, pred),
                mean_residual=float(residual.mean()),
                p95_absolute_error=float(np.quantile(abs(residual), .95)),
                max_absolute_error=float(abs(residual).max()),
                negative_prediction_fraction=float((pred < 0).mean()))


def compare_metric_row(row, calculated, name):
    fields = [key for key in calculated if key in row]
    close([float(row[key]) for key in fields], [calculated[key] for key in fields], name)


def independent_threshold(y, scores):
    """Compute the F1 optimum by cumulative confusion counts, not PR-curve API."""
    order = np.argsort(-scores, kind='stable')
    s, target = np.asarray(scores)[order], np.asarray(y, int)[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(target)[ends]
    fp = ends + 1 - tp
    fn = target.sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    # Descending score order means the first tied maximum is most conservative.
    best = np.flatnonzero(np.isclose(f1, f1.max(), atol=1e-12, rtol=0))[0]
    return float(s[ends[best]])


def verify_preprocessing(processor, frame, fit_ids, features, name):
    """Stored imputer/scaler statistics must equal statistics from fitting rows."""
    values = frame.loc[fit_ids, features].to_numpy(float)
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isnan(median), 0, median)
    close(processor.named_steps['imputer'].statistics_, median, f'{name}: fit-only medians')
    filled = np.where(np.isnan(values), median, values)
    scaler = processor.named_steps['scaler']
    close(scaler.mean_, filled.mean(axis=0), f'{name}: fit-only feature means')
    close(scaler.var_, filled.var(axis=0), f'{name}: fit-only feature variances')
    check(int(scaler.n_samples_seen_) == len(fit_ids), f'{name}: scaler fitting row count')


def verify_summary(runs, summary, groups, metrics, name):
    per_fold = runs.groupby(groups + ['fold'])[metrics].mean()
    means = per_fold.groupby(level=groups).mean()
    deviations = per_fold.groupby(level=groups).std(ddof=1)
    indexed = summary.set_index(groups)
    check(set(indexed.index) == set(means.index), f'{name}: summary candidate coverage')
    for metric in metrics:
        close(indexed.loc[means.index, f'{metric}_mean'].to_numpy(), means[metric].to_numpy(),
              f'{name}: {metric} seed-averaged fold mean')
        close(indexed.loc[means.index, f'{metric}_std'].to_numpy(), deviations[metric].to_numpy(),
              f'{name}: {metric} fold SD')


def verify_classification(root, frame, manifest):
    import torch
    module = import_script(HERE / 'classification.py')
    results, models = root / 'results/classification', root / 'models/classification'
    completion = read_json(results / 'completion.json')
    check(completion['complete'] and not completion['smoke'], 'classification full run complete')
    protocol = read_json(results / 'protocol.json')
    saved_split = read_json(results / 'splits/split_manifest.json')
    check(saved_split == manifest, 'classification uses shared exact split manifest')
    cv = pd.read_csv(results / 'cv_metrics.csv')
    keys = ['method', 'fold', 'seed', 'split', 'threshold_mode']
    check(not cv.duplicated(keys).any(), 'classification metric rows unique')
    expected_runs = len(protocol['candidates']) * len(protocol['seeds']) * protocol['folds']
    check(len(cv) == expected_runs * 8, 'classification all candidate/fold/seed/role metrics',
          training_runs=expected_runs)
    for (method, fold, seed), rows in cv.groupby(['method', 'fold', 'seed']):
        run = f'fold{fold}_seed{seed}_{method}'
        folder = models / run
        metadata = read_json(folder / 'metadata.json')
        indices = np.load(folder / 'row_indices.npz')
        predictions = np.load(results / f'{run}_predictions.npz')
        parts = manifest['folds'][int(fold) - 1]
        for split, ids in parts.items():
            check(np.array_equal(indices[split], ids), f'{run}: {split} rows match manifest')
            check(np.array_equal(predictions[f'{split}_row_index'], ids), f'{run}: {split} predictions aligned')
            close(predictions[f'{split}_y'], frame.loc[ids, 'pov_exp'], f'{run}: {split} labels match source')
        score = predictions['evaluation_score']
        check(np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all(),
              f'{run}: finite evaluation probabilities')
        threshold = independent_threshold(predictions['calibration_y'], predictions['calibration_score'])
        close(threshold, metadata['threshold'], f'{run}: threshold from calibration only', atol=1e-12, rtol=0)
        for row in rows.to_dict('records'):
            split = row['split']
            expected_cut = 0.5 if row['threshold_mode'] == 'fixed_0.5' else threshold
            close(row['threshold'], expected_cut, f'{run}: stored threshold', atol=1e-12, rtol=0)
            compare_metric_row(row, classification_metrics(predictions[f'{split}_y'],
                               predictions[f'{split}_score'], expected_cut), f'{run}: {split} {row["threshold_mode"]} metrics')
        sampled = indices['resampled_fit_positions']
        check(((sampled >= 0) & (sampled < len(parts['fit']))).all(), f'{run}: resampling only from fit rows')
        check(len(sampled) == metadata['training_rows_after_resampling'], f'{run}: resampled count')
        steps_per_epoch = int(np.ceil(len(sampled) / metadata['batch_size']))
        check(metadata['optimizer_steps'] == metadata['epochs_executed'] * steps_per_epoch,
              f'{run}: actual optimizer update count')
        check(metadata['retained_optimizer_steps'] == metadata['retained_epoch'] * steps_per_epoch,
              f'{run}: retained checkpoint update count')
        processor = joblib.load(folder / 'preprocessor.joblib')
        verify_preprocessing(processor, frame, parts['fit'], metadata['features'], run)
        history = pd.read_csv(root / 'logs/classification' / f'{run}_history.csv')
        check(len(history) == metadata['epochs_executed'], f'{run}: epoch history complete')
        if not metadata['config']['early_stopping']:
            check(metadata['retained_epoch'] == protocol['max_epochs'], f'{run}: fixed-epoch baseline budget')
    verify_summary(cv.query("split == 'evaluation'"), pd.read_csv(results / 'cv_summary.csv'),
                   ['method', 'threshold_mode'], module.METRICS, 'classification')
    selection_path = results / 'locked_selection.json'
    selection = read_json(selection_path)
    check(not selection['holdout_used_for_selection'], 'classification selection excludes holdout')
    folded = cv.query("split == 'evaluation' and threshold_mode == 'calibrated'").groupby(['method', 'fold']).f1.mean().unstack('method')
    eligible = []
    for method in protocol['candidates']:
        gains = folded[method] - folded['baseline']
        if method != 'baseline' and (gains > 0).sum() >= np.ceil(.8 * len(gains)) and gains.mean() > gains.std(ddof=1) / np.sqrt(len(gains)):
            eligible.append(method)
    chosen = sorted(eligible, key=lambda method: (-folded[method].mean(), method))[0] if eligible else 'baseline'
    check(chosen == selection['method'] == completion['selected'], 'classification selection rule independently recomputed')
    holdout = pd.read_csv(results / 'historical_holdout_predictions.csv')
    check(np.array_equal(holdout.row_index, manifest['holdout']), 'classification holdout rows match quarantine')
    close(holdout.y_true, frame.loc[holdout.row_index, 'pov_exp'], 'classification holdout labels match source')
    metrics = pd.read_csv(results / 'historical_holdout_metrics.csv')
    for row in metrics.to_dict('records'):
        compare_metric_row(row, classification_metrics(holdout.y_true,
                           holdout[f'{row["method"]}_score'], row['threshold']),
                           f'classification holdout {row["method"]}/{row["threshold_mode"]}')
    for method in dict.fromkeys(['baseline', chosen]):
        folder = models / f'final_{method}'
        check(selection_path.stat().st_mtime_ns <= (folder / 'metadata.json').stat().st_mtime_ns,
              f'classification selection record precedes final {method} artifact')
        metadata = read_json(folder / 'metadata.json')
        indices = np.load(folder / 'row_indices.npz')
        check(set(indices['fit']) | set(indices['stop']) | set(indices['calibration']) == set(manifest['dev']),
              f'classification final {method}: all training roles inside development')
        processor = joblib.load(folder / 'preprocessor.joblib')
        verify_preprocessing(processor, frame, indices['fit'], metadata['features'], f'classification final {method}')
        model = module.FFN(len(metadata['features']), metadata['config']['dropout'])
        model.load_state_dict(torch.load(folder / 'weights.pt', weights_only=True, map_location='cpu'))
        ready = processor.transform(frame.loc[holdout.row_index, metadata['features']]).astype(np.float32)
        reloaded = module.predict_scores(model, ready)
        close(reloaded, holdout[f'{method}_score'].to_numpy(), f'classification final {method}: reload parity', atol=1e-7, rtol=1e-7)
    for filename in ['weights.pt', 'preprocessor.joblib', 'metadata.json', 'row_indices.npz']:
        check(sha256(models / 'selected' / filename) == sha256(models / f'final_{chosen}' / filename),
              f'classification selected artifact matches chosen model: {filename}')
    return module, models / 'selected', holdout, chosen


def verify_regression(root, frame, manifest):
    module = import_script(HERE / 'regression.py')
    results, models = root / 'results/regression', root / 'models/regression'
    completion = read_json(results / 'completion.json')
    check(not completion['smoke'], 'regression full run complete')
    protocol = read_json(results / 'protocol.json')
    check(read_json(root / 'split_manifest.json') == manifest, 'regression uses shared exact split manifest')
    runs = pd.read_csv(results / 'cv_runs.csv')
    predictions = pd.read_csv(results / 'cv_predictions.csv')
    keys = ['variant', 'fold', 'seed']
    check(not runs.duplicated(keys).any(), 'regression run keys unique')
    expected_runs = len(protocol['variants']) * len(protocol['seeds']) * protocol['folds']
    check(len(runs) == expected_runs, 'regression all candidate/fold/seed runs present', training_runs=expected_runs)
    for row in runs.to_dict('records'):
        name, fold, seed = row['variant'], row['fold'], row['seed']
        run = f'fold{fold}_seed{seed}_{name}'
        subset = predictions[(predictions.variant == name) & (predictions.fold == fold) & (predictions.seed == seed)]
        parts = manifest['folds'][fold - 1]
        check(np.array_equal(subset.row_index.to_numpy(), parts['evaluation']), f'{run}: evaluation row alignment')
        close(subset.actual.to_numpy(), frame.loc[subset.row_index, 'rspcexp23'].to_numpy(), f'{run}: actual labels match source')
        check(np.isfinite(subset.predicted).all(), f'{run}: finite predictions')
        close(subset.residual.to_numpy(), (subset.predicted - subset.actual).to_numpy(), f'{run}: residual column')
        compare_metric_row(row, regression_metrics(subset.actual, subset.predicted), f'{run}: original-unit metrics recomputed')
        check(row['n_fit'] == len(parts['fit']) + len(parts['calibration']) and row['n_stop'] == len(parts['stop']),
              f'{run}: regression merges fit+calibration only')
        check(row['optimizer_steps'] == row['epochs_run'] * int(np.ceil(row['n_fit'] / 32)), f'{run}: optimizer steps')
        check(row['steps_at_selected_epoch'] == row['selected_epoch'] * int(np.ceil(row['n_fit'] / 32)), f'{run}: retained checkpoint steps')
        history = pd.read_csv(root / 'logs/regression' / f'{run}_history.csv')
        check(len(history) == row['epochs_run'], f'{run}: complete training history')
    verify_summary(runs, pd.read_csv(results / 'cv_summary.csv'), ['variant'], module.METRICS, 'regression')
    selection_path = results / 'selection.json'
    selection = read_json(selection_path)
    digest = sha256(selection_path)
    check(digest == completion['selection_sha256_before_holdout'] == completion['selection_sha256_after_holdout'],
          'regression selection locked before and unchanged after holdout')
    check(not selection['holdout_used_for_selection'], 'regression selection excludes holdout')
    selected = selection['selected_variant']
    check(selected == completion['selected_variant'], 'regression completion matches selection')
    subgroup = pd.read_csv(results / 'cv_subgroup_diagnostics.csv', dtype={'value': str})
    fold_scores = runs.groupby(['variant', 'fold']).rmse.mean().unstack('variant')
    eligible = []
    for variant in protocol['variants']:
        name = variant['name']
        if name == 'original':
            continue
        gains = fold_scores.original - fold_scores[name]
        bad_group = False
        for field in ['rural', 'region']:
            for value, subset in predictions.groupby(field):
                baseline = subset[subset.variant == 'original']
                candidate = subset[subset.variant == name]
                if baseline.row_index.nunique() >= 100:
                    ratio = mean_absolute_error(candidate.actual, candidate.predicted) / mean_absolute_error(baseline.actual, baseline.predicted)
                    bad_group |= ratio > 1.10
        if gains.mean() / fold_scores.original.mean() >= .01 and (gains > 0).sum() >= np.ceil(.8 * len(gains)) and gains.mean() > gains.std(ddof=1) / np.sqrt(len(gains)) and not bad_group:
            eligible.append(name)
    independent_choice = min(eligible, key=lambda name: fold_scores[name].mean()) if eligible else 'original'
    check(independent_choice == selected, 'regression selection rule and group guard independently recomputed')
    holdout = pd.read_csv(results / 'holdout_predictions.csv')
    holdout_metrics = pd.read_csv(results / 'holdout_metrics.csv')
    common = import_script(HERE / 'common.py')
    final_parts = common.final_partitions(frame, manifest['dev'])
    final_fit = np.sort(np.r_[final_parts['fit'], final_parts['calibration']])
    for row in holdout_metrics.to_dict('records'):
        name = row['variant']
        subset = holdout[holdout.variant == name]
        check(np.array_equal(subset.row_index.to_numpy(), manifest['holdout']), f'regression {name}: holdout quarantined rows')
        close(subset.actual.to_numpy(), frame.loc[subset.row_index, 'rspcexp23'].to_numpy(), f'regression {name}: holdout labels')
        compare_metric_row(row, regression_metrics(subset.actual, subset.predicted), f'regression {name}: holdout metrics')
        folder = models / name
        check(selection_path.stat().st_mtime_ns <= (folder / 'metadata.json').stat().st_mtime_ns,
              f'regression selection record precedes final {name} artifact')
        fitted = module.load_model(folder)
        verify_preprocessing(fitted[1], frame, final_fit, fitted[3]['features'], f'regression final {name}')
        target = frame.loc[final_fit, 'rspcexp23'].to_numpy(float)
        if fitted[3]['target'] == 'log1p_standard':
            target = np.log1p(target)
        close(fitted[2].mean_[0], target.mean(), f'regression final {name}: fit-only target mean')
        close(fitted[2].var_[0], target.var(), f'regression final {name}: fit-only target variance')
        reloaded = module.predict_rows(frame, subset.row_index.to_numpy(), fitted)
        close(reloaded, subset.predicted.to_numpy(), f'regression final {name}: reload inference parity')
    # Recompute every subgroup metric directly from per-row predictions.
    for table_name, source in [('cv_subgroup_diagnostics.csv', predictions), ('holdout_subgroup_diagnostics.csv', holdout)]:
        table = pd.read_csv(results / table_name, dtype={'value': str})
        for row in table.to_dict('records'):
            subset = source[(source.variant == row['variant']) & (source.partition == row['partition'])]
            value = row['value']
            if row['group'] == 'upper_5pct_expenditure':
                mask = subset[row['group']].astype(bool) == (value.lower() == 'true')
            else:
                mask = pd.to_numeric(subset[row['group']]) == float(value)
            subset = subset[mask]
            check(len(subset) > 0, f'regression subgroup {row["variant"]}/{row["group"]}/{value}: nonempty')
            compare_metric_row(row, regression_metrics(subset.actual, subset.predicted),
                               f'regression subgroup {row["variant"]}/{row["group"]}/{value}: metrics')
    return module, models / selected, holdout, selected


def verify_inference(module, model_dir, task, frame):
    """Exercise inference on feature-only CSV plus literal binary and missing values."""
    common = import_script(HERE / 'common.py')
    sample = frame.loc[:127, common.FEATURES].copy()
    # Deliberately use pandas string dtype, whitespace, mixed case and a No value.
    sample['electcon'] = sample.electcon.map(lambda x: ' YES ' if pd.notna(x) else np.nan)
    sample.loc[0, 'electcon'] = 'No'
    sample.loc[1, 'electcon'] = np.nan
    numeric = sample.copy()
    numeric['electcon'] = numeric.electcon.map(lambda x: 1 if isinstance(x, str) and x.strip().lower() == 'yes' else 0 if isinstance(x, str) and x.strip().lower() == 'no' else np.nan)
    with tempfile.TemporaryDirectory(prefix='ec521_verify_') as tmp:
        source, destination = Path(tmp) / 'new_data.csv', Path(tmp) / 'predictions.csv'
        sample.to_csv(source, index=False)
        module.inference(SimpleNamespace(model_dir=model_dir, predict=source, prediction_output=destination))
        produced = pd.read_csv(destination)
        if task == 'classification':
            import torch
            meta = read_json(model_dir / 'metadata.json')
            processor = joblib.load(model_dir / 'preprocessor.joblib')
            model = module.FFN(len(meta['features']), meta['config']['dropout'])
            model.load_state_dict(torch.load(model_dir / 'weights.pt', weights_only=True, map_location='cpu'))
            expected = module.predict_scores(model, processor.transform(numeric).astype(np.float32))
            close(produced.poverty_score.to_numpy(), expected, 'classification feature-only inference with string/missing values')
            check(np.array_equal(produced.predicted_pov_exp, expected >= meta['threshold']), 'classification inference calibrated labels')
        else:
            fitted = module.load_model(model_dir)
            expected = module.predict_rows(numeric, numeric.index.to_numpy(), fitted)
            close(produced.predicted_rspcexp23.to_numpy(), expected, 'regression feature-only inference with string/missing values')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'SES2023.csv')
    parser.add_argument('--stage', choices=['splits', 'all'], default='all')
    parser.add_argument('--output-root', type=Path, default=HERE)
    parser.add_argument('--report', type=Path, default=HERE / 'checks/verification.json')
    parser.add_argument('--skip-originals', action='store_true',
                        help='For extracted submission ZIPs without the original repository files; dataset digest is still checked')
    args = parser.parse_args()
    try:
        original_count = 0 if args.skip_originals else verify_originals(args.data.resolve().parent)
        frame, groups = verify_data(args.data)
        manifest = verify_manifest(HERE / 'checks/split_manifest.json', frame, groups)
        if args.stage == 'all':
            import torch
            torch.set_num_threads(1)
            classification = verify_classification(args.output_root, frame, manifest)
            regression = verify_regression(args.output_root, frame, manifest)
            verify_inference(classification[0], classification[1], 'classification', frame)
            verify_inference(regression[0], regression[1], 'regression', frame)
        result = dict(status='passed', stage=args.stage, originals_checked=original_count,
                      rows=len(frame), checks=CHECKS)
    except Exception as exc:
        result = dict(status='failed', stage=args.stage, error=f'{type(exc).__name__}: {exc}',
                      checks=CHECKS)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
        print(result['error'], file=sys.stderr)
        raise
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2))
    print(f"PASS: {len(CHECKS)} checks; report {args.report}")


if __name__ == '__main__':
    main()
