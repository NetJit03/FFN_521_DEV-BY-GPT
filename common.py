"""Shared, deterministic data and split contract for EC521 experiments.

No preprocessing is fitted here. Row IDs are zero-based CSV data-row positions.
The historical test union is quarantined before any model selection.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

FEATURES = [
    'reg_1', 'reg_2', 'reg_3', 'reg_4', 'reg_5', 'rural', 'head_age',
    'head_female', 'head_schooling', 'shr_no05', 'shr_no615', 'shr_no1660',
    'shr_no60plus', 'own_house', 'rent_lease', 'solidwall', 'electcon',
    'radio', 'tv', 'vcd', 'phone', 'cphone', 'internet', 'fridge', 'washmach',
    'microwave', 'aircondition', 'elec_fan', 'bcycle', 'mcycle', 'car', 'pickup',
]
SPLIT_SEED = 20261020


def load_data(path):
    """Parse schema without imputing targets or estimating feature statistics."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing = set(FEATURES + ['pov_exp', 'rspcexp23']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    for col in frame:
        values = frame[col].map(
            lambda v: v.strip() if isinstance(v, str) else v)
        values = values.map(lambda v: {'yes': 1, 'no': 0}.get(v.lower(), v)
                            if isinstance(v, str) else v)
        numeric = pd.to_numeric(values, errors='coerce')
        invalid = values.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f'Unrecognized nonnumeric values in {col}')
        frame[col] = numeric
    if frame[['pov_exp', 'rspcexp23']].isna().any().any():
        raise ValueError('Missing targets: do not impute target values.')
    if not set(frame.pov_exp.unique()).issubset({0, 1}):
        raise ValueError('pov_exp must contain only 0 and 1.')
    if (frame.rspcexp23 <= 0).any():
        raise ValueError('Expected positive expenditure for log1p experiment.')
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError('Infinite values are not accepted.')
    if not (frame[[f'reg_{i}' for i in range(1, 6)]].sum(axis=1) == 1).all():
        raise ValueError('Region dummy columns must sum to one.')
    frame.attrs['data_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame


def feature_groups(frame):
    """Conservatively group identical observed predictor profiles.

    electcon is excluded only from this grouping key: its observed value is
    constant and its missingness should not let otherwise identical profiles
    cross partitions. It remains an input feature in every fitted model.
    Profiles are NOT claimed to identify households or sampling clusters.
    """
    key = pd.MultiIndex.from_frame(
        frame[[c for c in FEATURES if c != 'electcon']])
    return pd.factorize(key, sort=False)[0]


def _assert_disjoint(parts, groups):
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.intersect1d(parts[a], parts[b]).size:
                raise AssertionError(f'Row overlap: {a}/{b}')
            if np.intersect1d(groups[parts[a]], groups[parts[b]]).size:
                raise AssertionError(f'Predictor-profile overlap: {a}/{b}')


def final_partitions(df, dev, split_seed=SPLIT_SEED):
    """Return common fit/stop/calibration partitions (~60/20/20%)."""
    dev = np.asarray(dev, dtype=int)
    groups = feature_groups(df)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=split_seed)
    inner = list(splitter.split(dev, df.pov_exp.iloc[dev], groups[dev]))
    stop = dev[inner[0][1]]
    calibration = dev[inner[1][1]]
    fit = np.setdiff1d(dev, np.union1d(stop, calibration))
    parts = dict(fit=fit, stop=stop, calibration=calibration)
    _assert_disjoint(parts, groups)
    if not np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.sort(dev)):
        raise AssertionError('Incomplete inner partition')
    for name, ids in parts.items():
        if df.pov_exp.iloc[ids].nunique() != 2:
            raise ValueError(
                f'Both classes required in {name}; increase sample size.')
    return parts


def make_partitions(df, outdir, n_splits=5, split_seed=SPLIT_SEED):
    """Quarantine old test rows and group duplicates, then build stratified CV.

    Historical splits reproduce the original code with both stratification
    choices. The union is extended to include identical predictor profiles.
    This is a reused historical holdout, never described as an unseen test.
    """
    ids = np.arange(len(df))
    _, old_classification = train_test_split(
        ids, test_size=.2, random_state=1, stratify=df.pov_exp)
    _, old_regression = train_test_split(ids, test_size=.2, random_state=1)
    historical = np.union1d(old_classification, old_regression)
    groups = feature_groups(df)
    holdout = ids[np.isin(groups, groups[historical])]
    dev = np.setdiff1d(ids, holdout)
    _assert_disjoint(dict(dev=dev, holdout=holdout), groups)
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                 random_state=split_seed)
    folds = []
    for fold, (train, evaluation) in enumerate(
            outer.split(dev, df.pov_exp.iloc[dev], groups[dev]), 1):
        parts = final_partitions(df, dev[train], split_seed + fold)
        parts['evaluation'] = dev[evaluation]
        _assert_disjoint(parts, groups)
        folds.append(parts)
    result = dict(dev=dev, holdout=holdout, folds=folds)
    serial = dict(
        data_sha256=df.attrs.get('data_sha256'), split_seed=split_seed,
        n_splits=n_splits, historical_union_before_group_extension=len(
            historical),
        development_rows=len(dev), historical_holdout_rows=len(holdout),
        development_positives=int(df.pov_exp.iloc[dev].sum()),
        historical_holdout_positives=int(df.pov_exp.iloc[holdout].sum()),
        holdout_label='historically_exposed_holdout_diagnostic',
        group_definition='identical predictors except constant electcon; not household IDs',
        dev=dev.tolist(), holdout=holdout.tolist(),
        folds=[{k: v.tolist() for k, v in p.items()} for p in folds],
    )
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'split_manifest.json').write_text(json.dumps(serial, indent=2))
    return result
