"""Repeat four fixed representative full-budget fits, without selecting models.

Uses fold 1 / seed 0 for each baseline and the already selected candidate.
Expected values come from saved full experiment predictions. No model choice
or hyperparameter is changed by this check.
"""
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import torch

import classification as clf
import regression as reg
from common import load_data


def main():
    root = Path(__file__).resolve().parent
    df = load_data(root.parent / 'SES2023.csv')
    manifest = json.loads((root / 'checks/split_manifest.json').read_text())
    parts = {k: np.asarray(v, int) for k, v in manifest['folds'][0].items()}
    torch.set_num_threads(1)
    records = []
    chosen_c = json.loads(
        (root / 'results/classification/locked_selection.json').read_text())['method']
    chosen_r = json.loads(
        (root / 'results/regression/selection.json').read_text())['selected_variant']
    with tempfile.TemporaryDirectory(prefix='ec521_repeat_') as tmp:
        paths = {
            k: Path(tmp) / k for k in ['results', 'models', 'figures', 'logs']}
        for p in paths.values():
            p.mkdir()
        for method in dict.fromkeys(['baseline', chosen_c]):
            name = f'fold1_seed0_{method}'
            metadata = json.loads(
                (root / 'models/classification' / name / 'metadata.json').read_text())
            info, _, scores, _ = clf.fit_run(
                df, parts, method, 0, metadata['max_epochs'], paths, name)
            original = np.load(
                root / 'results/classification' / f'{name}_predictions.npz')
            delta = float(
                np.max(np.abs(scores['evaluation'] - original['evaluation_score'])))
            if delta != 0 or info['threshold'] != metadata['threshold']:
                raise AssertionError(
                    f'Classification repeat mismatch: {method}, {delta}')
            records.append(dict(task='classification', method=method, fold=1, seed=0,
                                exact_prediction_match=True, maximum_absolute_difference=delta,
                                threshold_match=True, epochs=info['epochs_executed']))
        fit = np.sort(np.r_[parts['fit'], parts['calibration']])
        for method in dict.fromkeys(['original', chosen_r]):
            variant = next(v for v in reg.VARIANTS if v['name'] == method)
            name = f'fold1_seed0_{method}'
            metadata = json.loads(
                (root / 'logs/regression' / f'{name}.json').read_text())['configuration']
            fitted = reg.fit_model(
                df, fit, parts['stop'], variant, 0, metadata['max_epochs'])
            predicted = reg.predict_rows(df, parts['evaluation'], fitted)
            original = pd.read_csv(
                root / 'results/regression' / f'{name}_predictions.csv')
            delta = float(
                np.max(np.abs(predicted - original.predicted.to_numpy())))
            # CSV decimal round-tripping can introduce <=1e-8 in original units.
            if delta > 1e-8 or fitted[3]['selected_epoch'] != metadata['selected_epoch']:
                raise AssertionError(
                    f'Regression repeat mismatch: {method}, {delta}')
            records.append(dict(task='regression', method=method, fold=1, seed=0,
                                maximum_absolute_difference=delta, tolerance=1e-8,
                                selected_epoch_match=True, epochs=fitted[3]['epochs_run']))
    output = root / 'checks/reproducibility.json'
    output.write_text(json.dumps(dict(
        status='passed', scope='Four representative full-budget fits; same environment', runs=records), indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
