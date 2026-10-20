# EC521: Feedforward Neural Networks

This submission develops and evaluates poverty classification (`pov_exp`) and expenditure regression (`rspcexp23`) using the supplied SES2023.csv. Results are produced by actual PyTorch training. The original repository files and existing local edits are preserved; nothing is committed or pushed to `main`.

## Start here

- `reports/EC521_Report.pdf`: final English report (two pages).
- `reports/EC521_Report.md`: editable English report text.
- `reports/Recap_TH.md`: Thai explanation of changes, results and limitations.
- `classification.py` and `regression.py`: runnable training, evaluation and inference scripts.
- `common.py`: strict input parsing and shared partition protocol.
- `results/`: actual experiment tables, predictions, settings and selection decisions.
- `figures/`: evaluation and training plots.
- `models/`: final fitted networks and preprocessing objects.
- `checks/`: environment, original-file hashes and independent validation evidence.
- `EXPERIMENT_PROTOCOL.md`: decisions fixed before training and final holdout evaluation.
- `audit_independent.md`: audit of all original scripts and old result files.

## Verified environment

Python 3.14.4 on macOS arm64; deterministic CPU training, one PyTorch compute thread per process. The supplied project `.venv` successfully performed a PyTorch backward/optimizer step, a scikit-learn transform and imbalanced-learn RandomOverSampler resampling. Versions are pinned in `requirements.txt`; complete installed-package versions are in `checks/environment_freeze.txt`.

From the original repository directory, reuse the already verified environment:

```sh
.venv/bin/python ec521_submission/classification.py --data SES2023.csv
.venv/bin/python ec521_submission/regression.py --data SES2023.csv
```

For a fresh environment with compatible package distributions available:

```sh
python3 -m venv .venv_ec521
.venv_ec521/bin/python -m pip install -r ec521_submission/requirements.txt
.venv_ec521/bin/python ec521_submission/classification.py --data SES2023.csv --output-root rerun_classification
.venv_ec521/bin/python ec521_submission/regression.py --data SES2023.csv --output-root rerun_regression
```

The included results were generated in the verified existing environment; installation in a separate new environment was not independently tested. Use a new `--output-root` when reproducing experiments to preserve the submitted evidence. Package versions, hardware and operating system can affect numerical reproducibility even with fixed seeds. No GPU is required.

Default experiment settings: five folds, model seeds 0 and 42, 50 maximum epochs. Scripts save figures without opening interactive windows. `--smoke` runs a small actual-data execution check and writes to separate smoke outputs; smoke scores must never be reported as experiment results. `--help` documents all command options.

## What the comparison means

The baseline preserves the original 32→64→32→1 network, ReLU, Adam (learning rate 0.001), 50 epochs and original batch sizes (classification 64; regression 32). Preprocessing and evaluation are corrected for all candidates. These are **original-architecture baselines under a corrected common protocol**, not exact reproductions of older CSV numbers: training partitions and shuffle sequences differ.

The old stratified and unstratified test splits both used random_state=1 and were already evaluated by the original workflows. Their union (17,867 rows) is quarantined from all new model development. Keeping identical predictor profiles on one side extends it to 18,878 rows. The remaining development set has 30,786 rows and 472 positive poverty cases. The final diagnostic holdout has 297 positives. It is explicitly **historically exposed**, so its scores are not evidence from a pristine external or future sample.

Five stratified group folds keep identical observed predictor patterns together. These profiles do not identify real households or sampling clusters. In each outer fold, classification uses about 60% of the outer training pool for fitting, 20% for early stopping and 20% for threshold calibration. Regression merges fitting and calibration subsets (80%) and uses the remaining 20% for stopping. Every candidate within a task uses the same available fitting rows, including fixed-epoch baselines.

Imputers, feature scalers and target scalers are fitted only to training rows. Oversampling applies only to the fitted classification subset, after fitting preprocessing on the original training rows. Thresholds are chosen on calibration data; outer-fold labels never choose thresholds or stopping epochs. Threshold tuning changes precision and recall but does not change average precision when the underlying scores are unchanged.

Actual optimizer-step counts are reported. Oversampling increases training rows and steps; early stopping reduces epochs. Thus comparisons have equal maximum epochs, not equal computational budgets. Generalization gaps use evaluation-mode predictions on the original, unresampled fitting data.

Model choice uses a prespecified paired-fold stability rule, described in `EXPERIMENT_PROTOCOL.md`, with regression rural/region residual safeguards. It is a practical rule, not a formal significance test. The summary mean and standard deviation use five fold means after averaging seeds; ten individual runs are also retained. These are not ten independent observations. Candidate configurations are selected using CV, so the winning CV score remains subject to selection optimism; configuration selection is not fully nested. Final model seed 0 is fixed in advance, and selection is saved before historical holdout prediction. No settings are changed in response to final holdout results.

## Scope and limitations

The supplied CSV has 49,664 rows, 32 predictors and two targets. There are 769 positive poverty cases (1.5484%). `electcon` contains 49,634 `Yes` values and 30 missing entries, making its observed value constant. Correcting global imputation and the discarded Yes/No conversion repairs unsafe code; it does not by itself demonstrate a material performance gain on this particular missingness pattern.

SMOTENC was considered and deferred because the inputs contain mutually exclusive region indicators and compositional household age shares. A valid synthetic sampler would require additional joint-constraint validation; random oversampling provides a lower-risk comparison under the same feature representation.

Expenditure is strongly right-skewed (median 6,758; maximum 360,795 in supplied units). Every regression metric is computed after inverse transformation to the original target units. Tail and subgroup diagnostics remain essential even when aggregate RMSE or R² improves. No outlier rows are removed and no evaluation-driven clipping is applied. A naive inverse log transformation estimates a different conditional quantity and is not a bias-corrected conditional mean.

There is no supplied variable dictionary establishing currency/time units, household identifier, survey weight, sampling cluster, external cohort or future-year sample. Interpret expenditure units as supplied `rspcexp23` units, and results as internal survey-row prediction rather than national population estimates, causal findings, or validated future performance.

## Method references

- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [imbalanced-learn: SMOTENC API and mixed categorical/continuous inputs](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html)
- [PyTorch: reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

These links support implementation principles. All numerical findings come from the supplied dataset and the included execution outputs.

## Use the selected models with new data

`new_households.csv` must contain the same 32 feature columns. Targets are optional and ignored for inference; column order is handled by the saved feature list. Missing numeric feature values are filled using the saved training medians. Unrecognized text or infinite values raise an error.

```sh
.venv/bin/python ec521_submission/classification.py --predict new_households.csv --model-dir ec521_submission/models/classification/selected --prediction-output classification_predictions.csv
.venv/bin/python ec521_submission/regression.py --predict new_households.csv --model-dir ec521_submission/models/regression/huber --prediction-output regression_predictions.csv
```

The selected classifier uses cutoff 0.05983597785234451. Its sigmoid output is a model score, not an independently calibrated poverty-risk estimate. The selected regression model is Huber with early stopping; it predicts original `rspcexp23` units. The saved models are the fixed development-trained models used for the final diagnostic comparison, not refits on the historical holdout.

## Verify the evidence

```sh
.venv/bin/python ec521_submission/verify_submission.py
.venv/bin/python ec521_submission/check_reproducibility.py
```

The first command independently checks the source-data hash, original file hashes, splits, fit-only preprocessing, thresholds, model-selection rules, saved predictions, metric tables and model reloads. The second repeats fold 1 / seed 0 for each baseline and selected candidate using the original training budgets. The delivered run passed 3,399 checks and four repeat fits with zero prediction differences in the same environment. These checks do not establish reproducibility across other software/hardware versions.

The ZIP contains `SES2023.csv` and `ec521_submission/`. It omits the original reference scripts, virtual environment and scratch smoke outputs. When working from an extracted ZIP, use `verify_submission.py --skip-originals`: the original-file preservation check cannot be repeated without those original files, but the dataset hash and all model/result checks still run. The original repository preservation result remains recorded in the included verification evidence. A fresh Python environment is needed as described above.

Optional report regeneration after completed experiment outputs:

```sh
.venv/bin/python -m pip install -r ec521_submission/requirements-report.txt
.venv/bin/python ec521_submission/build_report.py
```

The supplied English PDF was rendered and visually checked as two A4 pages. All eight evaluation PNGs were visually reviewed. `reports/Recap_TH.md` includes the important lowest-expenditure-quintile deterioration in the historical holdout, as well as the classification precision cost and high-expenditure tail errors. No result should be described as improvement for every group or as an independent external validation.
