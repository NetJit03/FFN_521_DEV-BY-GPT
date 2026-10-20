# EC521 experiment protocol (fixed before new training)

Protocol version: 1.0; created 20 October 2026 (Asia/Bangkok).

## Data and historical exposure

Use the supplied SES2023.csv unchanged. Exclude both targets from inputs. Preserve the original 32-feature architecture and use the same predictors in every candidate. Parse Yes/No explicitly; never impute a target. Missing predictor values are imputed by a training-only median and all features standardized using training-only statistics. The observed electcon feature is constant with 30 missing values; correcting its processing is a protocol repair, not evidence of a performance gain.

Reconstruct both historical 20% test sets (random_state=1, with/without poverty stratification) and quarantine their union. Extend the quarantine to all identical predictor profiles, using all predictors except constant electcon as the grouping key. No historical test row enters fitting, threshold tuning, stopping, or model selection. Use the union only as a final, historically exposed diagnostic after locking the model selection. It is not a pristine independent test set. No external generalization claim is warranted.

## Cross-validation and seeds

Use five StratifiedGroupKFold partitions, shuffle=True, split seed 20261020, stratified on poverty for both tasks. Group identical predictor profiles to prevent these crossing partitions. This grouping is conservative and does not identify households. For each outer training pool, use five inner stratified group folds with seed 20261020 + outer fold: first fold stopping, second calibration, remaining three fitting. Classification uses separate fit/stop/calibration/evaluation subsets. Regression combines fit+calibration for fitting and retains stopping and outer evaluation separately. All methods within each task use identical partitions and data availability. Final fits use the same inner strategy with seed 20261020, final model seed 0.

Run each candidate with model seeds 0 and 42 on each of five folds. Report each of ten runs, plus means and standard deviations across five fold means after averaging the two model seeds. Fold scores are dependent and are NOT ten independent statistical replicates. Candidate selection uses these CV scores and hence winner CV scores have selection optimism; this is not fully nested configuration selection.

## Models and fixed candidate set

Preserve the baseline network: 32 inputs, hidden widths 64 and 32, ReLU, one output. Adam learning rate 0.001, maximum 50 epochs, classification batch size 64, regression batch size 32. Use deterministic CPU training and one compute thread. Record actual epochs, selected epoch, optimizer steps, runtime and all settings. Random oversampling increases steps per epoch; early stopping reduces actual training steps. Equal epochs do not mean equal compute.

Classification: unweighted BCE baseline; positive weights 3, 10, and the training-fold negative/positive ratio; random oversampling minority/majority ratio 0.10; unweighted BCE with dropout 0.20, weight decay 0.0001 and early stopping (patience 7). Record both fixed threshold 0.5 and independently calibrated F1 thresholds for each model. Select threshold from calibration PR breakpoints, breaking F1 ties toward the highest threshold. Monitor unweighted BCE on stopping data for early stopping. Report F1, precision, recall, average precision and accuracy from outer evaluation probabilities.

SMOTENC is considered but not included: binary assets coexist with mutually exclusive region dummies and compositional household age shares. Naive synthetic interpolation can violate joint constraints; rebuilding and validating a separate sampler would reduce comparability and is lower priority than reliable comparisons.

Regression: MSE baseline; MSE early stopping (patience 5); dropout 0.30 MSE; combined dropout 0.30 + early stopping + weight decay 0.0001; SmoothL1 (Huber, beta=1 in standardized target units) + early stopping; standardized log1p target MSE + early stopping. Fit target scalers only on training data. Early stopping is based on original-unit stopping RMSE. Inverse-transform predictions and report original-unit RMSE, MAE and R2 for every candidate. Log inverse transformation is not a bias-corrected conditional mean. Do not trim outliers or clip predictions after viewing evaluation results.

## Selection rule fixed before holdout evaluation

Classification incumbent: baseline BCE with independently calibrated threshold. A more complex candidate is eligible only if its mean paired F1 gain exceeds one standard error of the five fold gains (averaging model seeds) and it improves at least four of five folds. Choose the eligible candidate with the largest mean F1, or retain the incumbent. Always report the original baseline at threshold 0.5 and precision/recall trade-offs. This rule is a stability heuristic, not a hypothesis test or confidence guarantee.

Regression incumbent: fixed-epoch MSE baseline. Require at least 1% mean RMSE reduction, a gain greater than one standard error of paired fold gains, and improvement on at least four of five folds. Also reject a candidate if any rural/region group with n>=100 has more than 10% worse CV MAE than baseline. Select the eligible candidate with lowest mean RMSE or retain baseline. Examine expenditure-tail residuals and disclose systematic errors; group diagnostics cannot demonstrate fairness.

Write selection metadata before predicting on the historical holdout. Retrain only the baseline and selected model on the final development split with predeclared seed 0; do not change settings after holdout evaluation. Save checkpoints, preprocessors and final thresholds/transforms, together with row-level predictions and split manifests.

## Verification and reporting

Smoke-test real data separately from reportable results. Check finite losses/predictions, row/group isolation, saved model reloading, metric recomputation, and deterministic repeat of representative training. Keep original-file SHA-256 hashes unchanged. Report all evaluated candidates and limitations, including rare positives, heavy expenditure tails, prior test exposure, configuration-selection optimism and lack of survey weights/household IDs/external or future data. Prepare the English report (1-2 pages) and Thai recap from newly generated outputs only.
