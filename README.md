# Tune & Ensemble a Classifier

Day 4 — Model Optimization. Primary dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
The supplied `sns.load_dataset('titanic')` example is covered by a separate Titanic companion notebook.

## Required deliverables

1. Executed notebook with Decision Tree baseline five-fold CV, GridSearchCV and RandomizedSearchCV best parameters and scores.
2. A trained Random Forest ensemble evaluated on exactly the same train/test split and training folds.
3. One four-column results table per dataset: Model | CV Accuracy | Test Accuracy | Best Params.

## Telco results

| Model | CV Accuracy | Test Accuracy | Best Params |
|---|---:|---:|---|
| Decision Tree baseline | 0.7247 | 0.7218 | `Default parameters; random_state=42` |
| Decision Tree GridSearchCV | 0.7938 | 0.7984 | `{"model__criterion": "entropy", "model__max_depth": 5, "model__min_samples_leaf": 5, "model__min_samples_split": 2}` |
| Decision Tree RandomizedSearchCV | 0.7938 | 0.7984 | `{"model__criterion": "entropy", "model__max_depth": 5, "model__min_samples_leaf": 5, "model__min_samples_split": 5}` |
| Random Forest ensemble | 0.7994 | 0.7963 | `Fixed: {"max_depth": null, "max_features": "sqrt", "min_samples_leaf": 2, "n_estimators": 200, "n_jobs": 1, "random_state": 42}` |

Random Forest is selected by training CV (79.94%); its hold-out accuracy is 79.63%. The tuned trees score 79.84% on the test set. Search CV scores also select parameters and may be optimistic; small differences on this single split do not establish superiority. Accuracy is the requested metric; class-level precision and recall are also displayed because churn is imbalanced.

## Files and running

- `Telco_Churn_Tune_and_Ensemble.ipynb`: primary executed notebook, with tables, chart, confusion matrix and checks.
- `Titanic_Tune_and_Ensemble.ipynb`: companion for the supplied Seaborn example.
- Matching `.py` files contain the complete runnable notebook code.
- `Telco_Results.csv` and `Titanic_Results.csv`: required results tables.
- `Model_Optimization_Submission.zip`: both notebooks, runnable scripts, tables, figures, search results, split manifests and provenance.

Install dependencies with `python -m pip install -r requirements.txt`, then open either notebook in Jupyter or run its `.py` file. Telco automatically locates an attached Kaggle dataset, a local CSV, or downloads the original public Kaggle archive. Titanic uses Seaborn's dataset loader. Internet is needed if the data is not already local/cached. Results are recreated under `results/telco` or `results/titanic`.

## Method

80/20 stratified split, seed 42; five identical shuffled stratified folds on training data; fold-local median/mode imputation and one-hot encoding inside a Pipeline. GridSearchCV tests 48 combinations; RandomizedSearchCV samples 24 of 512 combinations, seed 42. Both optimize accuracy and refit on training data. Random Forest uses 200 trees and a minimum leaf size of 2, fixed before evaluation. Customer ID and the target are excluded from features. Titanic excludes `alive`, which directly reveals survival.

The saved notebooks execute top-to-bottom. Assertions check four result rows, five folds, train/test disjointness, search candidate counts and independently recount each test accuracy from saved predictions. Source hashes and library versions are recorded in provenance. No source dataset is redistributed in this repository.
