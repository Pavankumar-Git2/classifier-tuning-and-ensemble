import matplotlib
matplotlib.use("Agg")

import io, json, hashlib, platform, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
DATASET = 'telco'
SEED = 42
result_dir = Path('results') / DATASET
result_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
print({'Python': platform.python_version(), 'scikit-learn': sklearn.__version__, 'pandas': pd.__version__, 'seed': SEED})

filename = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
candidates = [Path(filename), Path('data') / filename, Path('../../work/data') / filename]
if Path('/kaggle/input').exists():
    candidates.extend(Path('/kaggle/input').rglob(filename))
local = next((p for p in candidates if p.is_file()), None)
if local is not None:
    raw_bytes = local.read_bytes()
else:
    url = 'https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn'
    with urllib.request.urlopen(url, timeout=90) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    raw_bytes = archive.read(next(n for n in archive.namelist() if n.endswith(filename)))
df = pd.read_csv(io.BytesIO(raw_bytes))
source_hash = hashlib.sha256(raw_bytes).hexdigest()
display(df.head())
assert df.customerID.is_unique, 'Expected one observation per customer'
assert df.Churn.isin(['Yes', 'No']).all()
X = df.drop(columns=['customerID', 'Churn']).copy()
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
y = df.Churn.map({'No': 0, 'Yes': 1})

display(pd.DataFrame({'rows': [len(df)], 'features_used': [X.shape[1]], 'exact_duplicate_source_rows': [int(df.duplicated().sum())]}))
display(y.value_counts().rename_axis('Target class').to_frame('Count'))
display(X.isna().sum().loc[lambda s: s > 0].to_frame('Missing values'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
assert set(X_train.index).isdisjoint(X_test.index)
assert len(X_train) + len(X_test) == len(X)
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train, y_train))
print(f'Train: {len(X_train):,}; test: {len(X_test):,}; CV folds: {len(cv)}')
numeric = X.select_dtypes(include='number').columns.tolist()
categorical = X.columns.difference(numeric).tolist()
preprocessing = ColumnTransformer([
    ('numeric', SimpleImputer(strategy='median'), numeric),
    ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                              ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical)
])
def make_pipeline(model):
    return Pipeline([('preprocess', clone(preprocessing)), ('model', model)])
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print(f'Majority-class test accuracy (context only): {dummy.score(X_test, y_test):.4f}')

baseline = make_pipeline(DecisionTreeClassifier(random_state=SEED))
baseline_folds = cross_val_score(baseline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
baseline.fit(X_train, y_train)
print('Baseline fold accuracies:', np.round(baseline_folds, 4))
print(f'Baseline CV accuracy: {baseline_folds.mean():.4f} ± {baseline_folds.std():.4f} (fold SD)')

parameter_grid = {
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': [3, 5, 8, None],
    'model__min_samples_leaf': [1, 5, 15],
    'model__min_samples_split': [2, 10]
}
grid = GridSearchCV(make_pipeline(DecisionTreeClassifier(random_state=SEED)), parameter_grid,
                    cv=cv, scoring='accuracy', n_jobs=1, refit=True, error_score='raise')
grid.fit(X_train, y_train)
print('GridSearchCV best params:', grid.best_params_)
print(f'GridSearchCV best CV accuracy: {grid.best_score_:.4f}')

parameter_distributions = {
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': [2, 3, 4, 5, 6, 8, 12, None],
    'model__min_samples_leaf': [1, 2, 5, 10, 15, 25, 40, 60],
    'model__min_samples_split': [2, 5, 10, 20]
}
randomized = RandomizedSearchCV(make_pipeline(DecisionTreeClassifier(random_state=SEED)),
    parameter_distributions, n_iter=24, random_state=SEED, cv=cv, scoring='accuracy',
    n_jobs=1, refit=True, error_score='raise')
randomized.fit(X_train, y_train)
print('RandomizedSearchCV best params:', randomized.best_params_)
print(f'RandomizedSearchCV best CV accuracy: {randomized.best_score_:.4f}')

forest_params = {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': SEED, 'n_jobs': 1}
ensemble = make_pipeline(RandomForestClassifier(**forest_params))
ensemble_folds = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
ensemble.fit(X_train, y_train)
print('Random Forest fold accuracies:', np.round(ensemble_folds, 4))
print(f'Random Forest CV accuracy: {ensemble_folds.mean():.4f} ± {ensemble_folds.std():.4f} (fold SD)')

models = {'Decision Tree baseline': baseline, 'Decision Tree GridSearchCV': grid.best_estimator_,
          'Decision Tree RandomizedSearchCV': randomized.best_estimator_, 'Random Forest ensemble': ensemble}
cv_means = [baseline_folds.mean(), grid.best_score_, randomized.best_score_, ensemble_folds.mean()]
params = ['Default parameters; random_state=42', json.dumps(grid.best_params_, sort_keys=True),
          json.dumps(randomized.best_params_, sort_keys=True), 'Fixed: ' + json.dumps(forest_params, sort_keys=True)]
predictions = {name: model.predict(X_test) for name, model in models.items()}
results = pd.DataFrame({'Model': list(models), 'CV Accuracy': cv_means,
    'Test Accuracy': [accuracy_score(y_test, predictions[name]) for name in models], 'Best Params': params})
display(results.style.format({'CV Accuracy': '{:.4f}', 'Test Accuracy': '{:.4f}'}).set_properties(**{'text-align': 'left', 'white-space': 'normal'}))
results.to_csv(result_dir / 'model_comparison.csv', index=False)
cv_selected = results.iloc[int(np.argmax(cv_means))]['Model']
print('Preferred model based on training CV:', cv_selected)
print(classification_report(y_test, predictions[cv_selected], digits=4, zero_division=0))

labels = ['Baseline DT', 'GridSearch DT', 'Randomized DT', 'Random Forest']
fig, ax = plt.subplots(figsize=(9, 4.6))
pos = np.arange(len(labels)); width = .35
bars1 = ax.bar(pos - width/2, results['CV Accuracy'], width, label='5-fold training CV', color='#3569a8')
bars2 = ax.bar(pos + width/2, results['Test Accuracy'], width, label='20% hold-out test', color='#dba64b', hatch='//')
ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)
ax.set(xticks=pos, xticklabels=labels, ylim=(0, 1.12), ylabel='Accuracy', title=f'{DATASET.title()}: model accuracy on identical folds and split')
ax.legend(loc='upper center', ncol=2); fig.tight_layout()
fig.savefig(result_dir / 'accuracy_comparison.png', bbox_inches='tight'); plt.show()
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, predictions[cv_selected], ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'CV-selected model: {cv_selected}', fontsize=10)
fig.tight_layout(); fig.savefig(result_dir / 'confusion_matrix.png', bbox_inches='tight'); plt.show()

selected_row = results.loc[results.Model.eq(cv_selected)].iloc[0]
gain = (selected_row['CV Accuracy'] - results.iloc[0]['CV Accuracy']) * 100
conclusion = (f"**{cv_selected}** has the highest training CV accuracy ({selected_row['CV Accuracy']:.4f}), "
    f"a {gain:.2f} percentage-point change from the baseline, with hold-out accuracy {selected_row['Test Accuracy']:.4f}. "
    "Tree regularization controls depth and leaf sizes, while Random Forest averages many trees to reduce variance. "
    "These are results on one split; search CV can be optimistic, and nested CV or repeated validation would be needed for a stronger comparison.")
display(Markdown(conclusion))
(result_dir / 'CONCLUSION.md').write_text(conclusion, encoding='utf-8')

grid_results = pd.DataFrame(grid.cv_results_)
random_results = pd.DataFrame(randomized.cv_results_)
grid_results.to_csv(result_dir / 'grid_search_results.csv', index=False)
random_results.to_csv(result_dir / 'randomized_search_results.csv', index=False)
fold_scores = pd.DataFrame({'Fold': range(1, 6), 'Baseline': baseline_folds,
    'GridSearchCV best': [grid_results.loc[grid.best_index_, f'split{i}_test_score'] for i in range(5)],
    'RandomizedSearchCV best': [random_results.loc[randomized.best_index_, f'split{i}_test_score'] for i in range(5)],
    'Random Forest': ensemble_folds})
fold_scores.to_csv(result_dir / 'cv_fold_scores.csv', index=False)
test_predictions = pd.DataFrame({'source_row': X_test.index, 'actual': y_test.to_numpy(), **predictions})
test_predictions.to_csv(result_dir / 'test_predictions.csv', index=False)
split_manifest = pd.DataFrame({'source_row': X.index, 'split': np.where(X.index.isin(X_test.index), 'test', 'train')})
split_manifest.to_csv(result_dir / 'split_manifest.csv', index=False)
provenance = {'dataset': DATASET, 'sha256': source_hash, 'rows': len(df), 'train_rows': len(X_train),
    'test_rows': len(X_test), 'seed': SEED, 'folds': 5, 'scoring': 'accuracy',
    'grid_candidates': len(grid_results), 'random_candidates': len(random_results),
    'selected_by_training_cv': cv_selected, 'sklearn': sklearn.__version__, 'pandas': pd.__version__}
(result_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
assert results.shape == (4, 4)
assert len(grid_results) == 48 and len(random_results) == 24
assert results[['CV Accuracy', 'Test Accuracy']].apply(lambda col: col.between(0, 1).all()).all()
for name in models:
    assert np.isclose(results.loc[results.Model.eq(name), 'Test Accuracy'].iloc[0],
                      (test_predictions[name] == test_predictions['actual']).mean())
print('Checks passed: four models, shared split, five folds, 48 grid / 24 random candidates, independently recounted test accuracies.')
display(fold_scores.round(4))