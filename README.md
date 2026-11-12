# Fake Job Posting Detector

Machine learning project to detect fraudulent job postings (Kaggle: Real / Fake Job Posting Prediction).

## Folder structure
- `notebooks/`: one notebook per person
- `findings.md`: EDA findings
- `eda_charts.png`: EDA charts
- `data/` and `models/`: kept on Google Drive, not in GitHub

## Frozen data files (Google Drive, data folder)
- `fake_job_postings.csv`: raw data
- `cleaned_jobs.csv`: cleaned data (column `text` is the cleaned text)
- `train.csv`, `val.csv`, `test.csv`: stratified 70/15/15 split. Do NOT re-split.

## Baseline (TF-IDF + Logistic Regression, test set)
| Precision | Recall | F1 | PR-AUC |
|---|---|---|---|
| 0.748 | 0.859 | 0.80 | 0.904 |

Other models must be compared against these numbers on the same `test.csv`.

## Team
| Person | Role | Notebook |
|---|---|---|
| 1 | Data + Baseline | `01_eda`, `02_baseline` |
| 2 | Features + XGBoost | |
| 3 | BERT + Explainability | |
| 4 | App + Integration + Report | |

## Known limitations
- Dataset is old and skewed.
- One scam template (Aker Solutions copy) appears in 52 fake postings, so scores can be inflated.
- Split is by row, so near-copies may appear in both train and test.
