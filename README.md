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

## Features + XGBoost (Person 2)

**Code**
- `src/features/rule_features.py`: rule-based fraud signals (`extract_features`, `extract_features_df`, `explain_features`). Feature table: `docs/feature_descriptions.md`.
- `src/models/xgboost_model.py`: trains XGBoost on TF-IDF + rule features, tunes on the validation set, evaluates on test, and exposes `predict(text)`.
- `tests/`: unit tests (`python -m pytest tests`).

**Train** (from the repo root; paths are examples)

```
pip install -r requirements_person2.txt
python -m src.models.xgboost_model --train data/train.csv --val data/val.csv --test data/test.csv --n-iter 25
```

Outputs: `models/xgboost/` (model, TF-IDF vectorizer, threshold) and `results/` (`xgboost_metrics.json`, `best_params.json`, `xgboost_search_log.csv`, PR curve, confusion matrix, feature importance, `xgboost_error_analysis.csv`).

**Protocol.** TF-IDF is fitted on train only. Hyper-parameters are chosen by validation PR-AUC. The decision threshold is picked on validation (max F1) and frozen. Test is scored once. Metrics are also reported on test rows whose text does not appear in train/val (`test_unseen_text`).

**Use in the app**

```python
from src.models.xgboost_model import predict
result = predict(posting_text)   # {"label": "fraud"|"real", "prob": 0.87, "reasons": [...]}
```

**Known limitations**
- The dataset masks emails, URLs and phones, so the email-domain and suspicious-URL features cannot be learned from it; they only fire on real pasted/uploaded postings.
- Scam wording (fees, WhatsApp, urgency) is rare in this dataset, so the model mostly relies on missing company profile/logo and on template phrases. It may rate an obvious scam posting as "real".
- For pasted text the structured fields (profile, logo, salary) are unknown, which weakens the prediction.
- `prob` is not calibrated (fraud is up-weighted during training); treat it as a score, not a true probability.
- Identical posting text appears in more than one split, which inflates scores; see `test_unseen_text`.

- ## App + Integration (Person 4)

**Code**
- `app/app.py`: Streamlit app. Accepts pasted text or an uploaded PDF/DOCX, shows the fraud probability, a verdict, and the rule-based signals. Optional BERT + LIME explanation (sidebar checkbox).
- `src/extract.py`: reads text from PDF (pdfplumber) and DOCX (python-docx).
- `test_cases/`: test postings (`.txt` and `full_postings/*.docx/pdf`) and ``results.md`.
- `docs/screenshots/`: app screenshots used in the report.

**Model files** (Google Drive, not in GitHub). Put them in `models/`:
- `xgboot_model.json`, `tfidf_vectorizer.pkl`, `meta.json` (XGBoost, TF-IDF vectorizer, feature order and threshold)


**Run** 

```
pip install -r requirements.txt
streamlit run app/app.py
```

**How the app decides.** The model input is the 33 rule features followed by the TF-IDF terms, in the order listed in `meta.json`; the app checks this at startup. A posting is flagged as fraudulent when the XGBoost probability reaches the saved threshold (0.462), or when 4 or more rule signals fire (hybrid verdict). The model score is always shown next to the verdict.

**App test results** (9 postings, see `test_cases/results.md`): 8 of 9 correct, 1 false negative, 0 false positives. This is a demonstration on a small sample, not a performance estimate.

**Known limitations**
- A polished scam with little aggressive wording can get a very low model score (3% to 6% in our tests). The hybrid rule catches it only when enough signals fire.
- Rule signals are keyword-based and do not understand negation, for example "we never ask for bank details" still triggers the personal-information signal.
- Pasted or uploaded text has no structured fields (company profile, logo), which weakens the model.
- LIME on BERT is slow on a CPU.
