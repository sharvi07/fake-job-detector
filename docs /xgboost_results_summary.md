# XGBoost results summary (Person 2)

## What was built
- **Features (33 rule-based + 3,000 TF-IDF):** rule-based signals from `src/features/rule_features.py` (missing company profile / requirements, logo and question flags, salary parsing, masked URL/email/phone counts, urgency, payment, messaging-app, "too good to be true" and personal-info phrases, ALL-CAPS ratios, text lengths) plus TF-IDF (1-2 word n-grams, 3,000 features) on Person 1's cleaned `text` column. Details: `docs/feature_descriptions.md`.
- **Model:** XGBoost (histogram trees), trained on the train split (12,319 postings, 599 fraud).
- **Protocol:** TF-IDF fitted on train only. Random search of 5 hyper-parameter configurations, scored by validation PR-AUC. Decision threshold (0.4624) chosen on validation by maximum F1, then frozen. Test set scored once. Fraud is up-weighted with `scale_pos_weight` (no resampling). Training took about 24 minutes (5 configurations + ablation) on Colab CPU.
- **Best configuration:** max_depth 8, learning_rate 0.05, subsample 0.85, colsample_bytree 0.5, min_child_weight 1, gamma 0.5, reg_lambda 1, full class weight; 478 trees. Validation PR-AUC across the 5 configurations ranged from 0.884 to 0.927.

## Results

| Split | Rows (fraud) | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Validation | 2,640 (129) | 0.929 | 0.814 | 0.868 | 0.927 | 0.994 |
| **Test, all rows** | 2,640 (128) | 0.865 | 0.750 | **0.803** | 0.889 | 0.988 |
| **Test, text not seen in train/val** | 2,371 (100) | 0.819 | 0.680 | **0.743** | 0.833 | 0.983 |

Test confusion matrix (threshold 0.4624): 96 fraud caught, 32 missed, 15 real postings flagged, 2,497 real postings passed. At the default threshold 0.5 the test F1 is 0.810 (precision 0.881, recall 0.750).

Validation scores are optimistic because validation was used for tuning, early stopping and the threshold. The test row is the one to quote, and the unseen-text row is the honest one (see below).

### Duplicate text across splits
269 of the 2,640 test rows (10%) have `text_raw` identical to a posting in train or val; 28 of them are fraud (22% of test fraud). The model classified all 269 correctly, so every one of its test errors is in the unseen rows. Removing the duplicates lowers test F1 from 0.80 to 0.74 and PR-AUC from 0.89 to 0.83. The baseline is affected in the same way; compare models on both rows.

### Ablation (same hyper-parameters, test set, all rows)

| Features | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Rule features only (33) | 0.752 | 0.641 | 0.692 | 0.749 |
| TF-IDF only (3,000) | 0.870 | 0.680 | 0.763 | 0.851 |
| **Both** | 0.865 | 0.750 | 0.803 | 0.889 |

The rule features add about 4 points of PR-AUC and F1 on top of TF-IDF. Most of their strength comes from metadata (missing company profile, missing logo), not from scam wording (see limitations).

## Error analysis (test set)

**Missed fraud (32 of 128).**
- They look like ordinary professional postings: e.g. "Software Web Applications Engineer", "SQL BI Tester", "Physician Assistant", "Process Validation Project Manager", "Engineering Senior Supervisor". Their median description is longer (about 1,080 characters vs 815 for caught fraud) and carries none of the usual scam wording.
- 75% of them have no company profile, so the strongest feature does not separate them from real postings; the model scored them 0.16 on average. 41% still scored above 0.2, so some are near-misses that a lower threshold would catch at the cost of more false alarms.
- One of them is titled "Urgent position for SQL BI Tester" yet scored 0.004, so urgency wording alone does not move the score.

**False alarms (15).** All 15 have no company profile and no logo. They are generic office jobs (receptionist, payroll assistant, staff accountant, office administrator), plus restaurant postings. The model has learned "no profile + no logo + generic admin job = fraud", which real small employers also match.

**Top features** (by gain): `no_company_profile` is first, `has_company_logo` is eighth, and most others are generic corporate phrases ("systems and", "on our", "procedures and", "maximize", "staffing"), which look like template text from repeated fraud postings rather than scam language.

## Limitations
- The dataset masks emails, URLs and phone numbers, so the free-email, email/URL mismatch and suspicious-URL features are always 0 here. They only work on pasted or uploaded postings in the app, and could not be evaluated.
- Scam wording (fees, WhatsApp/Telegram, personal-data requests) is almost absent from the labelled fraud postings (0 of 128 in test for payment requests and messaging apps), so the model does not respond to it. A hand-written scam posting with a free email, a fee request, WhatsApp and urgency language scored 0.33 in an early trial, below the threshold. Test the final model on real scam postings collected by the team.
- For pasted text the structured fields (company profile, logo, salary) are unknown, which removes the strongest feature.
- Scores are not calibrated probabilities (the fraud class is up-weighted).
- Part of the performance comes from repeated posting templates; the honest estimate is the unseen-text row.
- Only 5 hyper-parameter configurations were tried; a longer search may add a little.
- Person 1's EDA suggested two more features, country (Australia stands out) and an all-fields-missing flag. They were not included in the trained model due to time and are future work. A quick check on a small rule-feature-only model suggested they help, but XGBoost was not retrained with them.

## Comparison table (fill in after Person 1 and Person 3 report)

| Model | Precision | Recall | F1 | PR-AUC | F1 on unseen-text test |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression (Person 1) | 0.748 | 0.859 | 0.800 | 0.904 | (to compute) |
| **XGBoost, TF-IDF + rules (Person 2)** | 0.865 | 0.750 | 0.803 | 0.889 | 0.743 |
| BERT (Person 3) | | | | | |

### Reading the comparison
XGBoost and the baseline are essentially tied. F1 is 0.803 vs 0.800, and PR-AUC, which does not depend on a threshold, is slightly in the baseline's favour (0.904 vs 0.889). The two models sit at different operating points: XGBoost is more precise (0.865 vs 0.748), while the baseline catches more fraud (recall 0.859 vs 0.750). The test set has only 128 fraud postings, so a single posting moves recall by about 0.8 points and gaps of this size are probably within noise. Adding rule features and a tree model therefore did not beat a strong linear baseline on this data, which matches the warning in the project plan. The unseen-text column is left open for the baseline: it should be computed on the same rows (test rows whose `text_raw` is not in train or val) before the models are compared on it.
