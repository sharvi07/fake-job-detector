# Test results (Member 4: app testing)

| Posting | Expected | Model probability | Signals found | Verdict | Correct? | Notes |
|---|---|---|---|---|---|---|
| scam_A.txt | Scam | 88% | 4 | Fraudulent | Yes | Fee, WhatsApp/Telegram, urgency, personal info |
| scam_B.txt | Scam | 6% | 2 | Legitimate | No | Missed: free email + fee request, but low model score and only 2 signals |
| scam_C.txt | Scam | 73% | 3 | Fraudulent | Yes | WhatsApp, urgency, easy money |
| genuine_A.txt | Genuine | 1% | 0 | Legitimate | Yes | No signals |
| genuine_B.txt | Genuine | 7% | 0 | Legitimate | Yes | No signals |
| full_postings/scam_full_A.docx | Scam | 96% | 7 | Fraudulent | Yes | Model and rules agree |
| full_postings/scam_full_B.pdf | Scam | 3% | 4 | Fraudulent | Yes | Model alone missed it; caught by the hybrid rule (4 signals) |
| full_postings/genuine_full_A.docx | Genuine | 1% | 0 | Legitimate | Yes | No signals |
| full_postings/genuine_full_B.pdf | Genuine | 2% | 1 | Legitimate | Yes | One false signal: posting says it never asks for bank details |

Summary: 8 of 9 correct; 1 false negative (short scam_B), 0 false positives.

## Earlier example (from first run)
- Pasted scam with free email, fee request, WhatsApp, urgency and easy-money wording:
  model probability 49% (threshold 46.2%), 5 signals found.
- Pasted genuine posting: 0% probability, no signals.

## Observations and Limitations

**1. False Negatives**

One false negative was observed in `scam_B.txt`. The model assigned a fraud probability of 6%, with only two scam signals detected. Since the probability was below the classification threshold and the number of signals was below the hybrid minimum of four, the posting was classified as legitimate.

**2. False Positives**

No false-positive verdicts were observed among the four genuine postings tested. However, `genuine_full_B.pdf` produced one misleading rule-based signal because the posting mentioned bank details in a negative context.

**3. PDF/DOCX Upload Differences**

DOCX upload testing was completed successfully. The uploaded documents produced predictions and explanations through the application. PDF upload testing is still pending, so a comparison between PDF and DOCX extraction cannot yet be reported.

**4. Overall Observations**

The hybrid approach combines the XGBoost model probability with rule-based scam signals. It helped identify `scam_full_B.pdf`, which had a low model probability of 3% but four detected scam signals. However, the approach still missed `scam_B.txt`, demonstrating that the selected four-signal condition does not detect every fraudulent posting.

The results are based on a small test set and should not be considered a complete measure of model performance.
