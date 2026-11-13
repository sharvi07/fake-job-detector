# Rule-based features (Person 2)

Produced by `src/features/rule_features.py` (`extract_features`, `extract_features_df`).
"Unknown" means the value is NaN when the input is plain pasted text with no
structured fields. XGBoost handles NaN natively.

The EMSCAD / Kaggle data masks emails, URLs and phone numbers as hashed tokens
(`#EMAIL_...#`, `#URL_...#`, `#PHONE_...#`). Features marked **real-address** need
real addresses, so they are all 0 on the dataset and can only fire on pasted or
uploaded postings in the app.

| Feature | Meaning | Why it signals fraud |
|---|---|---|
| `no_company_profile` | Company profile field is empty (unknown for pasted text) | Scam postings often hide who the employer is. Strongest signal in the data (67% of fraud vs 16% of real in test) |
| `no_requirements` | Requirements field is empty | Real jobs usually state requirements |
| `no_benefits` | Benefits field is empty | Weak signal |
| `has_salary` | Salary range present | Weak signal |
| `salary_min`, `salary_max`, `salary_ratio` | Parsed from `salary_range` ("30000-40000"); NaN if unparsable | Absurd ranges can be a lure; medians are similar in this data, so low value |
| `telecommuting`, `has_company_logo`, `has_questions` | Original dataset flags (unknown for pasted text) | Missing logo is a strong signal (82% of real postings have one, 33% of fraud) |
| `n_url_tokens`, `n_email_tokens`, `n_phone_tokens` | Count of masked URL / email / phone tokens in all text | Contact details in the text |
| `contact_in_description` | Email or phone appears inside the description | Scams push contact into the description |
| `n_real_emails`, `n_real_urls` | **real-address** counts | Context for the checks below |
| `free_email_domain` | **real-address** Contact email on Gmail/Yahoo/Hotmail etc. | Real employers use a company domain |
| `email_domain_mismatch` | **real-address** Email domain differs from every URL domain in the text | Impersonation of a company |
| `suspicious_url` | **real-address** Shortener (bit.ly), bare IP, or odd TLD (.xyz, .top ...) | Hides the real destination |
| `urgency_count` | Distinct urgency phrases ("apply immediately", "limited slots") | Pressure tactic |
| `payment_request_count` | Distinct fee/payment phrases ("registration fee", "gift card") | Advance-fee scam |
| `messaging_app_count` | WhatsApp / Telegram / WeChat mentions | Moves the victim off-platform |
| `too_good_count` | Phrases like "earn up to", "no experience", "per day" | Unrealistic promises |
| `personal_info_count` | Requests for SSN, passport, bank details, ID | Identity theft |
| `money_mentions` | Count of currency amounts ($500, 1,000 USD) | Pay is stressed in many scams |
| `caps_ratio`, `title_caps_ratio`, `caps_words` | Share of upper-case letters / count of ALL-CAPS words | Shouting style |
| `exclamation_count` | Number of `!` | Hype style |
| `title_len`, `desc_len`, `desc_words`, `total_words` | Text lengths (masked tokens removed) | Fraud postings tend to be shorter |

Phrase lists live in `src/features/keyword_lists.py`; extend them there.

## Reasons shown to the user

`explain_features(feats)` turns triggered features into sentences, e.g.
"Uses a free email domain (Gmail/Yahoo/etc.)", "Asks for a fee or payment",
"No company profile". The mapping is the `REASONS` dict in `rule_features.py`.
