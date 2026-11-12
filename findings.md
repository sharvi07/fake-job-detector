# EDA Findings (Person 1)

## Dataset
- Shape: 17880 rows x 18 columns
- Real: 17014, Fake: 866 (4.84% fake, heavily imbalanced)
- 281 exact duplicates (ignoring job_id): 271 real, 10 fake. Drop them.

## Strong signals
- company_profile missing: fraud 17.7% vs 1.9% when present
- has_company_logo: 82% of real vs 33% of fake
- has_questions: 50% of real vs 29% of fake
- Country: AU 18.7% fraud (214 postings), US 6.9%, most others below 3%
- All 10 fields missing: 22% fraud. None missing: 9% fraud. Middle: about 2.5 to 4%
  (U-shaped, so use all_missing and none_missing flags, not a plain count)

## Weak signals
- telecommuting: 7.4% fake vs 4.1% real
- Description length: fake median 846 vs real 1034 characters
- salary_range, requirements, benefits, department: almost no difference

## Text issues for cleaning
- HTML tags were removed without leaving a space (words glued together). Add a space when removing tags.
- URLs appear as #URL_<hash># tokens. Remove them.
- One scam template (copy of Aker Solutions profile) appears in 52 postings, all fake (about 6% of fake class).
  Models may memorise it, so results may look better than they really are.
- Fake postings reuse normal job words (customer, service, management), so text is hard to separate.

## Recommended text columns to merge
title, company_profile, description, requirements, benefits

## Recommended extra features
missing company_profile, has_company_logo, has_questions, telecommuting,
country (top 10 + Other), text length, all_missing, none_missing
