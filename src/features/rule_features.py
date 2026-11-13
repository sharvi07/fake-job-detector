"""Rule-based fraud signals (Step 3).

Main entry points
-----------------
extract_features(row_or_text) -> dict        one posting
extract_features_df(df)       -> DataFrame   many postings (fixed column order)
explain_features(feats, top_n) -> list[str]  human-readable reasons

`row_or_text` is either a dict / pandas Series with the dataset fields
(title, company_profile, description, requirements, benefits, salary_range,
telecommuting, has_company_logo, has_questions, optional text_raw) or a plain
string (pasted/uploaded posting). For a plain string the structured-field
features are NaN (unknown), which XGBoost handles natively.
"""
import math
import re

import numpy as np
import pandas as pd

from .keyword_lists import (MESSAGING_APPS, PAYMENT_REQUEST, PERSONAL_INFO_REQUEST,
                            TOO_GOOD_TO_BE_TRUE, URGENCY)
from .url_email_utils import (MASK_EMAIL_RE, MASK_PHONE_RE, MASK_URL_RE,
                              email_url_mismatch, extract_email_domains,
                              extract_urls, is_free_email_domain,
                              is_suspicious_url)

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
SALARY_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
MONEY_RE = re.compile(r"[$£€₹]\s?\d[\d,]*|\b\d[\d,]*\s?(?:usd|dollars)\b", re.I)
CAPS_WORD_RE = re.compile(r"\b[A-Z]{4,}\b")

_MOJIBAKE_FIXES = {"â€™": "'", "â€œ": '"', "â€\x9d": '"', "â€“": "-", "â€”": "-",
                   "â€˜": "'", "Â ": " ", "Â": ""}


def _compile(phrases):
    return [re.compile(r"\b" + re.escape(p) + r"\b") for p in phrases]


_PATTERNS = {
    "urgency": _compile(URGENCY),
    "payment": _compile(PAYMENT_REQUEST),
    "messaging": _compile(MESSAGING_APPS),
    "too_good": _compile(TOO_GOOD_TO_BE_TRUE),
    "personal_info": _compile(PERSONAL_INFO_REQUEST),
}


def fix_mojibake(s):
    if "â" in s or "Ã" in s or "Â" in s:
        try:
            return s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            for bad, good in _MOJIBAKE_FIXES.items():
                s = s.replace(bad, good)
    return s


def _s(value):
    """Value -> clean string ('' for missing)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return fix_mojibake(str(value))


def _count(name, text_lower):
    """Number of DISTINCT phrases from list `name` that appear in text."""
    return sum(1 for p in _PATTERNS[name] if p.search(text_lower))


def _strip_masks(text):
    for rx in (MASK_URL_RE, MASK_EMAIL_RE, MASK_PHONE_RE):
        text = rx.sub(" ", text)
    return text


def _caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def _salary(value):
    m = SALARY_RE.match(_s(value))
    if not m:
        return np.nan, np.nan
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo, hi) if hi >= lo and hi > 0 else (np.nan, np.nan)


def _flag(value):
    return np.nan if value is None or (isinstance(value, float) and math.isnan(value)) else int(value)


def extract_features(row_or_text):
    structured = not isinstance(row_or_text, str)
    row = {"description": row_or_text} if not structured else dict(row_or_text)
    f = {}

    fields = {k: _s(row.get(k)) for k in TEXT_FIELDS}
    raw = _s(row.get("text_raw")) or " ".join(fields[k] for k in TEXT_FIELDS)
    body = _strip_masks(raw)
    low = body.lower().replace("’", "'")

    # --- missing-field signals (unknown for pasted text) ---
    nan = np.nan
    f["no_company_profile"] = int(not fields["company_profile"].strip()) if structured else nan
    f["no_requirements"] = int(not fields["requirements"].strip()) if structured else nan
    f["no_benefits"] = int(not fields["benefits"].strip()) if structured else nan
    f["has_salary"] = int(bool(_s(row.get("salary_range")).strip())) if structured else nan
    lo, hi = _salary(row.get("salary_range")) if structured else (nan, nan)
    f["salary_min"], f["salary_max"] = lo, hi
    f["salary_ratio"] = hi / lo if lo and not np.isnan(lo) and lo > 0 else nan
    f["telecommuting"] = _flag(row.get("telecommuting")) if structured else nan
    f["has_company_logo"] = _flag(row.get("has_company_logo")) if structured else nan
    f["has_questions"] = _flag(row.get("has_questions")) if structured else nan

    # --- contact details (masked tokens and real addresses) ---
    f["n_url_tokens"] = len(MASK_URL_RE.findall(raw))
    f["n_email_tokens"] = len(MASK_EMAIL_RE.findall(raw))
    f["n_phone_tokens"] = len(MASK_PHONE_RE.findall(raw))
    desc_raw = fields["description"]
    f["contact_in_description"] = int(bool(
        MASK_EMAIL_RE.search(desc_raw) or MASK_PHONE_RE.search(desc_raw)
        or extract_email_domains(desc_raw)))

    emails = extract_email_domains(body)
    urls = extract_urls(body)
    f["n_real_emails"] = len(emails)
    f["free_email_domain"] = int(any(is_free_email_domain(d) for d in emails))
    f["email_domain_mismatch"] = email_url_mismatch(emails, urls)
    f["n_real_urls"] = len(urls)
    f["suspicious_url"] = int(any(is_suspicious_url(u) for u in urls))

    # --- language signals ---
    f["urgency_count"] = _count("urgency", low)
    f["payment_request_count"] = _count("payment", low)
    f["messaging_app_count"] = _count("messaging", low)
    f["too_good_count"] = _count("too_good", low)
    f["personal_info_count"] = _count("personal_info", low)
    f["money_mentions"] = len(MONEY_RE.findall(body))

    f["caps_ratio"] = _caps_ratio(body)
    f["title_caps_ratio"] = _caps_ratio(fields["title"])
    f["caps_words"] = len(CAPS_WORD_RE.findall(body))
    f["exclamation_count"] = body.count("!")

    f["title_len"] = len(fields["title"])
    f["desc_len"] = len(_strip_masks(desc_raw))
    f["desc_words"] = len(_strip_masks(desc_raw).split())
    f["total_words"] = len(body.split())
    return f


FEATURE_NAMES = list(extract_features({"description": ""}).keys())


def extract_features_df(df):
    rows = [extract_features(r) for r in df.to_dict("records")]
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=df.index)


# Feature -> (trigger condition, reason shown to the user)
REASONS = {
    "free_email_domain": (lambda v: v >= 1, "Uses a free email domain (Gmail/Yahoo/etc.)"),
    "email_domain_mismatch": (lambda v: v >= 1, "Contact email domain does not match the company website"),
    "no_company_profile": (lambda v: v == 1, "No company profile"),
    "no_requirements": (lambda v: v == 1, "No requirements listed"),
    "payment_request_count": (lambda v: v >= 1, "Asks for a fee or payment"),
    "messaging_app_count": (lambda v: v >= 1, "Asks to contact via WhatsApp/Telegram or similar"),
    "urgency_count": (lambda v: v >= 1, "Uses urgency language (\"apply immediately\")"),
    "too_good_count": (lambda v: v >= 2, "Promises easy money or no experience needed"),
    "personal_info_count": (lambda v: v >= 1, "Asks for sensitive personal information"),
    "suspicious_url": (lambda v: v >= 1, "Contains a suspicious or shortened link"),
    "caps_ratio": (lambda v: v >= 0.25, "Unusually high share of ALL CAPS text"),
    "exclamation_count": (lambda v: v >= 5, "Excessive exclamation marks"),
}


def explain_features(feats, top_n=5):
    """Readable reasons for the triggered signals, in the order of REASONS."""
    out = []
    for name, (cond, msg) in REASONS.items():
        v = feats.get(name)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and cond(v):
            out.append(msg)
    return out[:top_n]
