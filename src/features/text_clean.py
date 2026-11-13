"""Text cleaning identical to Person 1's `clean` (baseline notebook), so the
TF-IDF used by XGBoost sees the same kind of text as the baseline.

`prepare_for_cleaning` additionally maps REAL emails/URLs (pasted or uploaded
postings) to the same tokens the dataset uses for its masked ones, so the
model's `emailtoken` / `urltoken` words mean the same thing at prediction time.
"""
import re

from .url_email_utils import EMAIL_RE, URL_RE


def clean_text(t):
    t = re.sub(r"#(URL|EMAIL|PHONE)_[0-9a-f]+#", lambda m: f" {m.group(1).lower()}token ", t)
    t = re.sub(r"<[^>]+>", " ", t)               # HTML tags -> space
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)   # split glued words like overviewAker
    t = re.sub(r"[^a-zA-Z ]", " ", t).lower()
    return re.sub(r"\s+", " ", t).strip()


def prepare_for_cleaning(text):
    """Replace real emails/URLs by the dataset's mask tokens, then clean."""
    text = EMAIL_RE.sub(" #EMAIL_0# ", text or "")
    text = URL_RE.sub(" #URL_0# ", text)
    return clean_text(text)
