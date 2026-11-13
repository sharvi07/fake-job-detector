"""Helpers to pull emails and URLs out of text and compare their domains.

Note: the Kaggle/EMSCAD data masks emails, URLs and phones as hashed tokens
(#EMAIL_...#, #URL_...#, #PHONE_...#). The helpers here work on REAL addresses
(pasted text, uploaded PDFs/DOCX); the masked tokens are only counted.
"""
import re
from urllib.parse import urlparse

from .keyword_lists import FREE_EMAIL_DOMAINS, SUSPICIOUS_TLDS, URL_SHORTENERS

EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')]+", re.I)
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

MASK_URL_RE = re.compile(r"#URL_[0-9a-f]+#")
MASK_EMAIL_RE = re.compile(r"#EMAIL_[0-9a-f]+#")
MASK_PHONE_RE = re.compile(r"#PHONE_[0-9a-f]+#")

_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}


def registered_domain(host):
    """Cheap registered-domain guess (no network): example.co.uk -> example.co.uk."""
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def extract_email_domains(text):
    return [registered_domain(m.group(1)) for m in EMAIL_RE.finditer(text or "")]


def extract_urls(text):
    return [u.rstrip(".,;") for u in URL_RE.findall(text or "")]


def url_host(url):
    if not url.lower().startswith("http"):
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()


def is_free_email_domain(domain):
    return domain in FREE_EMAIL_DOMAINS


def is_suspicious_url(url):
    host = url_host(url)
    if not host:
        return False
    if IP_HOST_RE.match(host):
        return True
    if registered_domain(host) in URL_SHORTENERS:
        return True
    return host.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS


def email_url_mismatch(email_domains, urls):
    """1 if there are emails AND URLs and no email domain matches any URL domain."""
    if not email_domains or not urls:
        return 0
    url_domains = {registered_domain(url_host(u)) for u in urls}
    non_free = [d for d in email_domains if not is_free_email_domain(d)]
    if not non_free:
        return 0  # free-email case is covered by its own feature
    return int(not any(d in url_domains for d in non_free))
