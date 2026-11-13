"""Phrase lists used by the rule-based fraud features.

All phrases are lowercase and matched on word boundaries. Extend freely; every
list feeds one count feature in rule_features.py.
"""

URGENCY = [
    "apply immediately", "urgent", "urgently hiring", "immediate start",
    "immediate joining", "start immediately", "limited slots",
    "limited positions", "limited openings", "hurry", "act now", "apply now",
    "asap", "last chance", "today only", "don't miss", "dont miss",
]

PAYMENT_REQUEST = [
    "registration fee", "processing fee", "application fee", "training fee",
    "security deposit", "refundable deposit", "pay a fee", "send money",
    "wire transfer", "western union", "moneygram", "gift card", "bitcoin",
    "advance payment", "starter kit", "purchase equipment", "buy equipment",
    "bank account details", "cheque will be", "check will be sent",
]

MESSAGING_APPS = [
    "whatsapp", "telegram", "signal app", "wechat", "hangouts", "imessage",
    "text us", "text me",
]

TOO_GOOD_TO_BE_TRUE = [
    "earn up to", "make money", "easy money", "quick money", "per day",
    "weekly pay", "get paid daily", "no experience", "no interview",
    "guaranteed income", "unlimited earning", "be your own boss",
    "passive income", "financial freedom", "work from home",
]

PERSONAL_INFO_REQUEST = [
    "social security", "ssn", "passport copy", "copy of id", "id card",
    "date of birth", "bank details", "credit card", "driver's license",
    "drivers license", "scan of your",
]

FREE_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "ymail.com",
    "hotmail.com", "outlook.com", "live.com", "msn.com", "aol.com",
    "mail.com", "gmx.com", "yandex.com", "protonmail.com", "icloud.com",
    "rediffmail.com", "zoho.com",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly",
    "rb.gy", "shorturl.at", "tiny.cc",
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "work", "icu", "tk", "ml", "ga", "cf", "gq", "buzz",
    "loan", "rest",
}
