import math

from src.features.rule_features import (FEATURE_NAMES, explain_features,
                                        extract_features, extract_features_df)
from src.features.url_email_utils import registered_domain


def test_free_email_and_payment_and_messaging():
    f = extract_features("URGENT! Pay a registration fee. Contact john@gmail.com or WhatsApp us.")
    assert f["free_email_domain"] == 1
    assert f["payment_request_count"] >= 1
    assert f["messaging_app_count"] == 1
    assert f["urgency_count"] >= 1


def test_email_domain_mismatch():
    f = extract_features("Apply via hr@scamco.com. See www.realcorp.com for details.")
    assert f["email_domain_mismatch"] == 1
    g = extract_features("Apply via hr@realcorp.com. See www.realcorp.com for details.")
    assert g["email_domain_mismatch"] == 0


def test_suspicious_url():
    assert extract_features("Register at http://bit.ly/abc123")["suspicious_url"] == 1
    assert extract_features("Visit https://www.google.com")["suspicious_url"] == 0


def test_caps_ratio_ignores_masked_tokens():
    f = extract_features("hello world #URL_abcdef0123# #EMAIL_abcdef0123#")
    assert f["caps_ratio"] == 0.0
    assert f["n_url_tokens"] == 1 and f["n_email_tokens"] == 1


def test_missing_fields_structured_vs_plain_text():
    row = {"title": "Dev", "description": "Write code", "company_profile": None}
    assert extract_features(row)["no_company_profile"] == 1
    assert math.isnan(extract_features("Write code")["no_company_profile"])


def test_salary_parsing():
    f = extract_features({"description": "x", "salary_range": "30000-40000"})
    assert f["salary_min"] == 30000 and f["salary_max"] == 40000
    assert math.isnan(extract_features({"description": "x", "salary_range": "Jan-15"})["salary_min"])


def test_mojibake_fixed():
    f = extract_features({"description": "Weâ€™re hiring"})
    assert f["desc_len"] == len("We're hiring")


def test_df_has_fixed_columns_and_reasons():
    import pandas as pd
    df = pd.DataFrame([{"description": "Pay a fee to john@yahoo.com"}])
    out = extract_features_df(df)
    assert list(out.columns) == FEATURE_NAMES
    reasons = explain_features(out.iloc[0].to_dict())
    assert "Asks for a fee or payment" in reasons


def test_registered_domain():
    assert registered_domain("mail.example.co.uk") == "example.co.uk"
    assert registered_domain("www.example.com") == "example.com"
