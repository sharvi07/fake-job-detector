"""Streamlit app: Fake Job Offer Detector (final integration).

Run from the project root:  streamlit run app/app.py

Model input = [33 rule features (FEATURE_NAMES order)] + [TF-IDF terms],
exactly as listed in models/meta.json.
"""
import json
import pickle
import sys
import joblib
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.sparse import csr_matrix, hstack
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.extract import extract_text  # noqa: E402
from src.features.rule_features import (FEATURE_NAMES, explain_features,  # noqa: E402
                                        extract_features)

# Masks real emails/URLs -> "emailtoken"/"urltoken", then cleans (matches training)
from src.features.text_clean import prepare_for_cleaning  # noqa: E402

MODELS = ROOT / "models"
BERT_DIR = MODELS / "bert-fake-job"


@st.cache_resource
def load_xgb():
    """Load XGBoost + TF-IDF vectorizer + meta, and verify they line up."""
    model = XGBClassifier()
    model.load_model(str(MODELS / "xgboost_model.json"))
    vec = joblib.load(MODELS / "tfidf_vectorizer.pkl")
    with open(MODELS / "meta.json") as f:
        meta = json.load(f)

    n_rule = meta["n_rule_features"]
    names = meta["feature_names"]
    if names[:n_rule] != FEATURE_NAMES:
        raise ValueError("Rule feature order in meta.json differs from rule_features.py")
    if list(vec.get_feature_names_out()) != names[n_rule:]:
        raise ValueError("Vectorizer vocabulary differs from meta.json")
    return model, vec, meta


@st.cache_resource
def load_bert():
    """Load Member 3's BERT + LIME wrapper (optional, slow)."""
    from src.explainability import BERTExplainer
    return BERTExplainer(model_path=str(BERT_DIR))


def build_input(text, vec):
    """Same layout as training: rule features first, then TF-IDF."""
    feats = extract_features(text)  # raw text -> real emails/URLs are detected
    rule_row = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=float)
    cleaned = prepare_for_cleaning(text)
    X = hstack([csr_matrix(rule_row), vec.transform([cleaned])], format="csr")
    return X, feats


# ------------------------------- UI -------------------------------
st.set_page_config(page_title="Fake Job Offer Detector", page_icon="🚨")
st.title("🚨 Fake Job Offer Detector")
st.caption("Paste a job description below to check for scam risks.")

try:
    model, vec, meta = load_xgb()
except Exception as e:
    st.error(f"Could not load the model files from /models: {e}")
    st.stop()

threshold = meta["threshold"]
use_bert = False
if BERT_DIR.exists():
    use_bert = st.sidebar.checkbox("Also explain with BERT + LIME (slow)", value=False)

uploaded = st.file_uploader("Upload job offer (PDF/DOCX)", type=["pdf", "docx"])
pasted = st.text_area("Or paste job description", height=220)

if st.button("Analyse"):
    try:
        text = extract_text(uploaded, pasted).strip()
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()
    if not text:
        st.warning("Please upload a file or paste a job description.")
        st.stop()

    X, feats = build_input(text, vec)
    prob = float(model.predict_proba(X)[0, 1])

    
    # --------------------------------------------------
    # HYBRID VERDICT + RULE-BASED EXPLANATION
    # --------------------------------------------------

    reasons = explain_features(feats, top_n=12)
    n_signals = len(reasons)

    HYBRID_MIN_SIGNALS = 4

    if prob >= threshold:
        st.error(
            f"Likely FRAUDULENT: {prob:.0%} fraud probability"
        )

    elif n_signals >= HYBRID_MIN_SIGNALS:
        st.error(
            f"Likely FRAUDULENT: {n_signals} scam signals found "
            f"(model score alone: {prob:.0%})"
        )

    elif prob >= threshold / 2:
        st.warning(
            f"Suspicious: {prob:.0%} fraud probability. "
            "Verify the employer before applying."
        )

    else:
        st.success(
            f"Looks legitimate: {prob:.0%} fraud probability. "
            "Still verify the employer."
        )

    st.progress(min(prob, 1.0))

    st.subheader("Signals detected")

    if reasons:
        for r in reasons:
            st.markdown(f"🔴 {r}")
    else:
        st.write("No rule-based red flags found.")

    if use_bert:
        with st.spinner("Running BERT + LIME (this can take a few minutes on CPU)..."):
            try:
                bert = load_bert()

                def predict_batched(texts, batch_size=8):
                    """Run BERT in small batches so memory stays low."""
                    texts = list(texts)
                    out = [bert.predictor.predict(texts[i:i + batch_size])
                           for i in range(0, len(texts), batch_size)]
                    return np.vstack(out)

                bert_prob = float(predict_batched([text])[0][1])
                exp = bert.explainer.explainer.explain_instance(
                    text, predict_batched, num_features=8,
                    labels=(1,), num_samples=200)
                weights = exp.as_list(label=1)
                st.subheader("BERT explanation")
                st.write(f"BERT fraud probability: {bert_prob:.0%}")
                if bert_prob >= 0.5:
                    words = [w for w, wt in weights if wt > 0]
                    st.write("Words pushing towards fraud: " + ", ".join(words))
                else:
                    words = [w for w, wt in weights if wt < 0]
                    st.write("Words pushing towards legitimate: " + ", ".join(words))
                components.html(
                    f'<div style="background:white;padding:8px">{exp.as_html()}</div>',
                    height=420, scrolling=True)
            except Exception as e:
                st.error(f"BERT explanation failed: {e}")