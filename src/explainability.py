"""
explainability.py
Person 3 Deliverable (Step 5: Model Explainability using LIME)

Provides local word-level explanations for BERT predictions and exposes the
standardized classes and predict(text) interface.
"""

import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FakeJobPredictor:
    """
    Predictor wrapper class matching the evaluation notebook's expected interface.
    """

    def __init__(self, model_path="./models/bert-fake-job"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        return probs


class FakeJobExplainer:
    """
    LIME explainer wrapper matching the notebook's expected interface.
    """

    def __init__(self, predictor: FakeJobPredictor):
        self.predictor = predictor
        self.explainer = LimeTextExplainer(class_names=["legitimate", "fraudulent"])

    def explain(self, text, num_features=10):
        exp = self.explainer.explain_instance(
            text,
            self.predictor.predict,
            num_features=num_features,
            labels=(1,),
        )
        return exp


class BERTExplainer:
    """
    Comprehensive wrapper to fulfill the target output format:
    'Fraud 87%: reasons: fee, registration, whatsapp.'
    """

    def __init__(self, model_path="./models/bert-fake-job"):
        self.predictor = FakeJobPredictor(model_path)
        self.explainer = FakeJobExplainer(self.predictor)

    def explain_posting(self, text, num_features=5):
        probs = self.predictor.predict([text])[0]
        fraud_prob = float(probs[1])
        is_fraud = fraud_prob >= 0.5
        label = "Fraud" if is_fraud else "Real"

        exp = self.explainer.explain(text, num_features=num_features)
        local_exp = exp.as_list(label=1)

        reasons = [word for word, weight in local_exp if weight > 0]
        if not reasons:
            reasons = ["suspicious wording patterns"]

        reasons_str = ", ".join(reasons)
        formatted_output = f"{label} {fraud_prob * 100:.0f}%: reasons: {reasons_str}."

        return {
            "label": label,
            "probability": fraud_prob,
            "reasons": reasons,
            "formatted_string": formatted_output,
            "lime_html": exp.as_html()
        }


# Standardized team interface for Person 4's Streamlit integration
def predict(text, model_path="./models/bert-fake-job"):
    explainer = BERTExplainer(model_path=model_path)
    res = explainer.explain_posting(text)
    return res["label"], res["probability"], res["reasons"]