"""
evaluate_bert.py
Person 3 Deliverable: Evaluation Script for DistilBERT
Computes performance metrics across validation, all test rows, and unseen test splits.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def evaluate_bert_model(test_texts, test_labels, model_path="./models/bert-fake-job"):
    """
    Evaluates the fine-tuned BERT model on raw text lists/Series and returns metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        list(test_texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    all_probs = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch = {k: v[i : i + batch_size] for k, v in inputs.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.extend(probs)

    pred_probs = np.array(all_probs)[:, 1]
    preds = (pred_probs >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(test_labels, preds)
    pr_auc = average_precision_score(test_labels, pred_probs)
    roc_auc = roc_auc_score(test_labels, pred_probs)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "accuracy": float(acc),
    }
    return metrics, pred_probs


def evaluate(csv_path, model_path, text_column="text", label_column="fraudulent", output_path=None):
    """
    Notebook-facing evaluation function matching the exact signature in 05_bert_evaluation.ipynb.
    Safely loads CSV data, computes metrics on all_rows and unseen_holdout, and exports JSON.
    """
    df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    texts = df[text_column].fillna("")
    labels = df[label_column].values

    all_metrics, _ = evaluate_bert_model(texts, labels, model_path=model_path)

    # Check if train/val splits exist alongside test.csv to compute true unseen_holdout
    data_dir = Path(csv_path).parent
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"

    if train_path.is_file() and val_path.is_file():
        tr_df = pd.read_csv(train_path, on_bad_lines="skip", engine="python")
        va_df = pd.read_csv(val_path, on_bad_lines="skip", engine="python")
        seen_texts = set(tr_df[text_column].dropna()).union(set(va_df[text_column].dropna()))
        unseen_mask = ~df[text_column].isin(seen_texts)
        unseen_df = df[unseen_mask]
        unseen_metrics, _ = evaluate_bert_model(
            unseen_df[text_column].fillna(""),
            unseen_df[label_column].values,
            model_path=model_path,
        )
        unseen_count = len(unseen_df)
    else:
        unseen_metrics = all_metrics
        unseen_count = len(texts)

    results = {
        "all_rows": all_metrics,
        "unseen_holdout": unseen_metrics,
        "row_counts": {
            "all_rows": len(texts),
            "unseen_holdout": unseen_count,
        },
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results