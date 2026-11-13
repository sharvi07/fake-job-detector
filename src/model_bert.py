"""
model_bert.py
Person 3 Deliverable (Step 4: Fine-tuning DistilBERT for Fake Job Detection)

This module handles fine-tuning a pre-trained DistilBERT sequence classification
model on job posting text data and provides clean helper functions for loading
and running inference.
"""

import os
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    """
    Computes evaluation metrics tailored for imbalanced fraud detection.
    Focuses on Precision, Recall, F1-Score, and PR-AUC rather than raw accuracy.
    """
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(labels, probs)
    acc = accuracy_score(labels, preds)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "accuracy": acc,
    }


def fine_tune_bert(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    model_name="distilbert-base-uncased",
    output_dir="./models/bert-fake-job",
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
):
    """
    Fine-tunes DistilBERT on job posting text and saves the model artifacts.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels)})
    val_dataset = Dataset.from_dict({"text": list(val_texts), "label": list(val_labels)})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results_bert",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved successfully to {output_dir}")

    return model, tokenizer


def load_bert_model(model_path="./models/bert-fake-job"):
    """
    Loads a saved fine-tuned BERT model and tokenizer for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device