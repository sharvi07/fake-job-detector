# Comprehensive Technical Report: BERT-Based Fraudulent Job Posting Detection Pipeline

## 1\. Executive Summary & System Architecture

&nbsp;This project establishes an end-to-end, deep learning-based text classification pipeline utilizing a fine-tuned **DistilBERT** (`distilbert-base-uncased`) architecture to automatically distinguish between legitimate and fraudulent job listings.

To ensure production-grade robustness, maintainability, and auditability, the backend architecture is split into three decoupled modules:

1. **`model_bert.py`**: Manages data ingestion, text tokenization (using truncated sequence padding up to 512 tokens), dataset mapping, and fine-tuning via Hugging Face's `Trainer` API with custom metric evaluation callbacks.  
2. **`evaluate_bert.py`**: Computes extensive performance metrics across validation splits, the entire test set (`all_rows`), and a rigorously isolated `unseen_holdout` split to measure real-world out-of-distribution generalization.  
3. **`explainability.py`**: Integrates **LIME (Local Interpretable Model-agnostic Explanations)** to generate word-level local feature attributions, shedding light on model decision-making processes.

---

## 2\. Dataset Preprocessing & Tokenization Strategy

Job postings are inherently unstructured text objects consisting of descriptions, company profiles, requirements, and benefits.

* **Data Cleaning**: Missing values in text fields are filled with empty strings, and categorical metadata (e.g., telecommuting status, employment type) can be concatenated or processed alongside textual data.  
* **Tokenization**: We utilize the `distilbert-base-uncased` tokenizer. Texts are tokenized with a maximum sequence length limit of 512 tokens. Sequences exceeding this length are truncated, while shorter sequences are padded to maintain tensor uniformity.  
* **Class Imbalance Handling**: Because fraudulent postings represent a minority class in real-world scenarios, evaluation metrics heavily emphasize threshold-independent metrics like Precision-Recall Area Under Curve (PR-AUC) alongside standard accuracy.

---

## 3\. Model Training & Fine-Tuning Hyperparameters

The model was fine-tuned using GPU acceleration on the training and validation splits (`data/train.csv` and `data/val.csv`).

### Hyperparameter Configuration

* **Base Architecture**: `distilbert-base-uncased` (6 transformer layers, 768 hidden size, 12 attention heads, $\approx 66$ million parameters)  
* **Training Epochs**: 3  
* **Per-Device Batch Size**: 16  
* **Learning Rate**: $2\times 1{0}^{-5}$ with linear warmup  
* **Weight Decay**: 0.01  
* **Optimizer**: AdamW (`adam_beta1 = 0.9`, `adam_beta2 = 0.999`)

### Epoch-wise Validation Metrics Summary

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1-Score | PR-AUC | Accuracy |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | 0.0238 | 0.0842 | 0.7963 | 0.6667 | 0.7257 | 0.8094 | 0.9754 |
| **2** | 0.0570 | 0.0738 | 0.8981 | 0.7519 | 0.8186 | 0.8648 | 0.9837 |
| **3** | 0.0198 | 0.0815 | 0.8707 | 0.7829 | 0.8245 | 0.8673 | 0.9837 |

The training dynamics show steady convergence through Epoch 2, with F1-score stabilizing near 0.8245 by Epoch 3\.

---

## 4\. Rigorous Test Set Evaluation & Generalization Analysis

To validate model performance against overfitting, the fine-tuned checkpoint was evaluated against independent test splits (`data/test.csv`).

| Evaluation Split | Total Rows | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Accuracy |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **All Test Rows** | 2,640 | 0.8889 | 0.8125 | 0.8490 | 0.9056 | 0.9814 | 0.9860 |
| **Unseen Holdout** | 2,289 | 0.8588 | 0.7526 | 0.8022 | 0.8633 | 0.9738 | 0.9843 |

### Key Findings:

1. **High Precision & ROC-AUC**: The model achieves a ROC-AUC of $>0.97$ across both evaluation subsets, indicating exceptional discriminative capacity between genuine and fraudulent classes.  
2. **Generalization Drop**: On the strictly `unseen_holdout` split, the F1-score drops moderately from 0.8490 to 0.8022. This minor degradation is expected due to domain shift in vocabulary and phrasing used by newly emerged fraudulent templates.

---

## 5\. Model Explainability via LIME

Black-box deep learning models are prone to learning spurious correlations (e.g., specific company names or boilerplates). To mitigate this, `explainability.py` applies LIME to provide local feature attributions.

### Case Study: Legitimate Job Posting

* **Sample Text**: *"We are hiring a remote data analyst. Apply through our official careers portal."*  
* **Model Prediction**: Legitimate (**97.73%** confidence), Fraudulent (**2.27%** confidence).  
* **Top LIME Local Weights (Pushing towards Legitimate Class)**:  
  * `analyst`: $-0.2395$  
  * `our`: $-0.1808$  
  * `careers`: $-0.1253$  
  * `remote`: $-0.1190$  
  * `portal`: $-0.1148$

The analysis confirms that the transformer model correctly anchors its decisions on semantic concepts associated with professional hiring workflows (`careers`, `portal`, `analyst`) rather than relying on noisy background tokens.

---

## 6\. Recommendations & Future Enhancements

1. **Incorporate Metadata Features**: While text content is powerful, combining transformer embeddings with tabular metadata (e.g., `has_company_logo`, `has_questions`, `employment_type`) via a late-fusion architecture could further reduce false negatives.  
2. **Active Learning Loop**: Implement a feedback mechanism in the evaluation script to sample high-uncertainty holdout predictions for human annotation and subsequent model retraining.  
3. **Threshold Tuning**: Adjust the classification decision threshold dynamically based on business costs associated with false positives (flagging real jobs) versus false negatives (allowing scams).