import json

import numpy as np
import pandas as pd

from src.models import xgboost_model as xm


def test_dense_block_keeps_explicit_zeros_and_nan():
    a = np.array([[0.0, 1.0, np.nan], [0.0, 0.0, 2.0]])
    m = xm._dense_block_as_csr(a)
    assert m.nnz == 6                       # every cell stored, zeros included
    assert np.allclose(m.toarray()[0, :2], [0.0, 1.0]) and np.isnan(m.toarray()[0, 2])


def test_best_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    thr = xm.best_f1_threshold(y, p)
    m = xm.metrics_at(y, p, thr)
    assert m["tp"] + m["fn"] == 2 and m["tn"] + m["fp"] == 3
    assert m["recall"] == 1.0


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fraud = int(i % 10 == 0)
        if fraud:
            desc = "earn money fast from home no experience needed pay small fee " + " ".join(rng.choice(["alpha", "beta"], 5))
            profile = ""
        else:
            desc = "we are hiring an engineer to build software with our team " + " ".join(rng.choice(["gamma", "delta"], 5))
            profile = "we are a growing software company"
        rows.append({"job_id": seed * 10000 + i, "title": "Job", "company_profile": profile or None,
                     "description": desc, "requirements": "degree", "benefits": None,
                     "salary_range": None, "telecommuting": 0, "has_company_logo": 0 if fraud else 1,
                     "has_questions": 0, "fraudulent": fraud, "text_raw": f"Job {profile} {desc} {i}",
                     "text": f"job {profile} {desc}".lower()})
    return pd.DataFrame(rows)


def test_train_and_predict_end_to_end(tmp_path):
    paths = {}
    for name, n, seed in (("train", 400, 1), ("val", 200, 2), ("test", 200, 3)):
        paths[name] = str(tmp_path / f"{name}.csv")
        _synthetic(n, seed).to_csv(paths[name], index=False)
    mdir, rdir = str(tmp_path / "models"), str(tmp_path / "results")
    res = xm.train(paths["train"], paths["val"], paths["test"], n_iter=1, max_features=200,
                   model_dir=mdir, results_dir=rdir, ablation=False)
    assert res["test"]["pr_auc"] > 0.9
    for f in ("xgboost_metrics.json", "best_params.json", "xgboost_pr_curve.png",
              "xgboost_confusion_matrix.png", "feature_importance.png", "xgboost_error_analysis.csv"):
        assert (tmp_path / "results" / f).exists()
    json.load(open(tmp_path / "results" / "xgboost_metrics.json"))

    out = xm.predict("earn money fast from home no experience needed pay small fee", model_dir=mdir)
    assert set(out) == {"label", "prob", "reasons"} and 0.0 <= out["prob"] <= 1.0
    assert isinstance(out["reasons"], list)
