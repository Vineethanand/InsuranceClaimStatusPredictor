"""
evaluate.py
Metric functions built around the assessment's actual operating constraint:
the review team can only inspect the top 25% of claims by risk score.

Can also be run standalone:
    python src/evaluate.py --model_path outputs/logisticreg.pkl --data_path data/claims_history.csv
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from data.preprocess import load_claims_data, split_data_for_training, ALL_FEATURES, TARGET


def top_quantile_threshold(y_pred_scores: np.ndarray, top_frac: float = 0.25) -> float:
    """Score threshold such that exactly top_frac of `scores` fall at/above it."""
    return float(np.quantile(y_pred_scores, 1 - top_frac))


def metrics_at_top_fraction(y_denied_actual, y_prob, top_frac: float = 0.25) -> dict:
    """
    Get the metrics at the top fraction or the fraction the review team is interested in.
    Calculate the precision, recall, denial capture rate
    """
    threshold = top_quantile_threshold(y_prob, top_frac)
    y_prediction = (y_prob >= threshold).astype(int)

    denials_pred = int(y_prediction.sum())
    actual_denials = int(np.sum(y_denied_actual))
    denials_caught = int(np.sum((y_prediction == 1) & (np.array(y_denied_actual) == 1)))

    return {
        "threshold": threshold,
        "denials_predicted": denials_pred,
        "denial_percentage_of_total": denials_pred / len(y_denied_actual),
        "precision_at_top": precision_score(y_denied_actual, y_prediction, zero_division=0),
        "recall_at_top": recall_score(y_denied_actual, y_prediction, zero_division=0),
        "denial_capture_rate": denials_caught / actual_denials if actual_denials else 0.0,
    }


def get_eval_report(y_denied_actual, y_pred_scores, top_frac: float = 0.25) -> dict:
    report = {
        "pr_auc": average_precision_score(y_denied_actual, y_pred_scores),
        "roc_auc": roc_auc_score(y_denied_actual, y_pred_scores)
    }
    report.update({f"top{int(top_frac*100)}pct_{k}": v for k, v in metrics_at_top_fraction(y_denied_actual, y_pred_scores, top_frac).items()})
    return report


def confusion_at_top_fraction(y_denied_actual, y_pred_scores, top_frac: float = 0.25):
    threshold = top_quantile_threshold(y_pred_scores, top_frac)
    y_pred = (y_pred_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_denied_actual, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained claim-denial model.")
    parser.add_argument("--model_path", required=True, help="Path to a joblib .pkl pipeline")
    parser.add_argument("--data_path", required=True, help="Path to claims_history.csv")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--top_frac", type=float, default=0.25)
    args = parser.parse_args()

    pipeline : Pipeline = joblib.load(args.model_path)
    df_claim_data = load_claims_data(args.data_path)
    train, val, test = split_data_for_training(df_claim_data)
    split_map = {"train": train, "validation": val, "test": test}
    eval_df = split_map[args.split]

    X = eval_df[ALL_FEATURES]
    y = eval_df[TARGET].values
    y_prob_scores = pipeline.predict_proba(X)[:, 1]

    report = get_eval_report(y, y_prob_scores, args.top_frac)
    confusion = confusion_at_top_fraction(y, y_prob_scores, args.top_frac)
    report["confusion_at_top"] = confusion

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()