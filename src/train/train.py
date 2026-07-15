"""
train.py
Train a claim status predictor classifier

Usage:
    python src/train/train.py --data_path dataset/claims_history.csv --seed 42
"""

import argparse
import json
import os

import joblib
import numpy as np

from data.preprocess import load_claims_data, split_data_for_training, ALL_FEATURES, TARGET
from model.claim_predictor_model import build_model_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train a claim denial classifier.")
    parser.add_argument("--data_path", required=True, help="Path to claims_history.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--top_frac", type=float, default=0.25,
                         help="Fraction of claims the review team can inspect.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df_curr_data = load_claims_data(args.data_path)
    train, val, test = split_data_for_training(df_curr_data)

    X_train, y_train = train[ALL_FEATURES], train[TARGET].values
    X_val, y_val = val[ALL_FEATURES], val[TARGET].values
    X_test, y_test = test[ALL_FEATURES], test[TARGET].values


    pipeline = build_model_pipeline(seed=args.seed)
    pipeline.fit(X_train, y_train)

    val_scores = pipeline.predict_proba(X_val)[:, 1]
    val_report = full_report(y_val, val_scores, args.top_frac)

    test_scores = pipeline.predict_proba(X_test)[:, 1]
    test_report = full_report(y_test, test_scores, args.top_frac)

    model_path = os.path.join(args.output_dir, f"{args.model}.pkl")
    joblib.dump(pipeline, model_path)

    metrics = {"model": "logistic_regression", "seed": args.seed, "validation": val_report, "test": test_report}
    metrics_path = os.path.join(args.output_dir, f"{args.model}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()