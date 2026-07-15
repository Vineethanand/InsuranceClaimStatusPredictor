"""
Inference script to run the model on a new dataset
"""

import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from data.preprocess import load_claims_data, ALL_FEATURES, get_feature_names_from_preprocessor
from evaluate.evaluate import top_quantile_threshold

# Human-readable labels for engineered / less-obvious feature names, used
# when building the top_risk_factors explanation strings.
FEATURE_DESCRIPTIONS = {
    "prior_auth_vs_has_auth": "prior authorization required but not on file",
    "ref_req_vs_ref_present": "referral required but not on file",
    "payment_ratio": "ratio of expected payment to billed amount",
    "prior_auth_required": "prior authorization required",
    "has_prior_auth": "prior authorization on file",
    "referral_required": "referral required",
    "referral_present": "referral on file",
    "is_in_network": "provider is in-network",
    "missing_documentation_flag": "documentation missing",
    "eligibility_verified": "eligibility verified in advance",
    "days_to_submit": "days between service and submission",
    "total_billed": "total amount billed",
    "expected_payment": "expected payment",
    "num_procedures": "number of procedures",
    "num_diagnoses": "number of diagnoses",
}


def describe_feature(name: str) -> str:
    """Turn a raw (possibly one-hot-expanded) feature name into a readable label."""
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    # one-hot columns look like "payer_type_BCBS" or "visit_type_Inpatient"
    for base in ["payer_id", "payer_type", "visit_type"]:
        if name.startswith(base + "_"):
            value = name[len(base) + 1:]
            return f"{base.replace('_', ' ')} = {value}"
    return name


def top_risk_factors_for_row(row_transformed, coefs, feature_names, top_n=3):
    """Return the top_n features (by |contribution|) that pushed this row's
    score, phrased as short human-readable strings."""
    contributions = row_transformed * coefs
    order = np.argsort(-np.abs(contributions))[:top_n]
    factors = []
    for i in order:
        direction = "increases risk" if contributions[i] > 0 else "lowers risk"
        factors.append(f"{describe_feature(feature_names[i])} ({direction})")
    return "; ".join(factors)


def assign_risk_slab(scores: np.ndarray) -> Tuple[np.ndarray, float]:
    """High = top 25% by score (the reviewable quartile), Medium = next 25%,
    Low = bottom 50%. Thresholds are recomputed on this batch's own score
    distribution, since the review team's rule is 'top 25% of whatever came
    in today', not a fixed probability cutoff"""
    high_cut = top_quantile_threshold(scores, 0.25)
    medium_cut = top_quantile_threshold(scores, 0.50)
    slabs = np.where(scores >= high_cut, "High",
             np.where(scores >= medium_cut, "Medium", "Low"))
    return slabs, high_cut


def main():
    parser = argparse.ArgumentParser(description="Score current_claims.csv.")
    parser.add_argument("--model_path", default="outputs/logisticreg.pkl")
    parser.add_argument("--data_path", default="data/current_claims.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--top_frac", type=float, default=0.25)
    args = parser.parse_args()

    # Load the model
    pipeline :  Pipeline = joblib.load(args.model_path)
    df = load_claims_data(args.data_path)
    X = df[ALL_FEATURES]


    # Using the saved model find the probabilty of denial for each row
    denial_probability = pipeline.predict_proba(X)[:, 1]

    # predicted_denial / risk_slab: thresholded at the top 25% of THIS batch
    risk_slab, high_cut = assign_risk_slab(denial_probability)
    predicted_denial = (denial_probability >= high_cut).astype(int)

    # top_risk_factors: per-row coefficient contributions
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_preprocessor(preprocessor)
    X_transformed = preprocessor.transform(X)
    coefs = classifier.coef_[0]

    top_risk_factors = [
        top_risk_factors_for_row(X_transformed[i], coefs, feature_names)
        for i in range(X_transformed.shape[0])
    ]

    out = pd.DataFrame({
        "claim_id": df["claim_id"],
        "denial_probability": denial_probability,
        "predicted_denial": predicted_denial,
        "risk_slab": risk_slab,
        "top_risk_factors": top_risk_factors,
        "explanation": "This is a sample explanation.",
        # The explanation is a dummy placeholder.
        # In the real scenario, an LLM API would be placed to generate a natural language explanation
        # based on the top_risk_factors and other relevant information.
        # Use the utils.py inside src/utils/utils.py to generate the explanation based on the prompt template.
        # Pass the risk factors and probability to the utility function
        # Get the claim facts using the utility inside utility.py
    })
    out = out.sort_values("denial_probability", ascending=False).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "predictions_current_claims.csv")
    out.to_csv(output_path, index=False)

    print(f"Scored {len(out)} claims.")
    print(f"\nSaved predictions to {output_path}")
    print("\nTop 5 highest-risk claims:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()