import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


NUMERICAL_FEATURES = [
    "total_billed",
    "expected_payment",
    "num_procedures",
    "num_diagnoses",
    "days_to_submit",
    "payment_ratio",
]

BINARY_FEATURES = [
    "prior_auth_required",
    "has_prior_auth",
    "is_in_network",
    "missing_documentation_flag",
    "eligibility_verified",
    "referral_required",
    "referral_present",
    "prior_auth_vs_has_auth",        # derived data
    "ref_req_vs_ref_present",        # derived data
]

CATEGORICAL_FEATURES = [
    "payer_id",
    "payer_type",
    "visit_type",
]

ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_denied"  # 1 if claim denied, 0 if approved


def derive_features(df : pd.DataFrame) -> pd.DataFrame:
    """
    Derive new features from existing ones.
    """
    df = df.copy()

    # Derived feature: payment ratio
    df["payment_ratio"] = (df["expected_payment"] / df["total_billed"].replace(0, np.nan))
    df["payment_ratio"] = df["payment_ratio"].fillna(df["payment_ratio"].median())

    # Derived feature: prior_auth_vs_has_auth
    df["prior_auth_vs_has_auth"] = np.where(
        (df["prior_auth_required"] == 1) & (df["has_prior_auth"] == 1), 1, 0
    )

    # Derived feature: ref_req_vs_ref_present
    df["ref_req_vs_ref_present"] = np.where(
        (df["referral_required"] == 1) & (df["referral_present"] == 1), 1, 0
    )

    return df


def load_claims_data(file_path: str) -> pd.DataFrame:
    """
    Load claims data from a CSV file.
    """
    df = pd.read_csv(file_path)
    df = derive_features(df)
    return df

def split_data_for_training(df: pd.DataFrame):
    """
    Split the data into features and target variable.
    """
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)

    return train, test, val


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a preprocessing pipeline for the insurance claims data.
    """
    # Preprocessing for numerical features
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical features
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),  # Keep binary features as they are
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="passthrough"  # Keep binary features as they are
    )

    return preprocessor


def get_feature_names_from_preprocessor(preprocessor : ColumnTransformer) ->list:
    """
    Get the feature names after preprocessing.
    """
    feature_names = []

    # Get feature names for numerical features
    feature_names.extend(NUMERICAL_FEATURES)

    # Get feature names for binary features
    feature_names.extend(BINARY_FEATURES)

    # Get feature names for categorical features
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_feature_names)

    return feature_names
