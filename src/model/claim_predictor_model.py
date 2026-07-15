""""
Uses a logistic regression model to predict the status of an insurance claim based on the provided features.
The model is built using a scikit-learn pipeline that includes preprocessing steps and the classifier.
The logistic regression is configured to handle class imbalance by using balanced class weights.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from data.preprocess import build_preprocessing_pipeline


def build_model_pipeline(seed: int = 42, pos_weight: float = 1.0) -> Pipeline:
    preprocessor = build_preprocessing_pipeline()

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handles the ~1:3.6 imbalance directly
        random_state=seed,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return pipeline
