"""Employee attrition prediction model.

Builds a binary classifier (RandomForest / XGBoost) to predict employee
churn. Handles class imbalance via SMOTE from imbalanced-learn.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    ImbPipeline = Pipeline  # type: ignore

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]
NUMERIC_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]
TARGET = "Attrition"


class AttritionModel:
    """Binary classifier for employee attrition prediction."""

    def __init__(self, algorithm: str = "random_forest", **kwargs: Any) -> None:
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.pipeline: Optional[Any] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                if col not in self.label_encoders:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))
        return df

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET,
    ) -> AttritionModel:
        """Fit the attrition model.

        Args:
            df: HR DataFrame with attrition labels.
            target_col: Binary target column ('Yes'/'No').

        Returns:
            Self.
        """
        df_enc = self._encode(df)
        feature_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df_enc.columns]
        self.feature_names = feature_cols

        X = df_enc[feature_cols].values
        y = (df[target_col] == "Yes").astype(int).values

        estimator = RandomForestClassifier(
            n_estimators=self.kwargs.get("n_estimators", 200),
            max_depth=self.kwargs.get("max_depth", None),
            random_state=self.kwargs.get("random_state", 42),
            class_weight="balanced",
            n_jobs=-1,
        )

        if HAS_IMBLEARN:
            self.pipeline = ImbPipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ])
        else:
            logger.warning("imbalanced-learn not installed. Training without SMOTE.")
            self.pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ])

        self.pipeline.fit(X, y)
        logger.info("AttritionModel trained on %d samples.", len(X))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return attrition probability for each employee."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        df_enc = self._encode(df)
        X = df_enc[self.feature_names].values
        return self.pipeline.predict_proba(X)[:, 1]

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET,
    ) -> dict[str, Any]:
        """Compute classification metrics on provided data."""
        y_true = (df[target_col] == "Yes").astype(int).values
        y_prob = self.predict_proba(df)
        y_pred = (y_prob >= 0.5).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True)
        auc = roc_auc_score(y_true, y_prob)

        logger.info("ROC-AUC: %.4f", auc)
        return {"roc_auc": float(auc), "classification_report": report}

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        importances = self.pipeline.named_steps["model"].feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or Path(os.getenv("MODEL_ARTIFACT_PATH", "models/artifacts")) / "attrition_model.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> AttritionModel:
        with open(path, "rb") as fh:
            return pickle.load(fh)  # type: ignore[return-value]
