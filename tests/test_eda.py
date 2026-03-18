"""
Unit tests for the HRExploratoryAnalysis module.

Author: Gabriel Demetrios Lafis
License: MIT
"""

import numpy as np
import pandas as pd
import pytest

from src.eda import HRExploratoryAnalysis


@pytest.fixture
def sample_hr_df() -> pd.DataFrame:
    """Create a small synthetic HR DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "EmployeeID": range(1, n + 1),
            "Age": np.random.randint(22, 60, n),
            "Gender": np.random.choice(["Male", "Female"], n),
            "Department": np.random.choice(
                ["Sales", "R&D", "HR"], n, p=[0.4, 0.4, 0.2]
            ),
            "MonthlyIncome": np.random.randint(3000, 20000, n),
            "JobLevel": np.random.randint(1, 6, n),
            "YearsAtCompany": np.random.randint(0, 30, n),
            "Attrition": np.random.choice(["Yes", "No"], n, p=[0.2, 0.8]),
            "OverTime": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "DistanceFromHome": np.random.randint(1, 30, n),
        }
    )


@pytest.fixture
def eda(sample_hr_df: pd.DataFrame) -> HRExploratoryAnalysis:
    """Return an HRExploratoryAnalysis instance."""
    return HRExploratoryAnalysis(sample_hr_df)


class TestHRExploratoryAnalysis:
    """Tests for HRExploratoryAnalysis class."""

    def test_init_sets_columns(self, eda: HRExploratoryAnalysis) -> None:
        assert len(eda.numeric_cols) > 0
        assert len(eda.categorical_cols) > 0
        assert "MonthlyIncome" in eda.numeric_cols
        assert "Gender" in eda.categorical_cols

    def test_summary_statistics_shape(self, eda: HRExploratoryAnalysis) -> None:
        stats = eda.summary_statistics()
        assert isinstance(stats, pd.DataFrame)
        assert "skew" in stats.columns
        assert "kurtosis" in stats.columns
        assert "missing" in stats.columns
        assert len(stats) == len(eda.numeric_cols)

    def test_categorical_summary(self, eda: HRExploratoryAnalysis) -> None:
        result = eda.categorical_summary()
        assert isinstance(result, dict)
        assert "Gender" in result
        assert "count" in result["Gender"].columns
        assert "pct" in result["Gender"].columns

    def test_correlation_matrix(self, eda: HRExploratoryAnalysis) -> None:
        corr = eda.correlation_matrix()
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1]
        # Diagonal should be 1.0
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10

    def test_correlation_matrix_with_threshold(
        self, eda: HRExploratoryAnalysis
    ) -> None:
        corr = eda.correlation_matrix(threshold=0.5)
        # Values below threshold should be NaN (except diagonal)
        assert isinstance(corr, pd.DataFrame)

    def test_top_correlations(self, eda: HRExploratoryAnalysis) -> None:
        top = eda.top_correlations(n=5)
        assert isinstance(top, pd.DataFrame)
        assert len(top) <= 5
        assert "feature_1" in top.columns
        assert "feature_2" in top.columns
        assert "correlation" in top.columns

    def test_attrition_rate_by(self, eda: HRExploratoryAnalysis) -> None:
        rates = eda.attrition_rate_by("Department")
        assert isinstance(rates, pd.DataFrame)
        assert "attrition_rate" in rates.columns
        assert "total" in rates.columns
        assert all(0 <= r <= 100 for r in rates["attrition_rate"])

    def test_attrition_rate_by_overtime(
        self, eda: HRExploratoryAnalysis
    ) -> None:
        rates = eda.attrition_rate_by("OverTime")
        assert len(rates) == 2  # Yes and No

    def test_salary_equity_analysis(
        self, eda: HRExploratoryAnalysis
    ) -> None:
        result = eda.salary_equity_analysis()
        assert "overall" in result
        assert "group_means" in result["overall"]
        assert "gap_pct" in result["overall"]
        assert isinstance(result["overall"]["gap_pct"], float)

    def test_salary_equity_with_controls(
        self, eda: HRExploratoryAnalysis
    ) -> None:
        result = eda.salary_equity_analysis(
            control_cols=["Department", "JobLevel"]
        )
        assert "controlled" in result
        assert "mean_gap_pct" in result["controlled"]

    def test_detect_outliers_iqr(self, eda: HRExploratoryAnalysis) -> None:
        outliers = eda.detect_outliers_iqr()
        assert isinstance(outliers, dict)
        assert len(outliers) == len(eda.numeric_cols)
        for col, info in outliers.items():
            assert "lower_bound" in info
            assert "upper_bound" in info
            assert "outlier_count" in info
            assert info["outlier_count"] >= 0

    def test_detect_outliers_specific_columns(
        self, eda: HRExploratoryAnalysis
    ) -> None:
        outliers = eda.detect_outliers_iqr(columns=["MonthlyIncome"])
        assert len(outliers) == 1
        assert "MonthlyIncome" in outliers

    def test_data_quality_report(self, eda: HRExploratoryAnalysis) -> None:
        report = eda.data_quality_report()
        assert isinstance(report, pd.DataFrame)
        assert "dtype" in report.columns
        assert "null_count" in report.columns
        assert "unique" in report.columns
        assert len(report) == len(eda.df.columns)

    def test_generate_report(self, eda: HRExploratoryAnalysis) -> None:
        report = eda.generate_report()
        assert isinstance(report, dict)
        assert "shape" in report
        assert "data_quality" in report
        assert "summary_statistics" in report
        assert "top_correlations" in report
        assert "outliers" in report
        assert report["shape"]["rows"] == 100

    def test_immutability(
        self, sample_hr_df: pd.DataFrame, eda: HRExploratoryAnalysis
    ) -> None:
        """Ensure the original DataFrame is not modified."""
        original_len = len(sample_hr_df)
        _ = eda.generate_report()
        assert len(sample_hr_df) == original_len
