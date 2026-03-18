"""Unit tests for src.data_loader."""

from __future__ import annotations

import pandas as pd

from src.data_loader import _generate_synthetic_hr, load_hr_data


class TestGenerateSyntheticHR:
    def test_returns_dataframe(self) -> None:
        df = _generate_synthetic_hr(n_employees=50)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self) -> None:
        df = _generate_synthetic_hr(n_employees=100)
        assert len(df) == 100

    def test_required_columns_present(self) -> None:
        df = _generate_synthetic_hr(n_employees=20)
        for col in ["Age", "Attrition", "Department", "MonthlyIncome", "Gender"]:
            assert col in df.columns

    def test_attrition_binary(self) -> None:
        df = _generate_synthetic_hr(n_employees=200)
        assert set(df["Attrition"].unique()).issubset({"Yes", "No"})

    def test_attrition_rate_realistic(self) -> None:
        df = _generate_synthetic_hr(n_employees=1470)
        rate = (df["Attrition"] == "Yes").mean()
        assert 0.05 <= rate <= 0.40, f"Unexpected attrition rate: {rate:.2%}"

    def test_monthly_income_positive(self) -> None:
        df = _generate_synthetic_hr(n_employees=50)
        assert (df["MonthlyIncome"] > 0).all()

    def test_reproducibility(self) -> None:
        df1 = _generate_synthetic_hr(n_employees=10, seed=99)
        df2 = _generate_synthetic_hr(n_employees=10, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        df1 = _generate_synthetic_hr(n_employees=100, seed=1)
        df2 = _generate_synthetic_hr(n_employees=100, seed=2)
        assert not df1["MonthlyIncome"].equals(df2["MonthlyIncome"])


class TestLoadHRData:
    def test_synthetic_source_returns_df(self) -> None:
        df = load_hr_data(source="synthetic", n_employees=30)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30

    def test_gender_values_valid(self) -> None:
        df = load_hr_data(source="synthetic", n_employees=50)
        assert set(df["Gender"].unique()).issubset({"Male", "Female"})
