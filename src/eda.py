"""
Exploratory Data Analysis (EDA) Module for HR Analytics.

Provides comprehensive statistical analysis and visualization
functions for People Analytics datasets.

Author: Gabriel Demetrios Lafis
License: MIT
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HRExploratoryAnalysis:
    """Performs exploratory data analysis on HR datasets.
    
    Attributes:
        df: The HR DataFrame to analyze.
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with an HR DataFrame.
        
        Args:
            df: A pandas DataFrame containing HR data.
        """
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        logger.info(
            "HRExploratoryAnalysis initialized with %d rows, %d cols",
            len(df),
            len(df.columns),
        )

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def summary_statistics(self) -> pd.DataFrame:
        """Return extended descriptive statistics for numeric columns.
        
        Returns:
            DataFrame with count, mean, std, min, quartiles, max,
            skewness and kurtosis for each numeric column.
        """
        stats = self.df[self.numeric_cols].describe().T
        stats["skew"] = self.df[self.numeric_cols].skew()
        stats["kurtosis"] = self.df[self.numeric_cols].kurtosis()
        stats["missing"] = self.df[self.numeric_cols].isnull().sum()
        stats["missing_pct"] = (
            self.df[self.numeric_cols].isnull().mean() * 100
        ).round(2)
        return stats

    def categorical_summary(self) -> Dict[str, pd.DataFrame]:
        """Return value counts and proportions for categorical columns.
        
        Returns:
            Dictionary mapping column names to DataFrames with
            count and percentage for each category.
        """
        result: Dict[str, pd.DataFrame] = {}
        for col in self.categorical_cols:
            counts = self.df[col].value_counts()
            pct = self.df[col].value_counts(normalize=True).mul(100).round(2)
            result[col] = pd.DataFrame({"count": counts, "pct": pct})
        return result

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def correlation_matrix(
        self, method: str = "pearson", threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Compute the correlation matrix for numeric columns.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall').
            threshold: If set, only return pairs above this absolute value.
            
        Returns:
            Correlation matrix as a DataFrame.
        """
        corr = self.df[self.numeric_cols].corr(method=method)
        if threshold is not None:
            mask = corr.abs() < threshold
            corr = corr.where(~mask, other=np.nan)
        return corr

    def top_correlations(
        self, n: int = 10, method: str = "pearson"
    ) -> pd.DataFrame:
        """Return the top-N strongest correlations (excluding self).
        
        Args:
            n: Number of top pairs to return.
            method: Correlation method.
            
        Returns:
            DataFrame with columns [feature_1, feature_2, correlation].
        """
        corr = self.df[self.numeric_cols].corr(method=method)
        # Unstack and remove duplicates / self-correlations
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs.columns = ["feature_1", "feature_2", "correlation"]
        pairs["abs_corr"] = pairs["correlation"].abs()
        return (
            pairs.sort_values("abs_corr", ascending=False)
            .head(n)
            .drop(columns="abs_corr")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Attrition-specific analysis
    # ------------------------------------------------------------------

    def attrition_rate_by(
        self, group_col: str, attrition_col: str = "Attrition"
    ) -> pd.DataFrame:
        """Calculate attrition rate grouped by a categorical column.
        
        Args:
            group_col: Column name to group by.
            attrition_col: Name of the attrition column (binary 0/1 or Yes/No).
            
        Returns:
            DataFrame with group, total, attrition_count, attrition_rate.
        """
        df = self.df.copy()
        if not pd.api.types.is_numeric_dtype(df[attrition_col]):
            df[attrition_col] = (df[attrition_col] == "Yes").astype(int)

        grouped = (
            df.groupby(group_col)[attrition_col]
            .agg(["count", "sum"])
            .rename(columns={"count": "total", "sum": "attrition_count"})
        )
        grouped["attrition_rate"] = (
            grouped["attrition_count"] / grouped["total"] * 100
        ).round(2)
        return grouped.sort_values("attrition_rate", ascending=False)

    # ------------------------------------------------------------------
    # Salary equity analysis
    # ------------------------------------------------------------------

    def salary_equity_analysis(
        self,
        salary_col: str = "MonthlyIncome",
        group_col: str = "Gender",
        control_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze salary equity across groups.
        
        Args:
            salary_col: Name of the salary column.
            group_col: Column to compare (e.g. Gender, Ethnicity).
            control_cols: Optional list of columns to control for
                         (e.g. JobLevel, Department).
                         
        Returns:
            Dictionary with overall and controlled gap statistics.
        """
        result: Dict[str, Any] = {}

        # Overall gap
        group_means = self.df.groupby(group_col)[salary_col].mean()
        overall_gap = (
            (group_means.max() - group_means.min()) / group_means.max() * 100
        )
        result["overall"] = {
            "group_means": group_means.to_dict(),
            "gap_pct": round(overall_gap, 2),
        }

        # Controlled gap (within each combination of control columns)
        if control_cols:
            controlled_gaps = []
            for name, grp in self.df.groupby(control_cols):
                sub_means = grp.groupby(group_col)[salary_col].mean()
                if len(sub_means) >= 2:
                    gap = (
                        (sub_means.max() - sub_means.min())
                        / sub_means.max()
                        * 100
                    )
                    controlled_gaps.append(
                        {"control_group": str(name), "gap_pct": round(gap, 2)}
                    )
            result["controlled"] = {
                "mean_gap_pct": round(
                    np.mean([g["gap_pct"] for g in controlled_gaps]), 2
                )
                if controlled_gaps
                else None,
                "details": controlled_gaps,
            }

        return result

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def detect_outliers_iqr(
        self, columns: Optional[List[str]] = None, factor: float = 1.5
    ) -> Dict[str, Dict[str, Any]]:
        """Detect outliers using the IQR method.
        
        Args:
            columns: Columns to check (defaults to all numeric).
            factor: IQR multiplier (default 1.5).
            
        Returns:
            Dict mapping column names to outlier info (bounds, count).
        """
        cols = columns or self.numeric_cols
        outliers: Dict[str, Dict[str, Any]] = {}
        for col in cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            outliers[col] = {
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_count": int(mask.sum()),
                "outlier_pct": round(mask.mean() * 100, 2),
            }
        return outliers

    # ------------------------------------------------------------------
    # Data quality report
    # ------------------------------------------------------------------

    def data_quality_report(self) -> pd.DataFrame:
        """Generate a data quality report for all columns.
        
        Returns:
            DataFrame with dtype, missing, unique, and sample values.
        """
        report = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "non_null": self.df.notnull().sum(),
                "null_count": self.df.isnull().sum(),
                "null_pct": (self.df.isnull().mean() * 100).round(2),
                "unique": self.df.nunique(),
                "sample": self.df.iloc[0] if len(self.df) > 0 else None,
            }
        )
        return report

    # ------------------------------------------------------------------
    # Generate full EDA report
    # ------------------------------------------------------------------

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive EDA report dictionary.
        
        Returns:
            Dictionary containing all EDA results.
        """
        logger.info("Generating full EDA report...")
        report: Dict[str, Any] = {
            "shape": {"rows": self.df.shape[0], "columns": self.df.shape[1]},
            "data_quality": self.data_quality_report().to_dict(),
            "summary_statistics": self.summary_statistics().to_dict(),
            "top_correlations": self.top_correlations().to_dict(),
            "outliers": self.detect_outliers_iqr(),
        }
        logger.info("EDA report generated successfully.")
        return report
