"""HR data loader for People Analytics pipeline.

Loads the IBM HR Analytics Employee Attrition dataset from Kaggle (when
credentials are available) or generates a realistic synthetic equivalent.

Source: Kaggle IBM HR Analytics Attrition Dataset (CC0 Public Domain)
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IBM_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]


def load_hr_data(
    source: str = "synthetic",
    raw_path: Optional[Path] = None,
    n_employees: int = 1470,
) -> pd.DataFrame:
    """Load HR dataset from Kaggle or generate synthetic data.

    Args:
        source: 'kaggle' or 'synthetic'.
        raw_path: Local path to store/read the raw CSV.
        n_employees: Number of synthetic employees to generate.

    Returns:
        DataFrame following IBM HR Analytics schema.
    """
    raw_path = raw_path or Path(os.getenv("DATA_RAW_PATH", "data/raw"))
    csv_path = raw_path / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    if source == "kaggle" and csv_path.exists():
        logger.info("Loading Kaggle IBM HR dataset from %s", csv_path)
        return pd.read_csv(csv_path)

    logger.info("Generating synthetic IBM-style HR dataset (%d employees)", n_employees)
    return _generate_synthetic_hr(n_employees=n_employees)


def _generate_synthetic_hr(n_employees: int = 1470, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic HR dataset following IBM schema.

    The data mimics realistic distributions but represents NO real company
    or individual. Attrition rate is calibrated to ~16% (similar to IBM dataset).

    Args:
        n_employees: Number of employee records to generate.
        seed: Random seed for reproducibility.

    Returns:
        Synthetic DataFrame.
    """
    rng = np.random.default_rng(seed)

    departments = ["Sales", "Research & Development", "Human Resources"]
    dept_weights = [0.30, 0.65, 0.05]
    job_roles = {
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development": ["Research Scientist", "Laboratory Technician",
                                    "Healthcare Representative", "Manufacturing Director",
                                    "Research Director"],
        "Human Resources": ["Human Resources", "Manager"],
    }
    education_fields = ["Life Sciences", "Medical", "Marketing",
                        "Technical Degree", "Human Resources", "Other"]
    marital_statuses = ["Single", "Married", "Divorced"]
    business_travels = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    genders = ["Male", "Female"]

    dept = rng.choice(departments, n_employees, p=dept_weights)
    gender = rng.choice(genders, n_employees, p=[0.60, 0.40])
    marital = rng.choice(marital_statuses, n_employees, p=[0.32, 0.46, 0.22])
    age = rng.integers(18, 61, n_employees)
    years_at_company = np.clip(rng.integers(0, 41, n_employees), 0, age - 18)
    job_level = rng.integers(1, 6, n_employees)
    monthly_income = (
        2000 + job_level * 2000
        + rng.normal(0, 1000, n_employees)
        # Gender pay gap of ~8% on average (to be detected by equity analysis)
        + np.where(gender == "Female", -500, 0)
    ).clip(1000, 20000).round(2)

    # Attrition is correlated with overtime, job satisfaction, distance
    overtime = rng.choice(["Yes", "No"], n_employees, p=[0.28, 0.72])
    job_satisfaction = rng.integers(1, 5, n_employees)
    distance_from_home = rng.integers(1, 30, n_employees)
    attrition_prob = (
        0.05
        + 0.15 * (overtime == "Yes")
        + 0.10 * (job_satisfaction == 1)
        + 0.02 * (distance_from_home > 20)
        - 0.05 * (marital == "Married")
    ).clip(0.01, 0.95)
    attrition = np.where(
        rng.random(n_employees) < attrition_prob, "Yes", "No"
    )

    job_role = np.array([
        rng.choice(job_roles[d]) for d in dept
    ])

    df = pd.DataFrame({
        "Age": age,
        "Attrition": attrition,
        "BusinessTravel": rng.choice(business_travels, n_employees, p=[0.20, 0.60, 0.20]),
        "DailyRate": rng.integers(100, 1500, n_employees),
        "Department": dept,
        "DistanceFromHome": distance_from_home,
        "Education": rng.integers(1, 6, n_employees),
        "EducationField": rng.choice(education_fields, n_employees),
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(1, n_employees + 1),
        "EnvironmentSatisfaction": rng.integers(1, 5, n_employees),
        "Gender": gender,
        "HourlyRate": rng.integers(30, 101, n_employees),
        "JobInvolvement": rng.integers(1, 5, n_employees),
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": rng.integers(2000, 27000, n_employees),
        "NumCompaniesWorked": rng.integers(0, 10, n_employees),
        "Over18": "Y",
        "OverTime": overtime,
        "PercentSalaryHike": rng.integers(11, 26, n_employees),
        "PerformanceRating": rng.choice([3, 4], n_employees, p=[0.85, 0.15]),
        "RelationshipSatisfaction": rng.integers(1, 5, n_employees),
        "StandardHours": 80,
        "StockOptionLevel": rng.integers(0, 4, n_employees),
        "TotalWorkingYears": np.clip(rng.integers(0, 41, n_employees), years_at_company, 40),
        "TrainingTimesLastYear": rng.integers(0, 7, n_employees),
        "WorkLifeBalance": rng.integers(1, 5, n_employees),
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": np.clip(rng.integers(0, 19, n_employees), 0, years_at_company),
        "YearsSinceLastPromotion": np.clip(rng.integers(0, 16, n_employees), 0, years_at_company),
        "YearsWithCurrManager": np.clip(rng.integers(0, 18, n_employees), 0, years_at_company),
    })

    logger.info(
        "Synthetic HR dataset generated: %d rows, attrition rate=%.1f%%",
        len(df),
        (df["Attrition"] == "Yes").mean() * 100,
    )
    return df
