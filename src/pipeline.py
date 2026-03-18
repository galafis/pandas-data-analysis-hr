"""
End-to-end HR Analytics Pipeline.

Orchestrates the full workflow: data loading, EDA, modeling,
and report generation.

Author: Gabriel Demetrios Lafis
License: MIT
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data_loader import load_hr_data
from src.eda import HRExploratoryAnalysis
from src.attrition_model import AttritionModel

logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the pipeline.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ensured: %s", d)


def run_pipeline(
    n_employees: int = 500,
    random_state: int = 42,
    output_format: str = "json",
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Execute the full HR analytics pipeline.

    Steps:
        1. Load / generate HR data
        2. Run exploratory data analysis
        3. Train attrition prediction model
        4. Generate and save reports

    Args:
        n_employees: Number of synthetic employees to generate.
        random_state: Random seed for reproducibility.
        output_format: Report output format ('json' or 'csv').
        save_artifacts: Whether to persist artifacts to disk.

    Returns:
        Dictionary with pipeline results and metrics.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info("HR Analytics Pipeline - Run %s", run_id)
    logger.info("=" * 60)

    ensure_directories()
    results: Dict[str, Any] = {"run_id": run_id, "status": "started"}

    # ------------------------------------------------------------------
    # Step 1: Data Loading
    # ------------------------------------------------------------------
    logger.info("[1/4] Loading HR data (n=%d)...", n_employees)
    df = load_hr_data(n_employees=n_employees)
    results["data"] = {"rows": len(df), "columns": len(df.columns)}
    logger.info("  Loaded %d rows x %d columns", len(df), len(df.columns))

    if save_artifacts:
        raw_path = DATA_RAW / f"hr_data_{run_id}.csv"
        df.to_csv(raw_path, index=False)
        logger.info("  Raw data saved to %s", raw_path)

    # ------------------------------------------------------------------
    # Step 2: Exploratory Data Analysis
    # ------------------------------------------------------------------
    logger.info("[2/4] Running exploratory data analysis...")
    eda = HRExploratoryAnalysis(df)
    eda_report = eda.generate_report()
    results["eda"] = {
        "shape": eda_report["shape"],
        "top_correlations": eda_report["top_correlations"],
    }

    # Salary equity
    try:
        equity = eda.salary_equity_analysis(
            salary_col="MonthlyIncome",
            group_col="Gender",
            control_cols=["Department", "JobLevel"],
        )
        results["salary_equity"] = equity.get("overall", {})
        logger.info("  Salary equity gap: %.2f%%", equity["overall"]["gap_pct"])
    except Exception as exc:
        logger.warning("  Salary equity analysis skipped: %s", exc)

    # Attrition rates
    try:
        for col in ["Department", "JobLevel", "OverTime"]:
            if col in df.columns:
                rates = eda.attrition_rate_by(col)
                results.setdefault("attrition_rates", {})[col] = (
                    rates["attrition_rate"].to_dict()
                )
        logger.info("  Attrition rates computed.")
    except Exception as exc:
        logger.warning("  Attrition rate analysis skipped: %s", exc)

    if save_artifacts:
        eda_path = REPORTS_DIR / f"eda_report_{run_id}.json"
        with open(eda_path, "w") as f:
            json.dump(
                {"shape": eda_report["shape"], "outliers": eda_report["outliers"]},
                f,
                indent=2,
                default=str,
            )
        logger.info("  EDA report saved to %s", eda_path)

    # ------------------------------------------------------------------
    # Step 3: Attrition Modeling
    # ------------------------------------------------------------------
    logger.info("[3/4] Training attrition model...")
    model = AttritionModel(random_state=random_state)
    model.train(df)
    metrics = model.evaluate(df)
    results["model"] = {
        "roc_auc": metrics.get("roc_auc"),
        "classification_report": metrics.get("classification_report"),
    }
    logger.info(
        "  Model ROC-AUC: %.4f",
        metrics.get("roc_auc", 0),
    )

    # Feature importance
    try:
        fi = model.feature_importance()
        results["feature_importance"] = fi.head(10).to_dict(orient="records")
    except Exception as exc:
        logger.warning("  Feature importance skipped: %s", exc)

    if save_artifacts:
        model_path = MODELS_DIR / f"attrition_model_{run_id}.pkl"
        model.save(model_path)
        logger.info("  Model saved to %s", model_path)

    # ------------------------------------------------------------------
    # Step 4: Final Report
    # ------------------------------------------------------------------
    logger.info("[4/4] Generating final report...")
    results["status"] = "completed"
    results["completed_at"] = datetime.now().isoformat()

    if save_artifacts:
        report_path = REPORTS_DIR / f"pipeline_report_{run_id}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("  Final report saved to %s", report_path)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)

    return results


def main() -> None:
    """Entry point for CLI execution."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    n_employees = int(os.getenv("N_EMPLOYEES", "500"))
    setup_logging(log_level)

    try:
        results = run_pipeline(n_employees=n_employees)
        print(json.dumps(results, indent=2, default=str))
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
