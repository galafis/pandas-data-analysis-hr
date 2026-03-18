"""People Analytics HR pipeline."""

__version__ = "1.0.0"

from src.data_loader import load_hr_data
from src.eda import HRExploratoryAnalysis
from src.attrition_model import AttritionModel

__all__ = [
    "load_hr_data",
    "HRExploratoryAnalysis",
    "AttritionModel",
    "__version__",
]
