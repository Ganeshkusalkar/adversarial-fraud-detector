from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseDataLoader(ABC):
    """
    Abstract Base Class enforcing standard processing interfaces
    across all disparate datasets (IEEE-CIS, PaySim, Elliptic).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load_raw(self) -> Any:
        """Reads raw source data files from disk."""
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> pd.DataFrame:
        """Applies sanitization, imputation, and transformation logic."""
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes targeted domain-specific feature engineering."""
        pass
