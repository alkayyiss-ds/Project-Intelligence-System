"""
src/data/pipeline.py
Data loading and preprocessing pipeline for Project Intelligence System.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data pipeline: load → clean → preprocess → split.

    Usage:
        pipeline = DataPipeline(data_dir="data/processed")
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run("dataset.csv")
    """

    def __init__(
        self,
        data_dir: str = "data",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}

    def load(self, filename: str, target_col: Optional[str] = None) -> pd.DataFrame:
        """Load CSV from data directory."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} cols")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: drop duplicates, handle nulls."""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            logger.info(f"Removed {before - after} duplicate rows")

        # Fill numeric nulls with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

        return df

    def encode_categoricals(self, df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
        """Label-encode all object columns."""
        exclude = exclude or []
        for col in df.select_dtypes(include="object").columns:
            if col not in exclude:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split into train / val / test sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y if y.nunique() < 20 else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.val_size, random_state=self.random_state
        )

        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit StandardScaler on train, transform all splits."""
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)
        return X_train_s, X_val_s, X_test_s

    def run(self, filename: str, target_col: str):
        """Full pipeline: load → clean → encode → split → scale."""
        df = self.load(filename)
        df = self.clean(df)
        df = self.encode_categoricals(df, exclude=[target_col])
        X_train, X_val, X_test, y_train, y_val, y_test = self.split(df, target_col)
        X_train, X_val, X_test = self.scale(X_train, X_val, X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test
