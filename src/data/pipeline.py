"""
src/data/pipeline.py
End-to-end NLP Data Pipeline for Project Intelligence System (IndoDiscourse dataset).
Includes:
- Data loading & raw sanitization
- Logical hierarchy enforcement (sublabels -> toxicity)
- Label conflict resolution & deduplication
- Dual-track cleaning (clean_text_bert & clean_text_tfidf)
- Stratified train/val/test splitting
- TF-IDF extraction & artifact exporting
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.text_preprocessing import clean_text_bert, clean_text_tfidf, normalize_slang

logger = logging.getLogger(__name__)

LABEL_COLUMNS = [
    "toxicity",
    "polarized",
    "profanity_obscenity",
    "threat_incitement_to_violence",
    "insults",
    "identity_attack",
    "sexually_explicit",
]

SUBLABEL_COLUMNS = [
    "profanity_obscenity",
    "threat_incitement_to_violence",
    "insults",
    "identity_attack",
    "sexually_explicit",
]


class TextDataPipeline:
    """
    End-to-end data pipeline for Indonesian Toxic Comment Classification:
    load -> harmonize_labels -> deduplicate -> clean_text (dual-track) -> split -> export.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw(self, filename: str = "indodiscourse_main_0000.parquet") -> pd.DataFrame:
        """Memuat dataset mentah dan membuang missing value pada kolom text."""
        filepath = self.raw_dir / filename
        if not filepath.exists():
            # Fallback to csv if parquet not found
            csv_path = self.raw_dir / filename.replace(".parquet", ".csv")
            if csv_path.exists():
                filepath = csv_path
            else:
                raise FileNotFoundError(f"Dataset tidak ditemukan di {filepath} atau {csv_path}")

        if filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        df = df.dropna(subset=["text"]).copy()
        logger.info(f"Loaded raw dataset: {len(df):,} baris valid.")
        return df

    def harmonize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Konsolidasi voting anotator menjadi biner (>= 0.5 -> 1).
        2. Penegakan hierarki logis (Hierarchy Enforcement): jika sublabel >= 1 maka toxicity = 1.
        """
        for col in LABEL_COLUMNS:
            if col in df.columns:
                if df[col].dtype == object and isinstance(df[col].iloc[0], (list, np.ndarray)):
                    df[col] = df[col].apply(lambda x: int(np.mean([int(v) for v in x]) >= 0.5))
                else:
                    df[col] = df[col].astype(int)

        if "related_to_election_2024" in df.columns:
            if df["related_to_election_2024"].dtype == object and isinstance(df["related_to_election_2024"].iloc[0], (list, np.ndarray)):
                df["related_to_election_2024"] = df["related_to_election_2024"].apply(lambda x: int(np.mean([int(v) for v in x]) >= 0.5))
        if "is_noise_or_spam_text" in df.columns:
            if df["is_noise_or_spam_text"].dtype == object and isinstance(df["is_noise_or_spam_text"].iloc[0], (list, np.ndarray)):
                df["is_noise_or_spam_text"] = df["is_noise_or_spam_text"].apply(lambda x: int(np.mean([int(v) for v in x]) >= 0.5))

        # Hierarchy enforcement
        sublabels_present = [c for c in SUBLABEL_COLUMNS if c in df.columns]
        if sublabels_present and "toxicity" in df.columns:
            df.loc[df[sublabels_present].sum(axis=1) > 0, "toxicity"] = 1

        logger.info("Harmonisasi label & hierarchy enforcement selesai.")
        return df

    def deduplicate_and_resolve_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Meresolusi duplikat teks berlabel kontradiktif melalui voting agregasi mayoritas
        serta membuang duplikat murni untuk mencegah data leakage.
        """
        target_and_meta = [c for c in LABEL_COLUMNS + ["related_to_election_2024", "is_noise_or_spam_text"] if c in df.columns]
        agg_rules = {col: lambda x: int(np.mean(x) >= 0.5) for col in target_and_meta}

        for meta_col in ["text_id", "topic", "initial_paragraph"]:
            if meta_col in df.columns:
                agg_rules[meta_col] = "first"

        df_dedup = df.groupby("text", as_index=False).agg(agg_rules)
        logger.info(f"Deduplikasi selesai: {len(df_dedup):,} teks unik.")
        return df_dedup

    def apply_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menambahkan kolom teks bersih untuk TF-IDF dan Deep Learning (IndoBERT)."""
        logger.info("Menjalankan pembersihan teks dual-track...")
        df["clean_text_bert"] = df["text"].apply(clean_text_bert)
        df["clean_text_tfidf"] = df["text"].apply(clean_text_tfidf)
        df["clean_text"] = df["clean_text_tfidf"]  # Backward compatibility

        # Filter jika ada teks yang kosong setelah pembersihan
        df = df[df["clean_text_tfidf"].str.strip() != ""].copy().reset_index(drop=True)
        return df

    def split_data(
        self, df: pd.DataFrame, target_col: str = "toxicity"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Pembagian berstrata 80% Train, 10% Validation, 10% Test."""
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[target_col] if target_col in df.columns else None,
        )

        relative_val_size = self.val_size / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_df,
            test_size=relative_val_size,
            random_state=self.random_state,
            stratify=train_df[target_col] if target_col in train_df.columns else None,
        )

        logger.info(f"Split data: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def fit_and_save_tfidf(self, train_df: pd.DataFrame) -> TfidfVectorizer:
        """Fit TF-IDF Vectorizer pada data train dan simpan sebagai artefak."""
        self.tfidf_vectorizer.fit(train_df["clean_text_tfidf"])
        vectorizer_path = self.processed_dir / "tfidf_vectorizer.joblib"
        joblib.dump(self.tfidf_vectorizer, vectorizer_path)
        logger.info(f"TF-IDF Vectorizer tersimpan di: {vectorizer_path}")
        return self.tfidf_vectorizer

    def export_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Menyimpan hasil partisi ke file parquet dan sample CSV."""
        train_df.to_parquet(self.processed_dir / "train.parquet", index=False)
        val_df.to_parquet(self.processed_dir / "val.parquet", index=False)
        test_df.to_parquet(self.processed_dir / "test.parquet", index=False)
        train_df.head(1000).to_csv(self.processed_dir / "train_sample_1000.csv", index=False)
        logger.info(f"Semua split data berhasil diekspor ke: {self.processed_dir}")

    def run(self, raw_filename: str = "indodiscourse_main_0000.parquet"):
        """Menjalankan full pipeline dari data mentah sampai artefak terproses."""
        df_raw = self.load_raw(raw_filename)
        df_harmonized = self.harmonize_labels(df_raw)
        df_dedup = self.deduplicate_and_resolve_conflicts(df_harmonized)
        df_clean = self.apply_cleaning(df_dedup)
        train_df, val_df, test_df = self.split_data(df_clean)
        self.fit_and_save_tfidf(train_df)
        self.export_splits(train_df, val_df, test_df)
        return train_df, val_df, test_df


# Alias untuk backward compatibility
DataPipeline = TextDataPipeline
