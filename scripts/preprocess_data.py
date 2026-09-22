"""
scripts/preprocess_data.py
CLI Script untuk menjalankan pipeline preprocessing teks IndoDiscourse secara otomatis.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.pipeline import TextDataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    logging.info("Memulai eksekusi TextDataPipeline...")
    pipeline = TextDataPipeline(
        raw_dir="data/raw",
        processed_dir="data/processed",
        test_size=0.1,
        val_size=0.1,
        random_state=42,
    )
    train_df, val_df, test_df = pipeline.run("indodiscourse_main_0000.parquet")
    logging.info("Pipeline Preprocessing selesai dengan sukses!")
    logging.info(f"Summary: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
