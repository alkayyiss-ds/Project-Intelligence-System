"""
Script untuk mengunduh dataset IndoDiscourse dari HuggingFace.
Menggunakan direct Parquet download — tidak memerlukan library 'datasets'.

Dataset: https://huggingface.co/datasets/Exqrch/IndoDiscourse
Kolom main: text_id, annotators_id, text, initial_paragraph, topic,
            is_noise_or_spam_text, related_to_election_2024,
            toxicity, polarized, profanity_obscenity,
            threat_incitement_to_violence, insults,
            identity_attack, sexually_explicit
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HF_BASE   = "https://huggingface.co/datasets/Exqrch/IndoDiscourse/resolve/refs%2Fconvert%2Fparquet"
HEADERS   = {"User-Agent": "Mozilla/5.0"}

# Parquet file URLs untuk setiap config
PARQUET_URLS = {
    "main": [
        f"{HF_BASE}/main/main/0000.parquet",
    ],
    "annotator": [
        f"{HF_BASE}/annotator/annotator/0000.parquet",
    ],
}

LABEL_COLS = [
    "toxicity", "polarized", "profanity_obscenity",
    "threat_incitement_to_violence", "insults",
    "identity_attack", "sexually_explicit",
]

# ── Helper ────────────────────────────────────────────────────────────────────
def download_parquet(url: str, dest: Path, retries: int = 3) -> bool:
    """Download file parquet dengan retry."""
    for attempt in range(1, retries + 1):
        try:
            print(f"   Unduh ({attempt}/{retries}): {url.split('/')[-1]} ...")
            resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r   Progress: {pct:5.1f}%  ({downloaded:,}/{total:,} bytes)", end="")
            print(f"\r   ✅ Selesai: {dest.name} ({downloaded:,} bytes)          ")
            return True
        except Exception as e:
            print(f"\n   ⚠️  Gagal attempt {attempt}: {e}")
            if attempt < retries:
                wait = 5 * attempt
                print(f"   Tunggu {wait}s sebelum retry...")
                time.sleep(wait)
    return False


def load_parquets(paths: list[Path]) -> pd.DataFrame:
    """Gabungkan beberapa file parquet menjadi satu DataFrame."""
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def download_and_save():
    print("=" * 60)
    print("📥 DOWNLOAD DATASET IndoDiscourse — HuggingFace")
    print("=" * 60)

    all_meta = {}

    for config, urls in PARQUET_URLS.items():
        print(f"\n🗂️  Config: '{config}'")
        parquet_files = []

        for i, url in enumerate(urls):
            dest = RAW_DIR / f"indodiscourse_{config}_{i:04d}.parquet"
            ok = download_parquet(url, dest)
            if ok:
                parquet_files.append(dest)
            else:
                print(f"   ❌ Gagal download {url}")

        if not parquet_files:
            print(f"   ❌ Tidak ada file berhasil didownload untuk config '{config}'")
            continue

        # Load & gabungkan
        df = load_parquets(parquet_files)
        print(f"   📊 Total baris: {len(df):,}")
        print(f"   📋 Kolom ({len(df.columns)}): {list(df.columns)}")

        # Simpan CSV
        csv_path = RAW_DIR / f"indodiscourse_{config}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"   💾 CSV   → {csv_path.name}")

        # Simpan JSON
        json_path = RAW_DIR / f"indodiscourse_{config}.json"
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"   💾 JSON  → {json_path.name}")

        all_meta[config] = {
            "rows": len(df),
            "columns": list(df.columns),
            "files": {
                "parquet": [str(p) for p in parquet_files],
                "csv": str(csv_path),
                "json": str(json_path),
            }
        }

        # Statistik khusus untuk config 'main'
        if config == "main":
            print("\n   📈 Distribusi Topik:")
            if "topic" in df.columns:
                for topic, cnt in df["topic"].value_counts().items():
                    print(f"      {topic:<30}: {cnt:,}")

            print("\n   📈 Label Stats (proporsi annotasi 'yes'):")
            for col in LABEL_COLS:
                if col not in df.columns:
                    continue
                try:
                    all_labels = []
                    for cell in df[col].dropna():
                        items = cell if isinstance(cell, list) else [cell]
                        all_labels.extend(str(x).lower() for x in items)
                    total = len(all_labels)
                    yes   = sum(1 for l in all_labels if l in ("yes", "1", "true"))
                    pct   = yes / total * 100 if total else 0
                    print(f"      {col:<38}: {pct:5.1f}%  ({yes:,}/{total:,})")
                except Exception:
                    pass

    # Simpan metadata
    meta = {
        "dataset": "Exqrch/IndoDiscourse",
        "source": "https://huggingface.co/datasets/Exqrch/IndoDiscourse",
        "license": "Apache-2.0",
        "task": "multi-label text classification",
        "language": "Indonesian",
        "label_columns": LABEL_COLS,
        "description": (
            "Multi-labeled dataset for Indonesian discourse analysis. "
            "Covers toxicity, polarization, profanity, threats, insults, "
            "identity attacks, and sexually explicit content. "
            "Each text is annotated by multiple annotators with demographic info."
        ),
        "configs": all_meta,
    }

    meta_path = RAW_DIR / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ SELESAI! File tersimpan di:")
    print(f"   {RAW_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    download_and_save()
