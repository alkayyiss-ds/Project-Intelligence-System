"""
src/data/text_preprocessing.py
Comprehensive Indonesian text preprocessing module for NLP & Deep Learning.
Supports:
- Slang & leetspeak normalization
- Dual-track cleaning: TF-IDF (classical ML) vs IndoBERT (Transformer)
- Negation-preserving stopword filtering
"""

import re
import string
from typing import Set, Dict, List, Optional
import nltk
from nltk.corpus import stopwords

# Download stopwords if not present
nltk.download("stopwords", quiet=True)

# ==============================================================================
# 1. KAMUS NORMALISASI SLANG, SINGKATAN, & OBFUSCATION (KATA DISAMARKAN)
# ==============================================================================
SLANG_DICT: Dict[str, str] = {
    # Kata Kasar / Disamarkan (Obfuscation / Leetspeak)
    "anj": "anjing", "anjg": "anjing", "ajg": "anjing", "anying": "anjing",
    "4nj1ng": "anjing", "4njing": "anjing", "anjir": "anjing", "njir": "anjing",
    "bgst": "bangsat", "b4ngs4t": "bangsat", "bsht": "bangsat",
    "kntl": "kontol", "k0nt0l": "kontol", "ktl": "kontol", "mmk": "memek",
    "bg": "bego", "bgo": "bego", "tll": "tolol", "t0l0l": "tolol",
    "gblk": "goblok", "goblk": "goblok", "g0bl0k": "goblok",
    "bct": "bacot", "bcot": "bacot", "bacottt": "bacot",
    
    # Kata Negasi Gaul (Diseragamkan ke kata negasi baku/standar)
    "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "gk": "tidak", "g": "tidak", "tdk": "tidak", "bkn": "bukan",
    "kaga": "tidak", "kagak": "tidak", "ndak": "tidak", "tak": "tidak",
    "blm": "belum", "lom": "belum", "jgn": "jangan",
    
    # Kata Gaul & Singkatan Percakapan Harian
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau", "tp": "tetapi",
    "bgt": "banget", "bener": "benar", "beneran": "benar",
    "dr": "dari", "pd": "pada", "krn": "karena", "coz": "karena",
    "drpd": "daripada", "gmn": "bagaimana", "gimana": "bagaimana",
    "knp": "kenapa", "napa": "kenapa", "dpt": "dapat", "sdh": "sudah",
    "udh": "sudah", "udah": "sudah", "blg": "bilang", "ngomong": "bicara",
    "km": "kamu", "lu": "kamu", "lo": "kamu", "elu": "kamu", "elo": "kamu",
    "gw": "saya", "gue": "saya", "aku": "saya", "ak": "saya",
    "org": "orang", "bnyk": "banyak", "byk": "banyak",
    "baper": "bawa perasaan", "caper": "cari perhatian", "pansos": "panjat sosial"
}

# ==============================================================================
# 2. DEFINISI STOPWORDS & PROTEKSI KATA NEGASI
# ==============================================================================
_indo_stopwords: Set[str] = set(stopwords.words("indonesian"))
_informal_fillers: Set[str] = {
    "ny", "d", "ya", "aja", "nih", "deh", "sih", "dong", "kan", "tuh", "nya", "amp",
    "rt", "si", "ku", "mu", "ini", "itu", "ada", "bisa", "jadi", "sama", "juga", "ke", "di", "dan", "atau"
}
NEGATION_TOKENS: Set[str] = {
    "tidak", "tak", "bukan", "jangan", "tanpa", "belum", "tiada"
}
SAFE_STOPWORDS_TFIDF: Set[str] = (_indo_stopwords.union(_informal_fillers)) - NEGATION_TOKENS


# ==============================================================================
# 3. FUNGSI TRANSFORMASI TEKS
# ==============================================================================
def normalize_slang(text: str) -> str:
    """Mengganti kata slang, singkatan, dan obfuscation kata kasar ke bentuk baku/terbaca."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    normalized = [SLANG_DICT.get(word, word) for word in words]
    return " ".join(normalized)


def clean_text_bert(text: str) -> str:
    """
    Pembersihan teks terstandarisasi khusus model Deep Learning / Transformer (IndoBERT).
    - Menghapus derau platform (URL, mention @user, hashtag #tag, entitas HTML, emoji).
    - Mempertahankan tanda baca kontekstual (!, ?, .), huruf kapital, dan stopwords utuh.
    - Menjaga keutuhan relasi semantik & tata bahasa.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # 2. Hapus Mention & Hashtag
    text = re.sub(r"[@#]\w+", " ", text)
    # 3. Hapus Entitas HTML
    text = re.sub(r"&\w+;", " ", text)
    # 4. Hapus Emoji & Simbol Non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")
    # 5. Normalisasi Huruf Berulang Berlebih ('bangeeeet' -> 'banget')
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # 6. Rapikan Spasi
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_tfidf(text: str) -> str:
    """
    Pembersihan teks agresif khusus model Klasikal / TF-IDF (SVM, Naive Bayes, Logistic Regression).
    - Lowercasing, normalisasi slang/obfuscation, penghapusan tanda baca & angka.
    - Stopword filtering dengan proteksi ketat kata negasi (tidak, bukan, jangan, dll.).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    # 2. Hapus URL, Mention, Hashtag, HTML
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    # 3. Hapus Emoji & Simbol Non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")
    # 4. Normalisasi Huruf Berulang
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    # 5. Normalisasi Slang & Kata Umpatan
    text = normalize_slang(text)
    # 6. Hapus Angka & Tanda Baca
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 7. Filter Stopwords (dengan proteksi kata negasi)
    tokens = [w for w in text.split() if w not in SAFE_STOPWORDS_TFIDF and len(w) > 2]
    return " ".join(tokens)
