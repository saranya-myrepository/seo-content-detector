from __future__ import annotations
import argparse
import csv
import json
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Optional: use sentence-transformers if available, else fallback to TF-IDF+PCA
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# ----------------------------
# Basic text cleaning
# ----------------------------
_WS = re.compile(r"\s+", re.MULTILINE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def clean_text(text: str) -> str:
    """Lowercase and remove extra whitespace."""
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", text.lower()).strip()

def count_sentences(text: str) -> int:
    """Simple sentence counter."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len([s for s in _SENT_SPLIT.split(text) if s.strip()])


# ----------------------------
# TF-IDF keyword extraction
# ----------------------------
def extract_top5_keywords(docs: List[str]) -> Tuple[List[str], TfidfVectorizer]:
    """Return top-5 TF-IDF keywords per document."""
    tfv = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                          stop_words="english", min_df=2)
    X = tfv.fit_transform(docs)
    vocab = np.array(tfv.get_feature_names_out())
    keywords_out = []

    for i in range(X.shape[0]):
        row = X[i]
        if row.nnz == 0:
            keywords_out.append("")
            continue
        idx = row.indices
        vals = row.data
        top_idx = idx[np.argsort(-vals)][:5]
        top_terms = vocab[top_idx]
        keywords_out.append("|".join(top_terms))
    return keywords_out, tfv


# ----------------------------
# Embeddings
# ----------------------------
def get_embeddings(texts: List[str], dim: int = 50) -> List[str]:
    """SentenceTransformer or TF-IDF+PCA fallback. Returns JSON-stringified vectors."""
    if _HAS_ST:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            V = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            if V.shape[1] > dim:
                V = PCA(n_components=dim, random_state=42).fit_transform(V)
            return [json.dumps(v.tolist()) for v in V]
        except Exception:
            pass

    # Fallback
    tfv = TfidfVectorizer(max_features=5000, stop_words="english")
    M = tfv.fit_transform(texts).toarray()
    if M.shape[1] > dim:
        M = PCA(n_components=dim, random_state=42).fit_transform(M)
    return [json.dumps(v.tolist()) for v in M]


# ----------------------------
# Main
# ----------------------------
def main(in_csv: str, out_csv: str):
    df = pd.read_csv(in_csv)
    if "url" not in df.columns or "body_text" not in df.columns:
        raise ValueError("Input must have 'url' and 'body_text' columns")

    tqdm.pandas(desc="Processing")

    # Clean text
    df["clean_text"] = df["body_text"].fillna("").progress_apply(clean_text)

    # Word count
    if "word_count" in df.columns:
        df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0).astype(int)
        mask = df["word_count"] == 0
        df.loc[mask, "word_count"] = df.loc[mask, "clean_text"].apply(lambda t: len(t.split()))
    else:
        df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))

    # Sentence count
    df["sentence_count"] = df["body_text"].fillna("").apply(count_sentences)

    # Flesch Reading Ease
    df["flesch_reading_ease"] = df["body_text"].fillna("").apply(
        lambda x: round(float(textstat.flesch_reading_ease(x or "")), 2)
    )

    # Top keywords
    topk, _ = extract_top5_keywords(df["clean_text"].tolist())
    df["top_keywords"] = topk

    # Embeddings (JSON strings)
    df["embedding"] = get_embeddings(df["clean_text"].tolist())

    # Format & export
    final = df[[
        "url",
        "word_count",
        "sentence_count",
        "flesch_reading_ease",
        "top_keywords",
        "embedding",
    ]].copy()

    final.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Features saved to {out_csv} ({len(final)} rows). Example:")
    print(final.head(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SEO Feature Extraction (Assignment Format)")
    ap.add_argument("in_csv", help="Path to extracted_content.csv")
    ap.add_argument("out_csv", help="Path to save features.csv")
    args = ap.parse_args()
    main(args.in_csv, args.out_csv)