from __future__ import annotations
import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

# --- Configuration ---
THIN_CONTENT_THRESHOLD = 500  # Flag pages with word count < 500 [cite: 93]
SIMILARITY_THRESHOLD = 0.80   # Define similarity threshold (e.g., > 0.80 = duplicate) [cite: 85]

def analyze_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs Thin Content and Cosine Similarity Duplicate Detection.
    """
    
    # --- 1. Thin Content Detection ---
    # Create binary "is_thin" column [cite: 94]
    df['is_thin'] = (df['word_count'] < THIN_CONTENT_THRESHOLD).astype(int)
    thin_pages_count = df['is_thin'].sum()

    # --- 2. Duplicate Detection (Cosine Similarity) ---
    
    # Convert embedding column (stored as list of lists) into a NumPy matrix
    try:
        # Use a lambda function to parse JSON string lists into Python lists
        embedding_matrix = np.array(df['embedding'].apply(json.loads).tolist())
    except Exception as e:
        print(f"Error converting embeddings to matrix. Check format in features.csv: {e}")
        # Return empty data structure if conversion fails
        return pd.DataFrame(), {'total_pages': len(df), 'duplicate_pairs': 0, 'thin_content_pages': thin_pages_count}
    
    # Compute pairwise cosine similarity matrix [cite: 89]
    similarity_matrix = cosine_similarity(embedding_matrix)
    
    duplicate_pairs = []

    # Identify and list duplicate pairs [cite: 86]
    # Iterate through the upper triangle of the matrix (i+1) to find unique pairs
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            similarity = similarity_matrix[i, j]
            # Flag pairs above threshold [cite: 90]
            if similarity >= SIMILARITY_THRESHOLD: 
                duplicate_pairs.append({
                    'url1': df.iloc[i]['url'],
                    'url2': df.iloc[j]['url'],
                    'similarity': round(similarity, 4)
                })

    df_duplicates = pd.DataFrame(duplicate_pairs)
    total_pages = len(df)
    duplicate_count = len(df_duplicates)
    
    # Prepare Summary
    summary = {
        'total_pages': total_pages,
        'duplicate_pairs': duplicate_count,
        'thin_content_pages': thin_pages_count,
        'thin_percentage': round(thin_pages_count / total_pages * 100) if total_pages > 0 else 0
    }

    return df_duplicates, summary


def main(input_features_csv: str, output_duplicates_csv: str):
    """
    Main execution function for the duplicate detection pipeline.
    """
    if not os.path.exists(input_features_csv):
        print(f"FATAL ERROR: Input file not found at {input_features_csv}. Run Phase 2 first.")
        sys.exit(1)
        
    print(f"Loading features from: {input_features_csv}")
    df_features = pd.read_csv(input_features_csv)

    if df_features.empty:
        print("Input DataFrame is empty. Aborting duplicate detection.")
        sys.exit(0)
        
    # Ensure required columns exist
    if not all(col in df_features.columns for col in ['url', 'word_count', 'embedding']):
        raise ValueError("Input CSV must contain 'url', 'word_count', and 'embedding' columns.")

    # Run the analysis
    df_duplicates, summary = analyze_duplicates(df_features)

    # Report basic statistics and save results to CSV [cite: 87]
    os.makedirs(os.path.dirname(output_duplicates_csv), exist_ok=True)
    df_duplicates.to_csv(output_duplicates_csv, index=False)
    
    print(f"\n--- Phase 3: Duplicate Detection Complete ---")
    print(f"Output saved to: {output_duplicates_csv}")
    print(f"Total pages analyzed: {summary['total_pages']}") 
    print(f"Duplicate pairs: {summary['duplicate_pairs']}")
    print(f"Thin content pages: {summary['thin_content_pages']} ({summary['thin_percentage']}%)") 

    if not df_duplicates.empty:
        print("\nExample Duplicate Pairs Found:")
        print(df_duplicates.head().to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SEO Duplicate Detection Pipeline (Phase 3)")
    ap.add_argument("input_features_csv", help="Path to features.csv (Input from Phase 2)")
    ap.add_argument("output_duplicates_csv", help="Path to save duplicates.csv")
    args = ap.parse_args()
    
    main(args.input_features_csv, args.output_duplicates_csv)