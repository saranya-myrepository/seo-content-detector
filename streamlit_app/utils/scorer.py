from __future__ import annotations
import pandas as pd
import numpy as np
import os
import sys
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from typing import List, Dict, Any, Tuple

# --- Configuration ---
TEST_SIZE = 0.3
RANDOM_STATE = 42

# --- Model & File Paths (Must be passed as arguments) ---
# We use output_model_path for MODEL_PATH later
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')


# 1. Create synthetic labels (Low/Medium/High)
def create_quality_label(row):
    word_count = row['word_count']
    readability = row['flesch_reading_ease']
    
    # High: word_count > 1500 AND 50 <= readability <= 70
    if word_count > 1500 and (50 <= readability <= 70):
        return 'High'
    # Low: word_count < 500 OR readability < 30
    elif word_count < 500 or readability < 30:
        return 'Low'
    # Medium: all other cases
    else:
        return 'Medium'

# 2. Compare to baseline: Rule-based classifier using word count only
def baseline_predict(wc):
    if wc > 1500: return 'High'
    elif wc < 500: return 'Low'
    else: return 'Medium'

def run_quality_scoring_pipeline(input_features_csv: str, output_model_path: str):
    """
    Loads features via command line argument, runs the quality scoring pipeline, 
    and saves the model object.
    """
    
    print("\n--- Starting Phase 4: Content Quality Scoring ---")
    
    # --- FIX: Load features directly from the argument path ---
    try:
        # Load the features file using the path passed via argparse (e.g., data/features.csv)
        df = pd.read_csv(input_features_csv)
        print(f"Loaded {len(df)} records from {input_features_csv}.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load features.csv from {input_features_csv}. Error: {e}")
        # Re-raise the error for better traceback when running in terminal
        raise


    # Apply synthetic labeling
    df['quality_label'] = df.apply(create_quality_label, axis=1)
    print(f"Synthetic labels created based on assignment rules.")

    # Select Features: word_count, sentence_count, flesch_reading_ease
    feature_cols = ['word_count', 'sentence_count', 'flesch_reading_ease'] 
    
    # Robustness check for missing columns
    if not all(col in df.columns for col in feature_cols):
        print(f"FATAL ERROR: Required features {feature_cols} not found in CSV. Aborting.")
        sys.exit(1)
        
    X = df[feature_cols]
    y = df['quality_label']

    # Train/test split: 70/30 (Stratify removed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Data split: Training size={len(X_train)}, Testing size={len(X_test)}")

    # Train a classification model: Logistic Regression
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model, output_model_path)
    print(f"Trained model saved to: {output_model_path}")

    # Predict and Evaluate
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    
    # Compare to baseline
    y_baseline_pred = X_test['word_count'].apply(baseline_predict)
    baseline_accuracy = accuracy_score(y_test, y_baseline_pred)


    print(f"\n--- Model Performance vs Baseline ---")
    print(f"Model Accuracy: {model_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}") 


    # Report metrics
    print("\nClassification Report (Precision, Recall, F1-score):")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Report Top 2-3 features 
    feature_importance = pd.Series(np.mean(np.abs(model.coef_), axis=0), index=feature_cols).sort_values(ascending=False)

    print("\nTop Features (by average absolute coefficient value):")
    for rank, (feature, importance) in enumerate(feature_importance.head(3).items()):
        print(f"{rank+1}. {feature} (importance: {importance:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SEO Content Quality Scoring Pipeline (Phase 4)")
    ap.add_argument("input_features_csv", help="Path to features.csv (Input from Phase 2)")
    ap.add_argument("output_model_path", help="Path to save the trained quality_model.pkl")
    args = ap.parse_args()
    
    run_quality_scoring_pipeline(args.input_features_csv, args.output_model_path)