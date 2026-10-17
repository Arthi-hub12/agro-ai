"""
Train the baseline Chennai pest-risk model (standard training, no adaptations).

Usage:
    python train_chennai_model.py --data data/AgroGuard_AI_Cleaned.csv
    python train_chennai_model.py --data data/AgroGuard_AI_Cleaned.csv --out models/chennai_risk_model.pkl
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

FEATURES = ["temperature_c", "humidity_pct", "rainfall_mm", "wind_speed_ms"]
TARGET = "pest_risk"
# The CSV stores labels as 0/1/2 (assumed Low/Medium/High, in that order).
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
LABELS = ["Low", "Medium", "High"]
# Deliberately NOT used as features:
#   solar_radiation  - not in the original 4-feature setup (easy to add to FEATURES)
#   anomaly_label    - likely derived from the target (leakage risk)
#   year/month/day/dayofweek - calendar columns, not weather evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the Chennai CSV")
    parser.add_argument("--out", default="models/chennai_risk_model.pkl", help="Where to save the model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load and sort by date (needed for a chronological split)
    df = pd.read_csv(args.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing columns: {missing_cols}")

    # Convert numeric labels (0/1/2) to names; leave string labels untouched
    if pd.api.types.is_numeric_dtype(df[TARGET]):
        unknown = set(df[TARGET].unique()) - set(LABEL_MAP)
        if unknown:
            raise ValueError(f"Unexpected pest_risk values: {unknown}")
        df[TARGET] = df[TARGET].map(LABEL_MAP)

    print(f"Rows: {len(df)}  ({df['date'].min().date()} to {df['date'].max().date()})")
    print("Class counts (full data):")
    print(df[TARGET].value_counts().to_string(), "\n")

    # 2. Chronological split: first 80% train, last 20% test (no shuffling)
    split = int(len(df) * (1 - args.test_size))
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print(f"Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    print("Class counts (test):")
    print(y_test.value_counts().to_string(), "\n")

    # 3. Train the Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=args.seed)
    model.fit(X_train, y_train)

    # 4. Evaluate, with a majority-class baseline for context
    pred = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_pred = baseline.predict(X_test)

    print(f"Model accuracy:    {accuracy_score(y_test, pred):.3f}")
    print(f"Baseline accuracy: {accuracy_score(y_test, base_pred):.3f}  (always predict '{base_pred[0]}')")
    print(f"Model macro-F1:    {f1_score(y_test, pred, average='macro', zero_division=0):.3f}")
    print(f"Baseline macro-F1: {f1_score(y_test, base_pred, average='macro', zero_division=0):.3f}\n")

    present = [l for l in LABELS if l in set(y_test) | set(pred)]
    print("Classification report:")
    print(classification_report(y_test, pred, labels=present, zero_division=0))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, pred, labels=present), index=present, columns=present), "\n")

    print("Feature importances (impurity-based, rough guide only):")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:15s} {imp:.3f}")

    # 5. Save the model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
