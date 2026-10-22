"""
Inference & Prediction Script for Credit Card Fraud Detection.
Uses trained Random Forest model with Tomek Links + BCBSMOTE preprocessing.

Usage:
------
# Predict on a single transaction:
python predict.py --step 1 --type TRANSFER --amount 181.0 --oldbalanceOrg 181.0 --oldbalanceDest 0.0 --newbalanceDest 0.0

# Batch inference on CSV:
python predict.py --input_csv data/sample_transactions.csv --output_csv results/predictions.csv
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

from src.model import FraudDetectionModel
from src.preprocessing import FEATURE_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Credit Card Fraud for Transactions")
    parser.add_argument("--model_path", type=str, default="models/fraud_detection_model.joblib",
                        help="Path to saved model checkpoint.")
    parser.add_argument("--encoder_path", type=str, default="models/label_encoder.joblib",
                        help="Path to saved LabelEncoder.")
    
    # Batch inference
    parser.add_argument("--input_csv", type=str, default=None,
                        help="Path to input CSV file for batch predictions.")
    parser.add_argument("--output_csv", type=str, default="results/predictions.csv",
                        help="Path to output CSV file for batch predictions.")
    
    # Single transaction inference
    parser.add_argument("--step", type=int, default=None, help="Step / hour of transaction (1-744).")
    parser.add_argument("--type", type=str, default=None,
                        choices=['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'],
                        help="Transaction type.")
    parser.add_argument("--amount", type=float, default=None, help="Transaction amount.")
    parser.add_argument("--oldbalanceOrg", type=float, default=None, help="Origin account balance before transaction.")
    parser.add_argument("--oldbalanceDest", type=float, default=None, help="Destination account balance before transaction.")
    parser.add_argument("--newbalanceDest", type=float, default=None, help="Destination account balance after transaction.")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold.")

    return parser.parse_args()


def get_risk_level(prob: float) -> str:
    if prob >= 0.80:
        return "CRITICAL"
    elif prob >= 0.50:
        return "HIGH"
    elif prob >= 0.20:
        return "MEDIUM"
    else:
        return "LOW"


def predict_single(model: FraudDetectionModel, encoder, args):
    if any(v is None for v in [args.step, args.type, args.amount, args.oldbalanceOrg, args.oldbalanceDest, args.newbalanceDest]):
        print("Error: For single transaction prediction, all feature arguments must be provided:")
        print("--step, --type, --amount, --oldbalanceOrg, --oldbalanceDest, --newbalanceDest")
        return

    # Encode transaction type
    try:
        type_encoded = encoder.transform([args.type])[0]
    except Exception:
        type_encoded = 0

    record = pd.DataFrame([{
        'step': args.step,
        'type': type_encoded,
        'amount': args.amount,
        'oldbalanceOrg': args.oldbalanceOrg,
        'oldbalanceDest': args.oldbalanceDest,
        'newbalanceDest': args.newbalanceDest
    }])[FEATURE_COLUMNS]

    prob = model.predict_proba(record)[0]
    pred = model.predict(record, threshold=args.threshold)[0]
    risk = get_risk_level(prob)

    print("\n" + "=" * 50)
    print("      TRANSACTION FRAUD DETECTION RESULT")
    print("=" * 50)
    print(f"Transaction Type  : {args.type}")
    print(f"Amount            : ${args.amount:,.2f}")
    print(f"Initial Bal (Orig): ${args.oldbalanceOrg:,.2f}")
    print(f"Initial Bal (Dest): ${args.oldbalanceDest:,.2f}")
    print(f"New Bal (Dest)    : ${args.newbalanceDest:,.2f}")
    print("-" * 50)
    print(f"Fraud Probability : {prob * 100:.2f}%")
    print(f"Prediction        : {'[ALERT: FRAUD DETECTED] (Class 1)' if pred == 1 else '[LEGITIMATE] (Class 0)'}")
    print(f"Risk Level        : {risk}")
    print("=" * 50 + "\n")


def predict_batch(model: FraudDetectionModel, encoder, input_csv: str, output_csv: str, threshold: float = None):
    print(f"Reading input transactions from: {input_csv}...")
    df = pd.read_csv(input_csv)
    df_processed = df.copy()

    if 'type' in df_processed.columns and df_processed['type'].dtype == object:
        df_processed['type'] = encoder.transform(df_processed['type'].astype(str))

    available_cols = [c for c in FEATURE_COLUMNS if c in df_processed.columns]
    X = df_processed[available_cols]

    probs = model.predict_proba(X)
    preds = model.predict(X, threshold=threshold)
    risk_levels = [get_risk_level(p) for p in probs]

    df['Fraud_Probability'] = np.round(probs, 4)
    df['Predicted_isFraud'] = preds
    df['Risk_Level'] = risk_levels

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Batch inference complete! Results saved to: {output_csv}")
    print(f"Total scored: {len(df):,}")
    print(f"Flagged as Fraud: {sum(preds == 1):,} ({(sum(preds == 1)/len(df))*100:.2f}%)")


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Trained model not found at '{args.model_path}'. Please run train.py first.")
    if not os.path.exists(args.encoder_path):
        raise FileNotFoundError(f"Label encoder not found at '{args.encoder_path}'. Please run train.py first.")

    model = FraudDetectionModel.load(args.model_path)
    encoder = joblib.load(args.encoder_path)

    if args.input_csv is not None:
        predict_batch(model, encoder, args.input_csv, args.output_csv, threshold=args.threshold)
    elif args.type is not None:
        predict_single(model, encoder, args)
    else:
        print("Please provide transaction features (--type, --amount, etc.) or --input_csv for batch prediction.")
        print("Run with --help for detailed usage options.")


if __name__ == '__main__':
    main()
