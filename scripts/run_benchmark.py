"""
Benchmark Comparison Script for Credit Card Fraud Detection Sampling Techniques.
Reproduces Table 4 and Figure 9 from Alamri & Ykhlef (IEEE Access, 2024):
Compares:
- SMOTE + RF
- B-SMOTE + RF
- ADASYN + RF
- SMOTE-Tomek + RF
- SMOTEENN + RF
- Hybrid Tomek Links BCBSMOTE (Proposed) + RF

Usage:
------
python run_benchmark.py --sample_size 250000
"""

import os
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess_data
from src.sampling import get_baseline_sampler
from src.visualization import plot_benchmark_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark comparison of sampling techniques")
    parser.add_argument("--data_path", type=str, default="archive/creditcard.csv",
                        help="Path to CSV dataset file (default: archive/creditcard.csv).")
    parser.add_argument("--sample_size", type=int, default=200000,
                        help="Number of samples to evaluate for benchmark comparison (default: 200,000 for rapid execution).")
    parser.add_argument("--sampling_ratio", type=float, default=0.02)
    parser.add_argument("--n_estimators", type=int, default=40)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="results/paysim_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print(" SAMPLING TECHNIQUES BENCHMARK COMPARISON (TABLE 4 & FIGURE 9) ")
    print(" Paper: Alamri & Ykhlef (IEEE Access, 2024)")
    print("=" * 75)

    # 1. Load Data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(
        file_path=args.data_path,
        sample_size=args.sample_size,
        test_size=0.20,
        random_state=42
    )

    methods = [
        'SMOTE',
        'B-SMOTE',
        'ADASYN',
        'SMOTE-Tomek',
        'SMOTEENN',
        'Proposed'
    ]

    results = []

    for method in methods:
        print(f"\nEvaluating Method: {method} ...")
        t0 = time.time()
        try:
            sampler = get_baseline_sampler(method, sampling_ratio=args.sampling_ratio, random_state=42)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            
            clf = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                n_jobs=-1,
                random_state=42
            )
            clf.fit(X_res, y_res)
            
            probs = clf.predict_proba(X_test)[:, 1]
            # Use optimal threshold or standard 0.5
            preds = (probs >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            prec = precision_score(y_test, preds, zero_division=0)
            auprc = average_precision_score(y_test, probs)
            elapsed = time.time() - t0
            
            print(f"[{method}] Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, Rec: {rec*100:.2f}%, Prec: {prec*100:.2f}%, AUPRC: {auprc*100:.2f}% ({elapsed:.1f}s)")
            
            results.append({
                'Method': method if method != 'Proposed' else 'Hybrid Tomek links BCBSMOTE (Proposed)',
                'Accuracy': acc,
                'F1_score': f1,
                'Recall': rec,
                'Precision': prec,
                'AUPRC': auprc,
                'Runtime_Sec': elapsed
            })
        except Exception as e:
            print(f"Error evaluating {method}: {e}")

    df_results = pd.DataFrame(results)
    
    # Format Table 4
    df_display = df_results.copy()
    for col in ['Accuracy', 'F1_score', 'Recall', 'Precision', 'AUPRC']:
        df_display[col] = df_display[col].apply(lambda x: f"{x*100:.2f}%")
        
    print("\n" + "=" * 80)
    print("                    TABLE 4. PERFORMANCE EVALUATION")
    print("=" * 80)
    print(df_display.to_string(index=False))
    print("=" * 80 + "\n")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "table_4_benchmark_evaluation.csv")
    df_display.to_csv(csv_path, index=False)
    print(f"Saved benchmark results table to: {csv_path}")

    # Plot Figure 9
    plot_path = os.path.join(args.output_dir, "figure_9_benchmark_comparison.png")
    plot_benchmark_comparison(df_results, output_path=plot_path)


if __name__ == '__main__':
    main()
