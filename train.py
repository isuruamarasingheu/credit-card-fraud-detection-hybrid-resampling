"""
Main Training Script for Credit Card Fraud Detection.
Paper: "Hybrid Undersampling and Oversampling for Handling Imbalanced Credit Card Data"
Authors: Maram Alamri and Mourad Ykhlef (IEEE Access, 2024).

Usage:
------
# Train on full dataset:
python train.py

# Train on a quick sample (e.g. 500k rows):
python train.py --sample_size 500000

# Custom parameters:
python train.py --sampling_ratio 0.02 --n_estimators 50 --max_depth 15
"""

import os
import argparse
import time
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import load_and_preprocess_data
from src.sampling import HybridTomekBCBSMOTE
from src.model import FraudDetectionModel
from src.visualization import (
    plot_class_distribution,
    plot_precision_recall_curve,
    plot_confusion_matrix_heatmap
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Credit Card Fraud Detection Model using Hybrid Tomek Links + BCBSMOTE and Random Forest"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="archive/PS_20174392719_1491204439457_log.csv",
        help="Path to PaySim CSV dataset file."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional number of rows to sample for fast training/testing. Default is full dataset."
    )
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=0.02,
        help="Target sampling ratio of minority to majority class for BCBSMOTE (default: 0.02)."
    )
    parser.add_argument(
        "--cluster_size",
        type=int,
        default=5,
        help="Minimum cluster size for BIRCH clustering in BCBSMOTE (default: 5)."
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=50,
        help="Number of trees in Random Forest classifier (default: 50)."
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=15,
        help="Maximum depth of decision trees in Random Forest (default: 15)."
    )
    parser.add_argument(
        "--optimize_threshold",
        action="store_true",
        default=True,
        help="Whether to optimize decision threshold on validation set (default: True)."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="models/fraud_detection_model.joblib",
        help="Output path to save the trained model checkpoint."
    )
    parser.add_argument(
        "--encoder_save_path",
        type=str,
        default="models/label_encoder.joblib",
        help="Output path to save the fitted LabelEncoder for transaction types."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation plots and metrics."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.model_save_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.encoder_save_path)), exist_ok=True)

    print("=" * 70)
    print(" CREDIT CARD FRAUD DETECTION USING HYBRID TOMEK LINKS & BCBSMOTE ")
    print(" Paper: Alamri & Ykhlef (IEEE Access, 2024)")
    print("=" * 70)
    print(f"Dataset path       : {args.data_path}")
    print(f"Sample size        : {args.sample_size if args.sample_size else 'Full Dataset (6.36M rows)'}")
    print(f"Sampling ratio     : {args.sampling_ratio}")
    print(f"Cluster size       : {args.cluster_size}")
    print(f"Random Forest trees: {args.n_estimators} (max_depth={args.max_depth})")
    print(f"Model output path  : {args.model_save_path}")
    print("=" * 70 + "\n")

    start_total_time = time.time()

    # 1. Load and Preprocess Data
    t0 = time.time()
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(
        file_path=args.data_path,
        sample_size=args.sample_size,
        test_size=0.20,
        random_state=42
    )
    print(f"Data loading & preprocessing took: {time.time() - t0:.2f}s")

    # Save label encoder
    joblib.dump(label_encoder, args.encoder_save_path)
    print(f"Saved LabelEncoder to: {args.encoder_save_path}")

    # Plot initial class distribution
    plot_class_distribution(
        pd.concat([y_train, y_test]),
        output_path=os.path.join(args.output_dir, "figure_1_class_distribution.png")
    )

    # 2. Configure Hybrid Resampler
    sampler = HybridTomekBCBSMOTE(
        cluster_size=args.cluster_size,
        sampling_ratio=args.sampling_ratio,
        random_state=42
    )

    # 3. Instantiate and Train Model
    model = FraudDetectionModel(
        sampler=sampler,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=42
    )

    t0 = time.time()
    model.fit(X_train, y_train, resample=True)
    training_time = time.time() - t0
    print(f"\nResampling + Random Forest Training finished in: {training_time:.2f}s")

    # 4. Optional: Optimize Decision Threshold
    if args.optimize_threshold:
        print("\nOptimizing decision threshold on training sample...")
        val_sample_size = min(len(X_train), 100000)
        rng = np.random.RandomState(42)
        val_idx = rng.choice(len(X_train), size=val_sample_size, replace=False)
        model.optimize_threshold(X_train.iloc[val_idx], y_train.iloc[val_idx], metric='f1')

    # 5. Evaluate on 20% Test Set
    t0 = time.time()
    metrics = model.evaluate(X_test, y_test)
    eval_time = time.time() - t0
    print(f"Evaluation on {len(X_test):,} test samples took: {eval_time:.2f}s")

    # 6. Generate and Save Visualizations
    plot_precision_recall_curve(
        y_test, metrics['Probabilities'], metrics['AUPRC'],
        output_path=os.path.join(args.output_dir, "figure_10_auprc_curve.png")
    )
    plot_confusion_matrix_heatmap(
        metrics['Confusion_Matrix'],
        output_path=os.path.join(args.output_dir, "table_5_confusion_matrix.png")
    )

    # Save metrics summary to CSV
    metrics_summary = pd.DataFrame([{
        'Accuracy': f"{metrics['Accuracy']*100:.2f}%",
        'F1_Score': f"{metrics['F1_Score']*100:.2f}%",
        'Recall': f"{metrics['Recall']*100:.2f}%",
        'Precision': f"{metrics['Precision']*100:.2f}%",
        'AUPRC': f"{metrics['AUPRC']*100:.2f}%",
        'ROC_AUC': f"{metrics['ROC_AUC']*100:.2f}%",
        'True_Positives': metrics['TP'],
        'False_Positives': metrics['FP'],
        'True_Negatives': metrics['TN'],
        'False_Negatives': metrics['FN'],
        'Decision_Threshold': f"{metrics['Threshold']:.4f}",
        'Training_Time_Sec': f"{training_time:.2f}"
    }])
    summary_csv_path = os.path.join(args.output_dir, "evaluation_metrics_summary.csv")
    metrics_summary.to_csv(summary_csv_path, index=False)
    print(f"\nSaved metrics summary to: {summary_csv_path}")

    # 7. Save Model Checkpoint
    model.save(args.model_save_path)

    total_time = time.time() - start_total_time
    print(f"\n[DONE] Full pipeline execution completed in {total_time:.2f}s ({total_time/60:.2f} min).")


if __name__ == '__main__':
    main()
