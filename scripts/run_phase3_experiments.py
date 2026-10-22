
"""
Automated Experiment Runner for Phase 2 & Phase 3 Technical Review Report.
Dataset: European Cardholders Credit Card Transactions (archive/creditcard.csv, OpenML ID 1597).
Executes:
1. Baseline vs Authors' BCBSMOTE Benchmark
2. 4-Stage Systematic Ablation Study
3. Multi-Classifier Comparison (Logistic Regression, Random Forest, XGBoost, Balanced RF)
4. Stress-Testing Suite (Label Noise Injection & Feature Corruption)
Exports:
- Tables: results/report_assets/*.csv
- Metrics: results/report_assets/experiment_metrics.json
- Figures: results/report_assets/*.png
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve
)

# Imbalanced-learn & XGBoost
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb

# Project modules
sys.path.append(os.path.abspath('.'))
from src.sampling import TomekLinksUndersampler, BCBSMOTEOversampler

# Style
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})

ASSET_DIR = "results/report_assets"
os.makedirs(ASSET_DIR, exist_ok=True)


def load_and_clean_data(filepath="archive/creditcard.csv"):
    print(f"Loading {filepath}...")
    df_raw = pd.read_csv(filepath)
    raw_shape = df_raw.shape
    n_raw_fraud = int(df_raw['Class'].sum())
    
    # Deduplication
    n_dups = int(df_raw.duplicated().sum())
    df = df_raw.drop_duplicates().reset_index(drop=True)
    clean_shape = df.shape
    n_clean_fraud = int(df['Class'].sum())
    
    print(f"Raw shape: {raw_shape}, Duplicates removed: {n_dups:,}, Clean shape: {clean_shape}")
    print(f"Fraud count: {n_clean_fraud:,} / {clean_shape[0]:,} ({n_clean_fraud/clean_shape[0]*100:.4f}%)")
    
    data_info = {
        'raw_rows': raw_shape[0],
        'raw_cols': raw_shape[1],
        'duplicates_removed': n_dups,
        'clean_rows': clean_shape[0],
        'clean_fraud': n_clean_fraud,
        'fraud_ratio_pct': round(n_clean_fraud/clean_shape[0]*100, 4)
    }
    return df, data_info


def run_experiment_1_baseline_vs_bcbsmote(X_train_s, y_train, X_test_s, y_test):
    print("\n" + "=" * 60)
    print(" EXPERIMENT 1: BASELINE (UN-RESAMPLED) VS AUTHORS' BCBSMOTE ")
    print("=" * 60)
    
    # 1. Baseline Model (No Resampling)
    t0 = time.time()
    rf_base = RandomForestClassifier(n_estimators=40, max_depth=14, n_jobs=-1, random_state=42)
    rf_base.fit(X_train_s, y_train)
    probs_base = rf_base.predict_proba(X_test_s)[:, 1]
    preds_base = (probs_base >= 0.5).astype(int)
    t_base = time.time() - t0
    
    cm_base = confusion_matrix(y_test, preds_base)
    base_res = {
        'Method': 'Baseline (No Resampling)',
        'Accuracy': float(accuracy_score(y_test, preds_base)),
        'F1_Score': float(f1_score(y_test, preds_base, zero_division=0)),
        'AUPRC': float(average_precision_score(y_test, probs_base)),
        'MCC': float(matthews_corrcoef(y_test, preds_base)),
        'Recall': float(recall_score(y_test, preds_base, zero_division=0)),
        'Precision': float(precision_score(y_test, preds_base, zero_division=0)),
        'TN': int(cm_base[0, 0]), 'FP': int(cm_base[0, 1]),
        'FN': int(cm_base[1, 0]), 'TP': int(cm_base[1, 1]),
        'Runtime_s': round(t_base, 2)
    }
    
    # 2. BCBSMOTE Hybrid Pipeline
    t0 = time.time()
    tl = TomekLinksUndersampler()
    X_tl, y_tl = tl.fit_resample(X_train_s, y_train)
    
    bc = BCBSMOTEOversampler(cluster_size=5, sampling_ratio=0.05, random_state=42)
    X_res, y_res = bc.fit_resample(X_tl, y_tl)
    
    rf_bc = RandomForestClassifier(n_estimators=40, max_depth=14, n_jobs=-1, random_state=42)
    rf_bc.fit(X_res, y_res)
    probs_bc = rf_bc.predict_proba(X_test_s)[:, 1]
    preds_bc = (probs_bc >= 0.5).astype(int)
    t_bc = time.time() - t0
    
    cm_bc = confusion_matrix(y_test, preds_bc)
    bc_res = {
        'Method': 'Hybrid BCBSMOTE (Proposed)',
        'Accuracy': float(accuracy_score(y_test, preds_bc)),
        'F1_Score': float(f1_score(y_test, preds_bc, zero_division=0)),
        'AUPRC': float(average_precision_score(y_test, probs_bc)),
        'MCC': float(matthews_corrcoef(y_test, preds_bc)),
        'Recall': float(recall_score(y_test, preds_bc, zero_division=0)),
        'Precision': float(precision_score(y_test, preds_bc, zero_division=0)),
        'TN': int(cm_bc[0, 0]), 'FP': int(cm_bc[0, 1]),
        'FN': int(cm_bc[1, 0]), 'TP': int(cm_bc[1, 1]),
        'Runtime_s': round(t_bc, 2)
    }
    
    df_exp1 = pd.DataFrame([base_res, bc_res])
    df_exp1.to_csv(os.path.join(ASSET_DIR, "table_1_baseline_vs_bcbsmote.csv"), index=False)
    print(df_exp1.to_string(index=False))
    
    # Plot PR Curves
    p_b, r_b, _ = precision_recall_curve(y_test, probs_base)
    p_bc, r_bc, _ = precision_recall_curve(y_test, probs_bc)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_b, p_b, color='#757575', lw=2, linestyle='--', label=f"Baseline RF (AUPRC = {base_res['AUPRC']*100:.2f}%)")
    ax.plot(r_bc, p_bc, color='#1E88E5', lw=2.5, label=f"Hybrid BCBSMOTE (AUPRC = {bc_res['AUPRC']*100:.2f}%)")
    ax.axhline(y=y_test.sum()/len(y_test), color='red', linestyle=':', label=f"Random Guess ({y_test.sum()/len(y_test)*100:.2f}%)")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Figure 1: Precision-Recall Curves (Baseline vs Proposed BCBSMOTE)')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "figure_1_pr_curve_comparison.png"), bbox_inches='tight')
    plt.close()
    
    # Plot Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for idx, (m_name, cm, ax) in enumerate([('Baseline RF (Un-resampled)', cm_base, axes[0]),
                                          ('Proposed Hybrid BCBSMOTE', cm_bc, axes[1])]):
        labels = np.array([[f"TN\n{cm[0, 0]:,}", f"FP\n{cm[0, 1]:,}"],
                           [f"FN\n{cm[1, 0]:,}", f"TP\n{cm[1, 1]:,}"]])
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Pred Genuine (0)', 'Pred Fraud (1)'],
                    yticklabels=['Actual Genuine (0)', 'Actual Fraud (1)'])
        ax.set_title(m_name)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "figure_2_confusion_matrices.png"), bbox_inches='tight')
    plt.close()
    
    # Cost curve
    thresholds = np.linspace(0.01, 0.99, 100)
    cost_fn, cost_fp = 500, 10
    costs = []
    for t in thresholds:
        cm = confusion_matrix(y_test, (probs_bc >= t).astype(int))
        costs.append((cm[1, 0] * cost_fn) + (cm[0, 1] * cost_fp))
    
    opt_t = thresholds[np.argmin(costs)]
    min_cost = min(costs)
    def_cost = costs[50]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, costs, color='#D32F2F', lw=2.5, label='Total Expected Financial Loss ($)')
    ax.axvline(opt_t, color='green', linestyle='--', label=f'Optimal Threshold = {opt_t:.2f} (Loss: ${min_cost:,.0f})')
    ax.axvline(0.50, color='blue', linestyle=':', label=f'Default Threshold = 0.50 (Loss: ${def_cost:,.0f})')
    ax.set_xlabel('Classification Decision Threshold')
    ax.set_ylabel('Total Expected Loss ($)')
    ax.set_title('Figure 3: Cost-Sensitive Loss Curve vs Decision Threshold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "figure_3_cost_sensitive_curve.png"), bbox_inches='tight')
    plt.close()
    
    cost_info = {
        'optimal_threshold': round(float(opt_t), 4),
        'min_financial_loss': round(float(min_cost), 2),
        'default_financial_loss': round(float(def_cost), 2),
        'financial_savings': round(float(def_cost - min_cost), 2)
    }
    
    return df_exp1, (X_res, y_res), (probs_base, probs_bc), cost_info


def run_experiment_2_ablation_study(X_train_s, y_train, X_test_s, y_test):
    print("\n" + "=" * 60)
    print(" EXPERIMENT 2: SYSTEMATIC ABLATION STUDY ")
    print("=" * 60)
    
    configs = [
        ('1. Baseline (No Resampling)', None, None),
        ('2. Tomek Links Only', TomekLinksUndersampler(), None),
        ('3. Borderline-SMOTE Only', None, BorderlineSMOTE(sampling_strategy=0.05, random_state=42)),
        ('4. Full Proposed BCBSMOTE', TomekLinksUndersampler(), BCBSMOTEOversampler(sampling_ratio=0.05, random_state=42))
    ]
    
    results = []
    for name, under, over in configs:
        t0 = time.time()
        X_cur, y_cur = X_train_s.copy(), y_train.copy()
        if under is not None:
            X_cur, y_cur = under.fit_resample(X_cur, y_cur)
        if over is not None:
            X_cur, y_cur = over.fit_resample(X_cur, y_cur)
            
        rf = RandomForestClassifier(n_estimators=30, max_depth=12, n_jobs=-1, random_state=42)
        rf.fit(X_cur, y_cur)
        
        probs = rf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        results.append({
            'Configuration': name,
            'F1_Score': float(f1_score(y_test, preds, zero_division=0)),
            'AUPRC': float(average_precision_score(y_test, probs)),
            'MCC': float(matthews_corrcoef(y_test, preds)),
            'Recall': float(recall_score(y_test, preds, zero_division=0)),
            'Precision': float(precision_score(y_test, preds, zero_division=0)),
            'Training_Time_s': round(time.time() - t0, 2)
        })
        
    df_ablation = pd.DataFrame(results)
    df_ablation.to_csv(os.path.join(ASSET_DIR, "table_2_ablation_study.csv"), index=False)
    print(df_ablation.to_string(index=False))
    
    # Plot Ablation Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_to_plot = ['F1_Score', 'AUPRC', 'Recall', 'Precision']
    df_plot = df_ablation.set_index('Configuration')[metrics_to_plot] * 100
    df_plot.plot(kind='bar', ax=ax, width=0.75, colormap='viridis')
    ax.set_ylabel('Score (%)')
    ax.set_title('Figure 4: Systematic Ablation Study of Resampling Components')
    ax.set_ylim([0, 110])
    ax.set_xticklabels(df_ablation['Configuration'], rotation=15, ha='right')
    ax.legend(title='Metric', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "figure_4_ablation_study.png"), bbox_inches='tight')
    plt.close()
    
    return df_ablation


def run_experiment_3_classifier_benchmark(X_train_s, y_train, X_res, y_res, X_test_s, y_test):
    print("\n" + "=" * 60)
    print(" EXPERIMENT 3: MULTI-CLASSIFIER BENCHMARKING (PHASE 3.c) ")
    print("=" * 60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest (Baseline)': RandomForestClassifier(n_estimators=40, max_depth=14, n_jobs=-1, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42),
        'Balanced Random Forest': BalancedRandomForestClassifier(n_estimators=40, random_state=42)
    }
    
    results = []
    for name, clf in models.items():
        t0 = time.time()
        if name == 'Balanced Random Forest':
            clf.fit(X_train_s, y_train)
        else:
            clf.fit(X_res, y_res)
            
        probs = clf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        results.append({
            'Classifier': name,
            'Accuracy': float(accuracy_score(y_test, preds)),
            'F1_Score': float(f1_score(y_test, preds, zero_division=0)),
            'AUPRC': float(average_precision_score(y_test, probs)),
            'MCC': float(matthews_corrcoef(y_test, preds)),
            'Recall': float(recall_score(y_test, preds, zero_division=0)),
            'Precision': float(precision_score(y_test, preds, zero_division=0)),
            'Runtime_s': round(time.time() - t0, 2)
        })
        
    df_models = pd.DataFrame(results)
    df_models.to_csv(os.path.join(ASSET_DIR, "table_3_classifier_benchmark.csv"), index=False)
    print(df_models.to_string(index=False))
    
    # Plot Classifier Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_to_plot = ['F1_Score', 'AUPRC', 'Recall', 'Precision']
    df_plot = df_models.set_index('Classifier')[metrics_to_plot] * 100
    df_plot.plot(kind='bar', ax=ax, width=0.75, colormap='magma')
    ax.set_ylabel('Score (%)')
    ax.set_title('Figure 5: Classifier Performance Comparison across Resampled Data')
    ax.set_ylim([0, 110])
    ax.set_xticklabels(df_models['Classifier'], rotation=15, ha='right')
    ax.legend(title='Metric', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "figure_5_classifier_benchmark.png"), bbox_inches='tight')
    plt.close()
    
    return df_models


def run_experiment_4_stress_testing(X_train_s, y_train, X_test_s, y_test, f1_full, auprc_full):
    print("\n" + "=" * 60)
    print(" EXPERIMENT 4: STRESS-TESTING SUITE (PHASE 3.d) ")
    print("=" * 60)
    
    rng = np.random.RandomState(42)
    
    # 4A: Label noise injection
    noise_levels = [0.0, 0.01, 0.05, 0.10]
    noise_records = []
    
    for nl in noise_levels:
        y_tr_noisy = y_train.copy()
        if nl > 0:
            flips = rng.choice(len(y_tr_noisy), size=int(nl * len(y_tr_noisy)), replace=False)
            y_tr_noisy[flips] = 1 - y_tr_noisy[flips]
            
        tl = TomekLinksUndersampler()
        X_tl_n, y_tl_n = tl.fit_resample(X_train_s, y_tr_noisy)
        bc = BCBSMOTEOversampler(sampling_ratio=0.05, random_state=42)
        X_res_n, y_res_n = bc.fit_resample(X_tl_n, y_tl_n)
        
        rf = RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_res_n, y_res_n)
        probs = rf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        noise_records.append({
            'Noise_Level': f"{int(nl*100)}%",
            'F1_Score': float(f1_score(y_test, preds, zero_division=0)),
            'AUPRC': float(average_precision_score(y_test, probs)),
            'MCC': float(matthews_corrcoef(y_test, preds)),
            'Recall': float(recall_score(y_test, preds, zero_division=0)),
            'Precision': float(precision_score(y_test, preds, zero_division=0))
        })
        
    df_noise = pd.DataFrame(noise_records)
    df_noise.to_csv(os.path.join(ASSET_DIR, "table_4a_noise_stress_test.csv"), index=False)
    print("--- 4A: Label Noise Stress Test ---")
    print(df_noise.to_string(index=False))
    
    # 4B: Critical Feature Pruning (V14, V17, V12)
    top_cols = ['V14', 'V17', 'V12']
    col_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
    keep_idx = [i for i, c in enumerate(col_names) if c not in top_cols]
    
    X_train_p = X_train_s[:, keep_idx]
    X_test_p = X_test_s[:, keep_idx]
    
    bc = BCBSMOTEOversampler(sampling_ratio=0.05, random_state=42)
    X_res_p, y_res_p = bc.fit_resample(X_train_p, y_train)
    
    rf_p = RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42)
    rf_p.fit(X_res_p, y_res_p)
    probs_p = rf_p.predict_proba(X_test_p)[:, 1]
    preds_p = (probs_p >= 0.5).astype(int)
    
    prune_res = {
        'Scenario': 'Critical Features Withheld (V14, V17, V12)',
        'F1_Score': float(f1_score(y_test, preds_p, zero_division=0)),
        'AUPRC': float(average_precision_score(y_test, probs_p)),
        'MCC': float(matthews_corrcoef(y_test, preds_p)),
        'Recall': float(recall_score(y_test, preds_p, zero_division=0)),
        'Precision': float(precision_score(y_test, preds_p, zero_division=0))
    }
    df_prune = pd.DataFrame([prune_res])
    df_prune.to_csv(os.path.join(ASSET_DIR, "table_4b_feature_pruning_test.csv"), index=False)
    print("\n--- 4B: Feature Pruning Stress Test ---")
    print(df_prune.to_string(index=False))
    
    # Plot Noise Degradation Curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_noise = [0, 1, 5, 10]
    ax.plot(x_noise, [r['F1_Score']*100 for r in noise_records], marker='o', lw=2.5, color='#1E88E5', label='F1-Score (%)')
    ax.plot(x_noise, [r['AUPRC']*100 for r in noise_records], marker='s', lw=2.5, color='#43A047', label='AUPRC (%)')
    ax.plot(x_noise, [r['Recall']*100 for r in noise_records], marker='^', lw=2, linestyle='--', color='#FB8C00', label='Recall (%)')
    ax.set_xlabel('Label Noise Injected (%)')
    ax.set_ylabel('Performance Score (%)')
    ax.set_title('Figure 6: Robustness Degradation under Label Noise Injection')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "figure_6_noise_stress_curve.png"), bbox_inches='tight')
    plt.close()
    
    return df_noise, df_prune


def main():
    print("Starting automated Phase 3 empirical evaluation suite...")
    start_time = time.time()
    
    df, data_info = load_and_clean_data()
    
    X = df.drop(columns=['Class']).values
    y = df['Class'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Scale strictly on training fold
    scaler = RobustScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 1. Exp 1: Baseline vs BCBSMOTE
    df_exp1, (X_res, y_res), (probs_base, probs_bc), cost_info = run_experiment_1_baseline_vs_bcbsmote(
        X_train_s, y_train, X_test_s, y_test
    )
    
    # 2. Exp 2: Ablation Study
    df_ablation = run_experiment_2_ablation_study(
        X_train_s, y_train, X_test_s, y_test
    )
    
    # 3. Exp 3: Classifier Benchmark
    df_models = run_experiment_3_classifier_benchmark(
        X_train_s, y_train, X_res, y_res, X_test_s, y_test
    )
    
    # 4. Exp 4: Stress-Testing
    df_noise, df_prune = run_experiment_4_stress_testing(
        X_train_s, y_train, X_test_s, y_test,
        f1_full=df_exp1.loc[1, 'F1_Score'],
        auprc_full=df_exp1.loc[1, 'AUPRC']
    )
    
    # Aggregate summary JSON
    total_time = round(time.time() - start_time, 2)
    all_metrics = {
        'dataset_info': data_info,
        'experiment_1_baseline_vs_bcbsmote': df_exp1.to_dict(orient='records'),
        'cost_sensitive_analysis': cost_info,
        'experiment_2_ablation_study': df_ablation.to_dict(orient='records'),
        'experiment_3_classifier_benchmark': df_models.to_dict(orient='records'),
        'experiment_4a_noise_stress_test': df_noise.to_dict(orient='records'),
        'experiment_4b_feature_pruning_test': df_prune.to_dict(orient='records'),
        'total_execution_time_s': total_time
    }
    
    with open(os.path.join(ASSET_DIR, "experiment_metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"\n[DONE] All experiments completed in {total_time:.2f}s! Assets saved to {ASSET_DIR}/")


if __name__ == '__main__':
    main()
