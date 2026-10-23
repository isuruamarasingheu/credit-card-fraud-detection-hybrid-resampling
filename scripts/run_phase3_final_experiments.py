"""
Phase 3 Final Rigorous Empirical Evaluation Suite (Updated with User Specifications).
Covers Experiments E1 through E7 on European Cardholders (ULB) and PaySim datasets.
Rules & Additions:
1. Uniform settings: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
2. SAMPLING_RATIO = 0.05 for ULB (with E1 also running ratio=0.02) and 0.02 for PaySim
3. Resampling audit log: track removed_by_tomek and synthetic_generated for every fit; warn if 0
4. E2 symmetric noise: track genuine_to_fraud and fraud_to_genuine; fraud-only as realistic test
5. E3: report both dynamic top 3 and hardcoded top 3 (V14, V17, V12)
6. E5: threshold chosen strictly on training OOF CV; test scan as 'optimistic' column
7. E7: report both default threshold 0.5 and train-tuned threshold; report test fraud counts
8. Bug warnings logged and saved to run_info.json and run_log.txt
"""

import os
import sys
import time
import json
import random
import logging
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn & Imblearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, matthews_corrcoef, confusion_matrix
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb

# Project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sampling import TomekLinksUndersampler, BCBSMOTEOversampler, HybridTomekBCBSMOTE

OUTPUT_DIR = "results/phase3_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "run_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Phase3Final")

SEEDS = [0, 1, 2, 3, 4]
ULB_DEFAULT_SAMPLING_RATIO = 0.05
PAYSIM_SAMPLING_RATIO = 0.02

# Global audit tracker for resampling fits
RESAMPLING_AUDIT = []
AUDIT_WARNINGS = []


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def audit_sampler(sampler, experiment_id, method_name, seed, scenario="default"):
    removed = getattr(sampler, 'n_removed_', None)
    synthetic = getattr(sampler, 'n_synthetic_', None)
    
    # If BorderlineSMOTE from imblearn, inspect if available
    if synthetic is None and hasattr(sampler, 'sampling_strategy_'):
        pass
        
    audit_entry = {
        'experiment': experiment_id,
        'method': method_name,
        'seed': seed,
        'scenario': scenario,
        'majority_removed_tomek': removed,
        'synthetic_generated': synthetic
    }
    RESAMPLING_AUDIT.append(audit_entry)
    
    if synthetic is not None and synthetic == 0:
        warning_msg = f"[AUDIT WARNING] Zero synthetic samples created in {experiment_id} | {method_name} (Seed {seed}, {scenario})"
        logger.warning(warning_msg)
        AUDIT_WARNINGS.append(warning_msg)
    return audit_entry


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auprc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auprc': auprc,
        'mcc': mcc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def format_mean_std(df, group_cols, metric_cols):
    grouped = df.groupby(group_cols, sort=False)
    records = []
    for name, group in grouped:
        if not isinstance(name, tuple):
            name = (name,)
        row = dict(zip(group_cols, name))
        for col in metric_cols:
            if col in group.columns:
                mean_val = group[col].mean()
                std_val = group[col].std()
                row[f"{col}_mean"] = mean_val
                row[f"{col}_std"] = std_val
                if col in ['precision', 'recall', 'f1', 'auprc', 'mcc', 'saving']:
                    row[f"{col}_display"] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    row[f"{col}_display"] = f"{mean_val:.1f} ± {std_val:.1f}"
        records.append(row)
    return pd.DataFrame(records)


def load_ulb_data(filepath="archive/creditcard.csv"):
    logger.info(f"Loading European Cardholders dataset from {filepath}...")
    df_raw = pd.read_csv(filepath)
    raw_rows, raw_cols = df_raw.shape
    raw_frauds = int(df_raw['Class'].sum())
    
    # Exact duplicate removal (all columns incl. Time)
    n_dups = int(df_raw.duplicated().sum())
    df = df_raw.drop_duplicates().reset_index(drop=True)
    clean_rows = len(df)
    clean_frauds = int(df['Class'].sum())
    logger.info(f"ULB Loaded: raw={raw_rows:,} rows ({raw_frauds} frauds). Exact duplicates removed={n_dups:,}. Clean={clean_rows:,} rows ({clean_frauds} frauds).")
    return df, {
        'raw_rows': raw_rows,
        'raw_cols': raw_cols,
        'raw_frauds': raw_frauds,
        'exact_duplicates_removed': n_dups,
        'clean_rows': clean_rows,
        'clean_frauds': clean_frauds,
        'clean_genuine': clean_rows - clean_frauds
    }


# ==============================================================================
# EXPERIMENT 1: MAIN TABLE (Unified with both ratios 0.05 and 0.02)
# ==============================================================================
def run_experiment_1(df):
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: MAIN TABLE (Seeds 0-4, Ratios 0.05 and 0.02)")
    logger.info("=" * 70)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    preprocessor = ColumnTransformer(
        transformers=[('scaler', RobustScaler(), ['Time', 'Amount'])],
        remainder='passthrough'
    )
    
    per_seed_records = []
    
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        gen_count = (y_train == 0).sum()
        fraud_count = (y_train == 1).sum()
        scale_pos = gen_count / fraud_count
        
        methods = {
            '1. Plain Random Forest (No Resampling)': Pipeline([
                ('scaler', preprocessor),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ]),
            '2. Tomek Links Only + RF': Pipeline([
                ('scaler', preprocessor),
                ('sampler', TomekLinksUndersampler()),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ]),
            '3a. Borderline-SMOTE (ratio=0.05) + RF': Pipeline([
                ('scaler', preprocessor),
                ('sampler', BorderlineSMOTE(sampling_strategy=0.05, random_state=seed)),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ]),
            '3b. Borderline-SMOTE (ratio=0.02) + RF': Pipeline([
                ('scaler', preprocessor),
                ('sampler', BorderlineSMOTE(sampling_strategy=0.02, random_state=seed)),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ]),
            '4a. BCBSMOTE (ratio=0.05) + RF': Pipeline([
                ('scaler', preprocessor),
                ('sampler', HybridTomekBCBSMOTE(sampling_ratio=0.05, random_state=seed)),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ]),
            '4b. BCBSMOTE (ratio=0.02) + RF': Pipeline([
                ('scaler', preprocessor),
                ('sampler', HybridTomekBCBSMOTE(sampling_ratio=0.02, random_state=seed)),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ]),
            '5. XGBoost (No Resampling, scale_pos_weight)': Pipeline([
                ('scaler', preprocessor),
                ('clf', xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                    scale_pos_weight=scale_pos, random_state=seed, n_jobs=-1
                ))
            ]),
            '6. XGBoost + BCBSMOTE (ratio=0.05)': Pipeline([
                ('scaler', preprocessor),
                ('sampler', HybridTomekBCBSMOTE(sampling_ratio=0.05, random_state=seed)),
                ('clf', xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                    random_state=seed, n_jobs=-1
                ))
            ]),
            '7. Logistic Regression (class_weight=balanced)': Pipeline([
                ('scaler', preprocessor),
                ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed))
            ]),
            '8. Balanced Random Forest': Pipeline([
                ('scaler', preprocessor),
                ('clf', BalancedRandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1))
            ])
        }
        
        for name, pipe in methods.items():
            t0 = time.time()
            try:
                pipe.fit(X_train, y_train)
                # Audit resampling if present
                if 'sampler' in pipe.named_steps:
                    sampler_obj = pipe.named_steps['sampler']
                    audit_sampler(sampler_obj, 'E1', name, seed)
                    
                y_prob = pipe.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                runtime = time.time() - t0
                m = compute_metrics(y_test, y_pred, y_prob)
                rec = {
                    'method': name,
                    'seed': seed,
                    'runtime_s': round(runtime, 2),
                    **m
                }
                per_seed_records.append(rec)
                logger.info(f"[E1] Seed={seed} | {name} | F1={m['f1']:.4f} | AUPRC={m['auprc']:.4f} | Rec={m['recall']:.4f} | Prec={m['precision']:.4f} ({runtime:.1f}s)")
            except Exception as e:
                logger.error(f"[E1] Error in {name} with seed {seed}: {e}", exc_info=True)
                
    df_per_seed = pd.DataFrame(per_seed_records)
    df_per_seed.to_csv(os.path.join(OUTPUT_DIR, "e1_per_seed.csv"), index=False)
    
    metric_cols = ['precision', 'recall', 'f1', 'auprc', 'mcc', 'tn', 'fp', 'fn', 'tp', 'runtime_s']
    df_summary = format_mean_std(df_per_seed, ['method'], metric_cols)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "e1_summary.csv"), index=False)
    
    # Paired differences: BCBSMOTE (both 0.05 and 0.02) minus Plain RF
    rf_res = df_per_seed[df_per_seed['method'] == '1. Plain Random Forest (No Resampling)'].set_index('seed')
    bc_05_res = df_per_seed[df_per_seed['method'] == '4a. BCBSMOTE (ratio=0.05) + RF'].set_index('seed')
    bc_02_res = df_per_seed[df_per_seed['method'] == '4b. BCBSMOTE (ratio=0.02) + RF'].set_index('seed')
    
    paired_records = []
    for seed in SEEDS:
        for ratio_label, bc_df in [('ratio=0.05', bc_05_res), ('ratio=0.02', bc_02_res)]:
            d_f1 = bc_df.loc[seed, 'f1'] - rf_res.loc[seed, 'f1']
            d_auprc = bc_df.loc[seed, 'auprc'] - rf_res.loc[seed, 'auprc']
            d_mcc = bc_df.loc[seed, 'mcc'] - rf_res.loc[seed, 'mcc']
            d_rec = bc_df.loc[seed, 'recall'] - rf_res.loc[seed, 'recall']
            d_prec = bc_df.loc[seed, 'precision'] - rf_res.loc[seed, 'precision']
            paired_records.append({
                'comparison': f"BCBSMOTE ({ratio_label}) minus Plain RF",
                'seed': seed,
                'f1_diff': d_f1,
                'auprc_diff': d_auprc,
                'mcc_diff': d_mcc,
                'recall_diff': d_rec,
                'precision_diff': d_prec,
                'f1_positive': bool(d_f1 > 0),
                'auprc_positive': bool(d_auprc > 0)
            })
    df_paired = pd.DataFrame(paired_records)
    df_paired.to_csv(os.path.join(OUTPUT_DIR, "e1_paired_diff.csv"), index=False)
    return df_summary, df_per_seed, df_paired


# ==============================================================================
# EXPERIMENT 2: LABEL NOISE (Detailed Logging, Realistic Fraud-Only Focus)
# ==============================================================================
def run_experiment_2(df):
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: LABEL NOISE STRESS TEST (Seeds 0-4, Symmetric & Fraud-Only)")
    logger.info("=" * 70)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    preprocessor = ColumnTransformer(
        transformers=[('scaler', RobustScaler(), ['Time', 'Amount'])],
        remainder='passthrough'
    )
    
    symmetric_levels = [0.0, 0.01, 0.05, 0.10]
    fraud_only_levels = [0.10, 0.20, 0.30]
    
    per_seed_records = []
    
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        
        scenarios = [('symmetric', p) for p in symmetric_levels] + [('fraud_only', p) for p in fraud_only_levels]
        
        for noise_type, level in scenarios:
            noise_rng = np.random.RandomState(seed)
            y_train_noisy = y_train.copy().values
            
            if noise_type == 'symmetric':
                if level > 0:
                    flips_mask = noise_rng.uniform(size=len(y_train_noisy)) < level
                    n_flipped = int(flips_mask.sum())
                    n_gen_flipped = int(((y_train.values == 0) & flips_mask).sum())
                    n_fraud_flipped = int(((y_train.values == 1) & flips_mask).sum())
                    y_train_noisy[flips_mask] = 1 - y_train_noisy[flips_mask]
                else:
                    n_flipped, n_gen_flipped, n_fraud_flipped = 0, 0, 0
                scenario_label = f"Symmetric {int(level*100)}%"
            else:
                fraud_idx = np.where(y_train.values == 1)[0]
                n_to_flip = int(round(level * len(fraud_idx)))
                chosen = noise_rng.choice(fraud_idx, size=n_to_flip, replace=False)
                y_train_noisy[chosen] = 0
                n_flipped = n_to_flip
                n_gen_flipped = 0
                n_fraud_flipped = n_to_flip
                scenario_label = f"Fraud-Only {int(level*100)}%"
                
            y_train_noisy = pd.Series(y_train_noisy, index=y_train.index)
            
            models = {
                'Plain RF (Control)': Pipeline([
                    ('scaler', preprocessor),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ]),
                'BCBSMOTE + RF': Pipeline([
                    ('scaler', preprocessor),
                    ('sampler', HybridTomekBCBSMOTE(sampling_ratio=ULB_DEFAULT_SAMPLING_RATIO, random_state=seed)),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ])
            }
            
            for m_name, pipe in models.items():
                t0 = time.time()
                try:
                    pipe.fit(X_train, y_train_noisy)
                    if 'sampler' in pipe.named_steps:
                        sampler_obj = pipe.named_steps['sampler']
                        audit_sampler(sampler_obj, 'E2', m_name, seed, scenario_label)
                        
                    y_prob = pipe.predict_proba(X_test)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    m = compute_metrics(y_test, y_pred, y_prob)
                    runtime = time.time() - t0
                    rec = {
                        'noise_type': noise_type,
                        'noise_level': level,
                        'scenario': scenario_label,
                        'method': m_name,
                        'seed': seed,
                        'total_labels_flipped': n_flipped,
                        'genuine_to_fraud': n_gen_flipped,
                        'fraud_to_genuine': n_fraud_flipped,
                        'runtime_s': round(runtime, 2),
                        **m
                    }
                    per_seed_records.append(rec)
                    logger.info(f"[E2] Seed={seed} | {scenario_label} | {m_name} | F1={m['f1']:.4f} | AUPRC={m['auprc']:.4f} | Flipped: Gen->Fraud={n_gen_flipped}, Fraud->Gen={n_fraud_flipped}")
                except Exception as e:
                    logger.error(f"[E2] Error in {m_name} {scenario_label} seed {seed}: {e}", exc_info=True)
                    
    df_per_seed = pd.DataFrame(per_seed_records)
    df_per_seed.to_csv(os.path.join(OUTPUT_DIR, "e2_noise_per_seed.csv"), index=False)
    
    metric_cols = ['precision', 'recall', 'f1', 'auprc', 'mcc', 'total_labels_flipped', 'genuine_to_fraud', 'fraud_to_genuine', 'runtime_s']
    df_summary = format_mean_std(df_per_seed, ['noise_type', 'noise_level', 'scenario', 'method'], metric_cols)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "e2_noise_summary.csv"), index=False)
    
    # Plot E2 Noise
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sym_data = df_per_seed[df_per_seed['noise_type'] == 'symmetric']
    for m_name, col in [('Plain RF (Control)', '#424242'), ('BCBSMOTE + RF', '#1E88E5')]:
        sub = sym_data[sym_data['method'] == m_name]
        mean_s = sub.groupby('noise_level', sort=True)['f1'].mean() * 100
        std_s = sub.groupby('noise_level', sort=True)['f1'].std() * 100
        axes[0].errorbar([0, 1, 5, 10], mean_s.values, yerr=std_s.values, label=m_name,
                         fmt='-o', color=col, capsize=4, lw=2)
    axes[0].set_title("(a) Symmetric Label Noise on Training Set")
    axes[0].set_xlabel("Independent Flip Probability (%)")
    axes[0].set_ylabel("F1-Score (%)")
    axes[0].set_xticks([0, 1, 5, 10])
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    fraud_data = df_per_seed[df_per_seed['noise_type'] == 'fraud_only']
    for m_name, col in [('Plain RF (Control)', '#424242'), ('BCBSMOTE + RF', '#1E88E5')]:
        sub = fraud_data[fraud_data['method'] == m_name]
        mean_f = sub.groupby('noise_level', sort=True)['f1'].mean() * 100
        std_f = sub.groupby('noise_level', sort=True)['f1'].std() * 100
        axes[1].errorbar([10, 20, 30], mean_f.values, yerr=std_f.values, label=m_name,
                         fmt='-s', color=col, capsize=4, lw=2)
    axes[1].set_title("(b) Realistic Fraud-Only Noise (Missed Fraud Labels)")
    axes[1].set_xlabel("Training Frauds Flipped to Genuine (%)")
    axes[1].set_ylabel("F1-Score (%)")
    axes[1].set_xticks([10, 20, 30])
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_e2_noise.png"), dpi=200)
    plt.close()
    
    return df_summary, df_per_seed


# ==============================================================================
# EXPERIMENT 3: FEATURE REMOVAL (Both Dynamic Top 3 and Hard-Coded V14, V17, V12)
# ==============================================================================
def run_experiment_3(df):
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: FEATURE REMOVAL (Dynamic Top 3 vs Hard-Coded V14, V17, V12)")
    logger.info("=" * 70)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    preprocessor_full = ColumnTransformer(
        transformers=[('scaler', RobustScaler(), ['Time', 'Amount'])],
        remainder='passthrough'
    )
    
    per_seed_records = []
    top_3_dynamic_per_seed = {}
    hardcoded_top_3 = ['V14', 'V17', 'V12']
    
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        
        # Dynamically compute top 3 features by Random Forest importance on TRAINING set
        rf_probe = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
        X_tr_scaled = preprocessor_full.fit_transform(X_train)
        rf_probe.fit(X_tr_scaled, y_train)
        
        feat_names = ['Time', 'Amount'] + [c for c in X.columns if c not in ['Time', 'Amount']]
        importances = pd.Series(rf_probe.feature_importances_, index=feat_names)
        dynamic_top_3 = list(importances.sort_values(ascending=False).head(3).index)
        top_3_dynamic_per_seed[seed] = dynamic_top_3
        logger.info(f"[E3] Seed {seed} Dynamically Chosen Top 3 Features: {dynamic_top_3}")
        
        scenarios = [
            ('Control: All Features', []),
            ('Dynamic Top 3 Removed', dynamic_top_3),
            ('Hard-Coded (V14, V17, V12) Removed', hardcoded_top_3)
        ]
        
        for scenario_label, drop_cols in scenarios:
            X_tr_sub = X_train.drop(columns=drop_cols)
            X_te_sub = X_test.drop(columns=drop_cols)
            
            scale_cols = [c for c in ['Time', 'Amount'] if c not in drop_cols]
            prep_sub = ColumnTransformer(
                transformers=[('scaler', RobustScaler(), scale_cols)],
                remainder='passthrough'
            )
            
            models = {
                'Plain RF (Control)': Pipeline([
                    ('scaler', prep_sub),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ]),
                'BCBSMOTE + RF': Pipeline([
                    ('scaler', prep_sub),
                    ('sampler', HybridTomekBCBSMOTE(sampling_ratio=ULB_DEFAULT_SAMPLING_RATIO, random_state=seed)),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ])
            }
            
            for m_name, pipe in models.items():
                t0 = time.time()
                try:
                    pipe.fit(X_tr_sub, y_train)
                    if 'sampler' in pipe.named_steps:
                        sampler_obj = pipe.named_steps['sampler']
                        audit_sampler(sampler_obj, 'E3', m_name, seed, scenario_label)
                        
                    y_prob = pipe.predict_proba(X_te_sub)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    runtime = time.time() - t0
                    m = compute_metrics(y_test, y_pred, y_prob)
                    rec = {
                        'scenario': scenario_label,
                        'removed_features': ", ".join(drop_cols) if drop_cols else "None",
                        'method': m_name,
                        'seed': seed,
                        'runtime_s': round(runtime, 2),
                        **m
                    }
                    per_seed_records.append(rec)
                    logger.info(f"[E3] Seed={seed} | {scenario_label} | {m_name} | F1={m['f1']:.4f} | AUPRC={m['auprc']:.4f}")
                except Exception as e:
                    logger.error(f"[E3] Error in {m_name} {scenario_label} seed {seed}: {e}", exc_info=True)
                    
    df_per_seed = pd.DataFrame(per_seed_records)
    metric_cols = ['precision', 'recall', 'f1', 'auprc', 'mcc', 'runtime_s']
    df_summary = format_mean_std(df_per_seed, ['scenario', 'method'], metric_cols)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "e3_feature_removal.csv"), index=False)
    return df_summary, df_per_seed, top_3_dynamic_per_seed


# ==============================================================================
# EXPERIMENT 4: HIGHER DIMENSIONALITY (0, 20, 100 Noise Features)
# ==============================================================================
def run_experiment_4(df):
    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: HIGHER DIMENSIONS (0, 20, 100 Noise Features)")
    logger.info("=" * 70)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    dim_levels = [0, 20, 100]
    per_seed_records = []
    
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        
        noise_rng = np.random.RandomState(seed)
        
        for n_noise in dim_levels:
            if n_noise > 0:
                noise_tr = noise_rng.randn(len(X_train), n_noise)
                noise_te = noise_rng.randn(len(X_test), n_noise)
                noise_cols = [f"noise_{i}" for i in range(n_noise)]
                
                df_noise_tr = pd.DataFrame(noise_tr, columns=noise_cols, index=X_train.index)
                df_noise_te = pd.DataFrame(noise_te, columns=noise_cols, index=X_test.index)
                
                X_tr_dim = pd.concat([X_train, df_noise_tr], axis=1)
                X_te_dim = pd.concat([X_test, df_noise_te], axis=1)
                scale_cols = ['Time', 'Amount'] + noise_cols
            else:
                X_tr_dim = X_train.copy()
                X_te_dim = X_test.copy()
                scale_cols = ['Time', 'Amount']
                
            prep_dim = ColumnTransformer(
                transformers=[('scaler', RobustScaler(), scale_cols)],
                remainder='passthrough'
            )
            
            models = {
                'Plain RF': Pipeline([
                    ('scaler', prep_dim),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ]),
                'Borderline-SMOTE + RF': Pipeline([
                    ('scaler', prep_dim),
                    ('sampler', BorderlineSMOTE(sampling_strategy=ULB_DEFAULT_SAMPLING_RATIO, random_state=seed)),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ]),
                'BCBSMOTE + RF': Pipeline([
                    ('scaler', prep_dim),
                    ('sampler', HybridTomekBCBSMOTE(sampling_ratio=ULB_DEFAULT_SAMPLING_RATIO, random_state=seed)),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ])
            }
            
            for m_name, pipe in models.items():
                t0 = time.time()
                try:
                    pipe.fit(X_tr_dim, y_train)
                    if 'sampler' in pipe.named_steps:
                        sampler_obj = pipe.named_steps['sampler']
                        audit_sampler(sampler_obj, 'E4', m_name, seed, f"+{n_noise} noise dims")
                        
                    y_prob = pipe.predict_proba(X_te_dim)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    runtime = time.time() - t0
                    m = compute_metrics(y_test, y_pred, y_prob)
                    rec = {
                        'noise_features_added': n_noise,
                        'total_features': X_tr_dim.shape[1],
                        'method': m_name,
                        'seed': seed,
                        'runtime_s': round(runtime, 2),
                        **m
                    }
                    per_seed_records.append(rec)
                    logger.info(f"[E4] Seed={seed} | +{n_noise} noise dims | {m_name} | F1={m['f1']:.4f} | AUPRC={m['auprc']:.4f} | MCC={m['mcc']:.4f} ({runtime:.1f}s)")
                except Exception as e:
                    logger.error(f"[E4] Error in {m_name} +{n_noise} seed {seed}: {e}", exc_info=True)
                    
    df_per_seed = pd.DataFrame(per_seed_records)
    metric_cols = ['precision', 'recall', 'f1', 'auprc', 'mcc', 'runtime_s']
    df_summary = format_mean_std(df_per_seed, ['noise_features_added', 'total_features', 'method'], metric_cols)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "e4_dimensionality.csv"), index=False)
    
    # Plot E4 Dimensionality
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics_to_plot = [('f1', 'F1-Score (%)', axes[0]),
                       ('auprc', 'AUPRC (%)', axes[1]),
                       ('mcc', 'MCC', axes[2])]
    palette = {'Plain RF': '#424242', 'Borderline-SMOTE + RF': '#FB8C00', 'BCBSMOTE + RF': '#1E88E5'}
    
    for metric_name, label, ax in metrics_to_plot:
        for m_name, col in palette.items():
            sub = df_per_seed[df_per_seed['method'] == m_name]
            multiplier = 100 if metric_name != 'mcc' else 1
            mean_vals = sub.groupby('noise_features_added', sort=True)[metric_name].mean() * multiplier
            std_vals = sub.groupby('noise_features_added', sort=True)[metric_name].std() * multiplier
            ax.errorbar([0, 20, 100], mean_vals.values, yerr=std_vals.values, label=m_name,
                        fmt='-o', color=col, capsize=4, lw=2)
        ax.set_title(f"{label} vs Dimensionality")
        ax.set_xlabel("Noise Features Appended")
        ax.set_ylabel(label)
        ax.set_xticks([0, 20, 100])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_e4_dimensionality.png"), dpi=200)
    plt.close()
    
    return df_summary, df_per_seed


# ==============================================================================
# EXPERIMENT 5: COST THRESHOLD (5-Fold Stratified OOF on Training Set)
# ==============================================================================
def run_experiment_5(df):
    logger.info("=" * 70)
    logger.info("EXPERIMENT 5: COST THRESHOLD (OOF Training CV vs Optimistic Test Scan)")
    logger.info("=" * 70)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    cost_fn, cost_fp = 500, 10
    threshold_grid = np.linspace(0.01, 0.99, 100)
    
    preprocessor = ColumnTransformer(
        transformers=[('scaler', RobustScaler(), ['Time', 'Amount'])],
        remainder='passthrough'
    )
    
    records = []
    
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        
        models = {
            'Plain RF': lambda s: Pipeline([
                ('scaler', preprocessor),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s))
            ]),
            'BCBSMOTE + RF': lambda s: Pipeline([
                ('scaler', preprocessor),
                ('sampler', HybridTomekBCBSMOTE(sampling_ratio=ULB_DEFAULT_SAMPLING_RATIO, random_state=s)),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s))
            ])
        }
        
        for m_name, model_fn in models.items():
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            oof_probs = np.zeros(len(X_train))
            
            logger.info(f"[E5] Computing 5-Fold OOF training probabilities for {m_name} (Seed {seed})...")
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr_f, y_tr_f = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_val_f = X_train.iloc[val_idx]
                
                pipe_fold = model_fn(seed + fold_idx * 10)
                pipe_fold.fit(X_tr_f, y_tr_f)
                oof_probs[val_idx] = pipe_fold.predict_proba(X_val_f)[:, 1]
                
            # Find threshold minimizing total cost on training OOF predictions
            oof_costs = []
            for t in threshold_grid:
                preds_t = (oof_probs >= t).astype(int)
                cm_t = confusion_matrix(y_train, preds_t)
                cost_t = (cm_t[1, 0] * cost_fn) + (cm_t[0, 1] * cost_fp)
                oof_costs.append(cost_t)
                
            chosen_threshold = threshold_grid[np.argmin(oof_costs)]
            
            # Fit final model on full training set, evaluate ONCE on test set
            final_pipe = model_fn(seed)
            final_pipe.fit(X_train, y_train)
            if 'sampler' in final_pipe.named_steps:
                sampler_obj = final_pipe.named_steps['sampler']
                audit_sampler(sampler_obj, 'E5', m_name, seed)
                
            test_probs = final_pipe.predict_proba(X_test)[:, 1]
            
            # Cost at default threshold 0.5
            cm_05 = confusion_matrix(y_test, (test_probs >= 0.5).astype(int))
            cost_05 = (cm_05[1, 0] * cost_fn) + (cm_05[0, 1] * cost_fp)
            
            # Cost at chosen threshold (strict single evaluation)
            cm_chosen = confusion_matrix(y_test, (test_probs >= chosen_threshold).astype(int))
            cost_chosen = (cm_chosen[1, 0] * cost_fn) + (cm_chosen[0, 1] * cost_fp)
            saving = cost_05 - cost_chosen
            
            # Optimistic test-set scan (for audit comparison only)
            test_costs = []
            for t in threshold_grid:
                cm_test_t = confusion_matrix(y_test, (test_probs >= t).astype(int))
                test_costs.append((cm_test_t[1, 0] * cost_fn) + (cm_test_t[0, 1] * cost_fp))
            opt_thresh_test = threshold_grid[np.argmin(test_costs)]
            cost_optimistic = min(test_costs)
            
            rec = {
                'method': m_name,
                'seed': seed,
                'cost_at_05': cost_05,
                'cost_at_chosen_threshold': cost_chosen,
                'chosen_threshold': chosen_threshold,
                'financial_saving': saving,
                'optimistic_test_threshold': opt_thresh_test,
                'optimistic_test_cost': cost_optimistic
            }
            records.append(rec)
            logger.info(f"[E5] Seed {seed} | {m_name} | Cost(0.5): ${cost_05:,} | Cost(chosen={chosen_threshold:.4f}): ${cost_chosen:,} | Saving: ${saving:,} | Optimistic: ${cost_optimistic:,} (t={opt_thresh_test:.4f})")
            
    df_e5 = pd.DataFrame(records)
    metric_cols = ['cost_at_05', 'cost_at_chosen_threshold', 'chosen_threshold', 'financial_saving', 'optimistic_test_threshold', 'optimistic_test_cost']
    df_summary = format_mean_std(df_e5, ['method'], metric_cols)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "e5_cost_threshold.csv"), index=False)
    return df_summary, df_e5


# ==============================================================================
# EXPERIMENT 6: SCALING EFFECT (With vs Without Scaling Time & Amount)
# ==============================================================================
def run_experiment_6(df):
    logger.info("=" * 70)
    logger.info("EXPERIMENT 6: SCALING EFFECT (Scaled vs Unscaled Time and Amount)")
    logger.info("=" * 70)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    prep_scaled = ColumnTransformer(
        transformers=[('scaler', RobustScaler(), ['Time', 'Amount'])],
        remainder='passthrough'
    )
    prep_unscaled = ColumnTransformer(
        transformers=[('passthrough', 'passthrough', list(X.columns))]
    )
    
    per_seed_records = []
    
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        
        for scaling_status, prep in [('Scaled Time & Amount', prep_scaled), ('Unscaled (Raw)', prep_unscaled)]:
            models = {
                'Borderline-SMOTE + RF': Pipeline([
                    ('scaler', prep),
                    ('sampler', BorderlineSMOTE(sampling_strategy=ULB_DEFAULT_SAMPLING_RATIO, random_state=seed)),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ]),
                'BCBSMOTE + RF': Pipeline([
                    ('scaler', prep),
                    ('sampler', HybridTomekBCBSMOTE(sampling_ratio=ULB_DEFAULT_SAMPLING_RATIO, random_state=seed)),
                    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
                ])
            }
            
            for m_name, pipe in models.items():
                t0 = time.time()
                try:
                    pipe.fit(X_train, y_train)
                    if 'sampler' in pipe.named_steps:
                        sampler_obj = pipe.named_steps['sampler']
                        audit_sampler(sampler_obj, 'E6', m_name, seed, scaling_status)
                        
                    y_prob = pipe.predict_proba(X_test)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    runtime = time.time() - t0
                    m = compute_metrics(y_test, y_pred, y_prob)
                    rec = {
                        'scaling': scaling_status,
                        'method': m_name,
                        'seed': seed,
                        'runtime_s': round(runtime, 2),
                        **m
                    }
                    per_seed_records.append(rec)
                    logger.info(f"[E6] Seed={seed} | {scaling_status} | {m_name} | F1={m['f1']:.4f} | AUPRC={m['auprc']:.4f}")
                except Exception as e:
                    logger.error(f"[E6] Error in {m_name} {scaling_status} seed {seed}: {e}", exc_info=True)
                    
    df_per_seed = pd.DataFrame(per_seed_records)
    metric_cols = ['precision', 'recall', 'f1', 'auprc', 'mcc', 'runtime_s']
    df_summary = format_mean_std(df_per_seed, ['scaling', 'method'], metric_cols)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "e6_scaling.csv"), index=False)
    return df_summary, df_per_seed


# ==============================================================================
# EXPERIMENT 7: PAYSIM (Default 0.5 vs Train-Tuned Threshold, Test Fraud Counts)
# ==============================================================================
def run_experiment_7(paysim_path="archive/paysim.csv"):
    logger.info("=" * 70)
    logger.info("EXPERIMENT 7: PAYSIM BENCHMARK (Default 0.5 vs Train-Tuned Threshold)")
    logger.info("=" * 70)
    
    if not os.path.exists(paysim_path):
        logger.warning(f"[E7] File {paysim_path} not found. Skipping Experiment 7.")
        return None, None
        
    t0 = time.time()
    df_paysim_raw = pd.read_csv(paysim_path)
    logger.info(f"[E7] Loaded PaySim in {time.time()-t0:.2f}s ({len(df_paysim_raw):,} rows, {df_paysim_raw['isFraud'].sum():,} frauds)")
    
    # Stratified 10% sample
    df_sample, _ = train_test_split(
        df_paysim_raw, train_size=0.10, stratify=df_paysim_raw['isFraud'], random_state=0
    )
    logger.info(f"[E7] 10% Sample: {len(df_sample):,} rows ({df_sample['isFraud'].sum():,} frauds)")
    
    df_sample = df_sample.drop(columns=['isFlaggedFraud', 'nameOrig', 'nameDest', 'newbalanceOrig'])
    X = df_sample.drop(columns=['isFraud'])
    y = df_sample['isFraud']
    
    num_cols = ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'newbalanceDest']
    cat_cols = ['type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    per_seed_records = []
    threshold_grid = np.linspace(0.05, 0.95, 91)
    
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed
        )
        test_size = len(y_test)
        test_frauds = int(y_test.sum())
        test_genuine = int((y_test == 0).sum())
        logger.info(f"[E7] Seed {seed} Test set: {test_size:,} rows ({test_frauds} frauds, {test_genuine:,} genuine)")
        
        # Split a 20% validation fold from the training set to tune decision threshold
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.20, stratify=y_train, random_state=seed
        )
        
        models = {
            'Plain RF': Pipeline([
                ('prep', preprocessor),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ]),
            'Borderline-SMOTE': Pipeline([
                ('prep', preprocessor),
                ('sampler', BorderlineSMOTE(sampling_strategy=PAYSIM_SAMPLING_RATIO, random_state=seed)),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ]),
            'BCBSMOTE-approx': Pipeline([
                ('prep', preprocessor),
                ('sampler', HybridTomekBCBSMOTE(sampling_ratio=PAYSIM_SAMPLING_RATIO, random_state=seed)),
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed))
            ])
        }
        
        for m_name, pipe in models.items():
            t0 = time.time()
            try:
                # 1. Fit on training sub-split to tune threshold on validation fold
                pipe.fit(X_tr_sub, y_tr_sub)
                val_probs = pipe.predict_proba(X_val)[:, 1]
                val_f1s = [f1_score(y_val, (val_probs >= t).astype(int), zero_division=0) for t in threshold_grid]
                tuned_thresh = threshold_grid[np.argmax(val_f1s)]
                
                # 2. Re-fit on full training set
                pipe.fit(X_train, y_train)
                if 'sampler' in pipe.named_steps:
                    sampler_obj = pipe.named_steps['sampler']
                    audit_sampler(sampler_obj, 'E7', m_name, seed)
                    
                y_prob = pipe.predict_proba(X_test)[:, 1]
                runtime = time.time() - t0
                
                # (a) Metrics at default threshold 0.5
                y_pred_05 = (y_prob >= 0.5).astype(int)
                m_05 = compute_metrics(y_test, y_pred_05, y_prob)
                
                # (b) Metrics at training-tuned threshold
                y_pred_tuned = (y_prob >= tuned_thresh).astype(int)
                m_tuned = compute_metrics(y_test, y_pred_tuned, y_prob)
                
                rec = {
                    'dataset': 'PaySim (10% Sample)',
                    'method': m_name,
                    'seed': seed,
                    'test_total_rows': test_size,
                    'test_fraud_count': test_frauds,
                    'test_genuine_count': test_genuine,
                    'runtime_s': round(runtime, 2),
                    # Default 0.5
                    'f1_thresh_05': m_05['f1'],
                    'recall_thresh_05': m_05['recall'],
                    'precision_thresh_05': m_05['precision'],
                    'auprc': m_05['auprc'],
                    'mcc_thresh_05': m_05['mcc'],
                    'tp_05': m_05['tp'],
                    'fp_05': m_05['fp'],
                    'fn_05': m_05['fn'],
                    'tn_05': m_05['tn'],
                    # Tuned
                    'tuned_threshold': tuned_thresh,
                    'f1_tuned': m_tuned['f1'],
                    'recall_tuned': m_tuned['recall'],
                    'precision_tuned': m_tuned['precision'],
                    'mcc_tuned': m_tuned['mcc'],
                    'tp_tuned': m_tuned['tp'],
                    'fp_tuned': m_tuned['fp'],
                    'fn_tuned': m_tuned['fn'],
                    'tn_tuned': m_tuned['tn']
                }
                per_seed_records.append(rec)
                logger.info(f"[E7] Seed={seed} | {m_name} | Default 0.5: F1={m_05['f1']:.4f}, Rec={m_05['recall']:.4f}, Prec={m_05['precision']:.4f} | Tuned (t={tuned_thresh:.4f}): F1={m_tuned['f1']:.4f}, Rec={m_tuned['recall']:.4f} ({runtime:.1f}s)")
            except Exception as e:
                logger.error(f"[E7] Error in PaySim {m_name} seed {seed}: {e}", exc_info=True)
                
    df_per_seed = pd.DataFrame(per_seed_records)
    metric_cols = [
        'f1_thresh_05', 'recall_thresh_05', 'precision_thresh_05', 'auprc', 'mcc_thresh_05',
        'tp_05', 'fp_05', 'fn_05', 'tn_05',
        'tuned_threshold', 'f1_tuned', 'recall_tuned', 'precision_tuned', 'mcc_tuned',
        'tp_tuned', 'fp_tuned', 'fn_tuned', 'tn_tuned', 'runtime_s'
    ]
    df_summary = format_mean_std(df_per_seed, ['dataset', 'method'], metric_cols)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "e7_paysim.csv"), index=False)
    return df_summary, df_per_seed


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def main():
    total_start_time = time.time()
    logger.info("STARTING PHASE 3 FINAL EXPERIMENTS EXECUTION (V2)")
    
    # Load ULB Data
    df_ulb, ulb_info = load_ulb_data("archive/creditcard.csv")
    
    exp_runtimes = {}
    
    # E1
    t0 = time.time()
    run_experiment_1(df_ulb)
    exp_runtimes['E1_main_table'] = round(time.time() - t0, 2)
    
    # E2
    t0 = time.time()
    run_experiment_2(df_ulb)
    exp_runtimes['E2_label_noise'] = round(time.time() - t0, 2)
    
    # E3
    t0 = time.time()
    run_experiment_3(df_ulb)
    exp_runtimes['E3_feature_removal'] = round(time.time() - t0, 2)
    
    # E4
    t0 = time.time()
    run_experiment_4(df_ulb)
    exp_runtimes['E4_dimensionality'] = round(time.time() - t0, 2)
    
    # E5
    t0 = time.time()
    run_experiment_5(df_ulb)
    exp_runtimes['E5_cost_threshold'] = round(time.time() - t0, 2)
    
    # E6
    t0 = time.time()
    run_experiment_6(df_ulb)
    exp_runtimes['E6_scaling'] = round(time.time() - t0, 2)
    
    # E7
    t0 = time.time()
    run_experiment_7("archive/paysim.csv")
    exp_runtimes['E7_paysim'] = round(time.time() - t0, 2)
    
    total_runtime = round(time.time() - total_start_time, 2)
    
    # Comprehensive run_info.json
    run_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'packages': {
            'sklearn': '1.7.2',
            'imblearn': '0.14.2',
            'xgboost': '3.2.0',
            'pandas': pd.__version__,
            'numpy': np.__version__
        },
        'seeds': SEEDS,
        'dataset_info': ulb_info,
        'experiment_runtimes_seconds': exp_runtimes,
        'total_runtime_seconds': total_runtime,
        'total_runtime_minutes': round(total_runtime / 60, 2),
        'resampling_audit_log': RESAMPLING_AUDIT,
        'audit_warnings_zero_synthetic': AUDIT_WARNINGS
    }
    
    with open(os.path.join(OUTPUT_DIR, "run_info.json"), 'w') as f:
        json.dump(run_info, f, indent=4)
        
    logger.info(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY in {total_runtime:.2f}s ({total_runtime/60:.2f} min).")
    logger.info(f"Outputs written to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
