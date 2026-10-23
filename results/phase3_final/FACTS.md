# PART A: FACTS TO CHECK (Verification & Code Audit)

This document answers all nine audit questions in Part A based on the codebase (`src/preprocessing.py`, `src/sampling.py`, `src/model.py`, `scripts/run_experiments.py`, `train.py`, and `results/`). Every statement is verified with exact file paths, line numbers, and empirical code executions.

---

## 1. DUPLICATES: ULB Dataset Row Counts & Deduplication

### File & Line References
- **`src/preprocessing.py`**: Lines 86–90 (`clean_and_verify_dataset`)
- **`scripts/run_experiments.py`**: Lines 54–77 (`load_and_clean_data`)

### Empirical Verification & Breakdown
The raw European Cardholders dataset (`archive/creditcard.csv`) contains:
- **Raw Rows**: 284,807 transactions (492 frauds, 284,315 genuine).

When deduplication is performed:

| Deduplication Strategy | Rows Before | Duplicates Removed | Frauds Removed | Genuine Removed | Rows Left | Frauds Left | Genuine Left |
|---|---|---|---|---|---|---|---|
| **Exact Duplicates (all 31 cols incl. `Time`)** | 284,807 | **1,081** | 19 | 1,062 | **283,726** | 473 | 283,253 |
| **Duplicates Ignoring `Time` (30 cols: V1–V28, Amount, Class)** | 284,807 | **9,144** | 19 | 9,125 | **275,663** | 473 | 275,190 |

### Explanation of Why 9,144 Rows Were Removed
1. In `scripts/run_experiments.py` (Line 4), the dataset was loaded from OpenML (dataset ID 1597). On OpenML, the European cardholders dataset is distributed **without the `Time` column** (only 30 columns: V1–V28, Amount, Class).
2. Calling `df_raw.drop_duplicates()` on the 30-column dataset in `scripts/run_experiments.py` (Line 62) therefore checks for duplicates **ignoring `Time`**.
3. Across V1–V28, Amount, and Class, there are **9,144 duplicate rows**. Exactly **19 are fraud duplicates** (leaving 473 frauds), and **9,125 are genuine duplicates** (leaving 275,190 genuine rows).
4. When `Time` is included, only **1,081 rows** have the exact same transaction timestamp down to the second. The remaining 8,063 transactions share identical PCA features and Amount but occurred at different elapsed seconds.
5. In both cases, **exactly 19 duplicate fraud rows are removed**, leaving **473 frauds** in the cleaned dataset.

---

## 2. LABEL NOISE: Injection Mechanism, Classes, Seeds & Bug Analysis

### File & Line References
- **`scripts/run_experiments.py`**: Lines 320–356 (`run_experiment_4_stress_testing`, sub-section 4A)
- **`src/sampling.py`**: Lines 178–186 (`BCBSMOTEOversampler.fit_resample`)

### Mechanism Breakdown
- **Which rows**: **Training rows only** (`y_tr_noisy = y_train.copy()`, Line 328). Test set labels were untouched.
- **Which classes**: **Both classes** (symmetric noise). Line 331 executes `y_tr_noisy[flips] = 1 - y_tr_noisy[flips]`, flipping genuine to fraud ($0 \to 1$) and fraud to genuine ($1 \to 0$).
- **Random seed**: `rng = np.random.RandomState(42)` initialized at Line 321 **outside the loop**, so each noise level drew sequentially from the generator state.
- **Actual labels flipped** (on training set of 220,530 rows: 220,152 genuine, 378 frauds):
  - **1% Noise**: 2,205 labels flipped: **2,198 genuine $\to$ fraud**, **7 frauds $\to$ genuine**. Resulting train labels: 217,961 genuine, 2,569 frauds.
  - **5% Noise**: 11,026 labels flipped: **11,009 genuine $\to$ fraud**, **17 frauds $\to$ genuine**. Resulting train labels: 209,160 genuine, 11,370 frauds.
  - **10% Noise**: 22,053 labels flipped: **22,011 genuine $\to$ fraud**, **42 frauds $\to$ genuine**. Resulting train labels: 198,183 genuine, 22,347 frauds.

### Root Cause of the Bug ($F_1 = 0.555$ at 1% vs $F_1 = 0.802$ at 10%)
Two interrelated bugs in `scripts/run_experiments.py` and `src/sampling.py` explain this counter-intuitive result:

1. **Massive Synthetic Contamination at 1% Noise**:
   In `src/sampling.py` (Lines 178–180):
   $$\text{target\_min\_count} = \text{int}(n_{\text{maj}} \times \text{sampling\_ratio})$$
   $$n_{\text{to\_generate}} = \max(0, \text{target\_min\_count} - n_{\text{min}})$$
   At 1% noise with `sampling_ratio = 0.05`, $\text{target\_min\_count} = 217,961 \times 0.05 = 10,898$. Since $n_{\text{min}} = 2,569$, BCBSMOTE generated **8,329 synthetic minority samples**. BIRCH clustered around the 2,198 falsely labeled genuine points, creating dense synthetic fraud clusters in the heart of normal transaction space. This caused False Positives to skyrocket and precision to crash to **45.3%**, collapsing $F_1$ to **0.555**.

2. **BCBSMOTE Completely Deactivated at 5% and 10% Noise**:
   At 5% noise, flipping 5% of the predominantly genuine dataset created **11,370** fraud labels. But `target_min_count` was only $209,160 \times 0.05 = 10,458$.
   Because $n_{\text{min}} (11,370) > \text{target\_min\_count} (10,458)$:
   $$n_{\text{to\_generate}} = \max(0, 10,458 - 11,370) = 0$$
   Similarly, at 10% noise ($n_{\text{min}} = 22,347 > 9,909$), $n_{\text{to\_generate}} = 0$.
   **BCBSMOTE generated exactly 0 synthetic samples at 5% and 10% noise.**
   
3. **Random Forest Robustness to Diffuse Noise**:
   Without oversampling, Random Forest was trained on raw label noise uniformly scattered across the vast majority space. Decision trees (especially with `max_depth=10`) averaged out diffuse noise across leaf nodes, maintaining high precision (86.6%) and achieving $F_1 = 0.802$.
   
4. **Lack of Control**:
   `scripts/run_experiments.py` lacked a plain Random Forest control under identical noise, obscuring the fact that BCBSMOTE was harming performance at 1% and non-operational at 10%.

---

## 3. COST ANALYSIS: Unit Costs, Threshold Selection & Run Details

### File & Line References
- **`scripts/run_experiments.py`**: Lines 169–199 (`run_experiment_1_baseline_vs_bcbsmote`)

### Parameters and Audit Findings
- **Cost of a Missed Fraud (False Negative, FN)**: **500** (`cost_fn = 500`, Line 171).
- **Cost of a False Alarm (False Positive, FP)**: **10** (`cost_fp = 10`, Line 171).
- **How Threshold 0.14 Was Chosen**:
  - In Line 174: `cm = confusion_matrix(y_test, (probs_bc >= t).astype(int))`
  - The threshold was determined by **scanning directly across the test set** (`y_test`) over 100 threshold steps from 0.01 to 0.99.
  - **Audit Verdict**: This represents **test-set data leakage (optimism bias)**. The threshold was not chosen on training out-of-fold predictions. The optimal threshold found on `y_test` was **0.1387** (rounded to 0.14).
- **Model and Run Producing Costs 12,590 and 8,700**:
  - Model: `rf_bc` (Hybrid BCBSMOTE + Random Forest, seed=42) from Experiment 1.
  - **Default Cost ($12,590)**: At threshold $t = 0.5049$ (threshold index 50):
    $$\text{Cost} = (25 \text{ FN} \times 500) + (9 \text{ FP} \times 10) = 12,500 + 90 = \$12,590$$
  - **Optimized Cost ($8,700)**: At test-scanned threshold $t = 0.1387$:
    $$\text{Cost} = (17 \text{ FN} \times 500) + (20 \text{ FP} \times 10) = 8,500 + 200 = \$8,700$$
    Reported financial saving: $\$12,590 - \$8,700 = \$3,890$.

---

## 4. TOP 3 FEATURES (V14, V17, V12): Selection Methodology

### File & Line References
- **`scripts/run_experiments.py`**: Lines 357–364 (`run_experiment_4_stress_testing`, sub-section 4B)

### Audit Findings
- Lines 358–359:
  ```python
  top_cols = ['V14', 'V17', 'V12']
  col_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
  keep_idx = [i for i, c in enumerate(col_names) if c not in top_cols]
  ```
- **Audit Verdict**: Features V14, V17, and V12 were **hardcoded**. They were **not dynamically computed from the training fold** during script execution. They were chosen based on standard exploratory correlation analyses common in the literature for the Kaggle/ULB dataset.

---

## 5. EXACT ALGORITHM SETTINGS & PARAMETERS

### File & Line References
- **`src/sampling.py`**: Lines 19–116 (`TomekLinksUndersampler`), Lines 119–293 (`BCBSMOTEOversampler`), Lines 295–321 (`HybridTomekBCBSMOTE`)
- **`scripts/run_experiments.py`**: Lines 108–133, Lines 208–226, Lines 266–280

### Parameter Specifications

1. **BIRCH Clustering Settings (`src/sampling.py`, Lines 131–133, 192–196)**:
   - `threshold`: **0.5**
   - `branching_factor`: **50**
   - `n_clusters`: **None** (determines subclusters automatically)
   - `cluster_size`: **5** (clusters with $< 5$ minority points are treated as noise/outliers)

2. **Borderline-SMOTE Settings (`src/sampling.py`, Lines 131–133, 219–277)**:
   - `k_neighbors`: **5** (nearest neighbors within the cluster for interpolation)
   - `m_neighbors`: **5** (nearest neighbors across dataset to categorize Danger / Safe / Noise)

3. **SAMPLING_RATIO**:
   - In `src/sampling.py` (Line 133, Line 303): default is **0.02** (paper specification).
   - In `scripts/run_experiments.py` (Lines 112, 212, 222, 335, 365): explicitly overridden to **0.05**.

4. **Tomek Links Strategy (`src/sampling.py`, Lines 19–116)**:
   - Uses custom `TomekLinksUndersampler`.
   - Standardizes features using `StandardScaler` (Line 56).
   - Finds mutual 1-nearest neighbor pairs between majority ($y=0$) and minority ($y=1$).
   - **Removes majority instances only** (Line 109: `keep_mask[self.removed_indices_] = False`).

5. **XGBoost Parameters (`scripts/run_experiments.py`, Line 269)**:
   - `n_estimators`: 50
   - `max_depth`: 6
   - `learning_rate`: 0.1
   - `eval_metric`: `'logloss'`
   - `random_state`: 42
   - `scale_pos_weight`: **NOT USED** (defaulted to 1.0; line 269 omitted `scale_pos_weight`).
   - Training input: Trained on `X_res, y_res` (data **already resampled by BCBSMOTE**).

6. **Balanced Random Forest Parameters (`scripts/run_experiments.py`, Lines 270, 276–279)**:
   - `n_estimators`: 40
   - `random_state`: 42
   - **Resampling status**: Trained on `X_train_s, y_train` (the **un-resampled** training set, Line 277). It was **not** run on data resampled by BCBSMOTE.

---

## 6. FEATURE SCALING: Placement, Pipeline & Columns

### File & Line References
- **`scripts/run_experiments.py`**: Lines 416–420 (`main`)
- **`src/sampling.py`**: Lines 56, 189

### Audit Findings
- **Placement**: Scaling was performed **manually outside** of an `imblearn.pipeline.Pipeline` or `sklearn.pipeline.Pipeline`.
- **Fitted on Train Only?**: **YES**. Line 417 explicitly called `scaler = RobustScaler().fit(X_train)`, followed by `X_train_s = scaler.transform(X_train)` and `X_test_s = scaler.transform(X_test)`. Test data was never used for fitting.
- **Columns Scaled**:
  - In `scripts/run_experiments.py`, `RobustScaler` was applied to **all 29 columns** present in that file (V1–V28 and Amount).
  - The `Time` column was absent in `scripts/run_experiments.py` (due to using the 30-column OpenML version).
  - Inside `src/sampling.py`, `StandardScaler` was additionally applied internally to `X_maj` and `X_min` during Tomek links distance computations (Line 56) and BIRCH clustering (Line 189).

---

## 7. DISCREPANCIES ACROSS TABLES: Root Cause Analysis

### File & Line References
- **`scripts/run_experiments.py`**: Line 87 (Exp 1), Line 224 (Exp 2), Line 338 (Exp 4)

### Audit Findings
In the previous report assets:
- **Table 1 Baseline RF**: $F_1 = 0.812$ ($0.8118$), Recall = $0.726$, Precision = $0.920$
- **Table 2 Baseline RF**: $F_1 = 0.805$ ($0.8047$), Recall = $0.716$, Precision = $0.919$
- **Table 4A Baseline RF (0% Noise)**: $F_1 = 0.798$ ($0.7977$), Recall = $0.726$, Precision = $0.885$

### Cause: Inconsistent Classifier Hyperparameters Across Functions
The seed (`random_state=42`) and the 80/20 train/test split were identical across all experiments. The discrepancies occurred because different functions instantiated Random Forest with **different hyperparameters**:
- Experiment 1 (Line 87): `RandomForestClassifier(n_estimators=40, max_depth=14)`
- Experiment 2 (Line 224): `RandomForestClassifier(n_estimators=30, max_depth=12)`
- Experiment 4A (Line 338): `RandomForestClassifier(n_estimators=30, max_depth=10)`

Each shallower depth reduced tree expressive capacity, shifting the baseline $F_1$ from 0.812 to 0.805 and 0.798.

---

## 8. TEST SET SIZES & CLASS COUNTS (ULB DATASET)

For a stratified 80/20 train/test split across the ULB dataset variations:

| Dataset Version | Split | Total Records | Genuine (Class 0) | Fraud (Class 1) | Fraud Ratio |
|---|---|---|---|---|---|
| **Exact Duplicates Removed (1,081 removed)** | **Train (80%)** | 226,980 | 226,602 | 378 | 0.1665% |
| | **Test (20%)** | 56,746 | 56,651 | 95 | 0.1674% |
| | **Total** | 283,726 | 283,253 | 473 | 0.1667% |
| **Duplicates Ignoring Time Removed (9,144 removed)** | **Train (80%)** | 220,530 | 220,152 | 378 | 0.1714% |
| | **Test (20%)** | 55,133 | 55,038 | 95 | 0.1723% |
| | **Total** | 275,663 | 275,190 | 473 | 0.1716% |
| **Raw Dataset (No deduplication)** | **Train (80%)** | 227,845 | 227,451 | 394 | 0.1729% |
| | **Test (20%)** | 56,962 | 56,864 | 98 | 0.1720% |
| | **Total** | 284,807 | 284,315 | 492 | 0.1727% |

---

## 9. PAYSIM BASELINE METRICS: Reproduction Origin

### File & Line References
- **`results/paysim_baseline/evaluation_metrics_summary.csv`**: Lines 1–2
- **`train.py`**: Lines 47–49, 124–129, 143–175

### Audit Findings
- **Metrics**: Accuracy = 99.94%, $F_1\text{-score} = 75.79\%$, Recall = $69.23\%$, Precision = $83.72\%$, AUPRC = $75.13\%$, ROC-AUC = $99.29\%$.
- **Confusion Matrix**: $\text{TP} = 36$, $\text{FP} = 7$, $\text{TN} = 39,941$, $\text{FN} = 16$.
- **Sample Size**: Total test rows $= 36 + 7 + 39,941 + 16 = 40,000$. With a 20% test split, the run was executed on a **subsample of 200,000 transactions** (`sample_size=200000`, `train.py` Line 48), not the full 6.36M dataset.
- **Split & Seed**: Stratified 80/20 train/test split with `random_state=42` (`train.py` Line 128).
- **Method**: **Proposed Hybrid BCBSMOTE + Random Forest**:
  - Sampler: `HybridTomekBCBSMOTE(cluster_size=5, sampling_ratio=0.02, random_state=42)` (`train.py` Lines 143–147)
  - Classifier: `RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)` (`train.py` Lines 150–156)
  - Threshold Optimization: Decision threshold was optimized on a 100,000 training subsample using `optimize_threshold(..., metric='f1')` (`train.py` Lines 164–169), yielding an optimal threshold of **0.3862** (Line 199).
- **Source File**: Saved directly to `results/paysim_baseline/evaluation_metrics_summary.csv`.
