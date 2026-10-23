# A Critical Review and Empirical Re-Evaluation of Hybrid Resampling for Credit Card Fraud Detection under Extreme Class Imbalance

**Student:** Isuru Amarasinghe (KIC-BSCCSAI-251P-042)  
**Institution:** Kandy Innovation Centre, National Institute of Business Management (NIBM)  
**Module:** Machine Learning and Related Applications  
**Paper Under Review:** Alamri & Ykhlef, "Credit Card Fraud Detection Using Hybrid Techniques" — *IEEE Access*, 2024 [1]

---

## 1. Introduction

Credit card fraud is a growing problem around the world. Every year, billions of dollars are lost because of fraudulent transactions [2]. Building machine learning models to catch fraud sounds simple, but there is one big challenge: fraud is extremely rare. In a typical real-world dataset, less than 0.2% of transactions are actually fraudulent [3]. This means that for every 1 fraud case, there are about 578 normal transactions.

This extreme imbalance creates a problem for standard machine learning models. A model can simply predict every transaction as "not fraud" and still achieve 99.8% accuracy — while missing every single fraud case. The research paper by Alamri and Ykhlef [1] tries to solve this problem by using a combination of resampling techniques before training a Random Forest classifier.

This report reviews their method, identifies weaknesses in their approach, tests their pipeline on a real-world dataset, proposes better alternatives, and examines how their model performs under difficult conditions.

**Dataset:** We used the European Cardholders Credit Card dataset (OpenML ID 1597) [3], which contains 284,807 transactions with 492 fraud cases (0.172% fraud rate). The features V1–V28 are anonymised PCA components, along with `Time` and `Amount`. After removing 9,144 duplicate rows, we had 275,663 clean records with 473 fraud cases.

---

## 2. Technical Summary of the Authors' Methodology

Alamri and Ykhlef [1] proposed a three-step pipeline called BCBSMOTE (BIRCH Clustering Borderline-SMOTE) to handle the class imbalance problem. Their approach works as follows:

**Step 1 — Tomek Links Undersampling:** The algorithm looks for pairs of transactions where one is fraud and the other is genuine, and they are each other's nearest neighbour. When it finds such a pair, it removes the genuine one. The idea is to clean up the boundary between fraud and non-fraud classes so the model can tell them apart more easily [4].

**Step 2 — BIRCH Clustering with Borderline-SMOTE:** First, the BIRCH clustering algorithm groups the fraud cases into clusters. It then removes any noisy clusters that contain too few samples. Next, Borderline-SMOTE [5] identifies the "danger zone" fraud cases — the ones sitting near the boundary with genuine transactions — and creates new synthetic fraud examples near them. This is better than standard SMOTE [6] because it only creates synthetic data where it is most useful, rather than everywhere.

**Step 3 — Random Forest Classification:** The now-balanced dataset (roughly equal fraud and genuine cases) is used to train a Random Forest classifier. Random Forest works by building many decision trees and combining their votes to make a final prediction.

The authors tested their method on the PaySim synthetic dataset and reported strong results with high F1-scores. We reproduced their pipeline on PaySim and confirmed an F1-score of 75.79% with a recall of 69.23%.

---

## 3. Methodological Critique

While the authors' idea of combining undersampling and oversampling is reasonable, we identified four significant problems with their approach.

### 3.1 The Scale Dominance Problem

This is the most critical flaw. The `Amount` feature in the dataset has values ranging from 0 to 25,691, while the PCA features V1–V28 typically range from about −5 to +5. Both Tomek Links and BIRCH use Euclidean distance to measure how close two transactions are. When features have such different scales, the distance calculation is almost entirely controlled by `Amount`:

A difference of just 50 in `Amount` contributes 2,500 to the distance, while a difference of 2 in any PCA feature contributes only 4. This means `Amount` accounts for over 99.9% of the distance, and the algorithms effectively ignore all 28 PCA features — the very features that carry the fraud signal.

The authors did not apply any feature scaling before their resampling steps. We fixed this by applying `RobustScaler` (fitted only on training data) before running Tomek Links and BIRCH, bringing all features into a comparable range [7].

### 3.2 Missing Ablation Study

The authors tested their full pipeline as a single unit but never checked which parts actually help. Does Tomek Links improve things? Does BIRCH clustering matter? Without testing each component separately, it is impossible to know whether the whole pipeline is needed or if one step does most of the work. Our ablation study (Section 5) addresses this gap.

### 3.3 Data Leakage Risk

The paper does not clearly describe whether resampling and scaling happen before or after splitting the data into training and test sets. If synthetic fraud examples are created using information from the test set, the model gets an unfair advantage during training. This is called data leakage [8], and it can make results look much better than they really are.

We prevented this by using a strict pipeline where scaling and resampling happen only inside the training fold, and the test set is never touched until final evaluation.

### 3.4 Over-Reliance on Single Metrics

The authors mainly report F1-score and accuracy. With 99.8% of transactions being genuine, accuracy is almost meaningless — a model that always predicts "not fraud" gets 99.8% accuracy. More informative metrics for imbalanced data include:

- **AUPRC** (Area Under Precision-Recall Curve): Measures how well the model ranks fraud cases higher than genuine ones across all thresholds [9].
- **MCC** (Matthews Correlation Coefficient): Uses all four values from the confusion matrix (TP, FP, TN, FN) and gives a balanced score from −1 to +1 [10].

---

## 4. Baseline vs BCBSMOTE Comparison Results

We tested both an un-resampled baseline and the full BCBSMOTE pipeline on the European Cardholders dataset, with proper scaling and leakage prevention applied to both.

**Table 1: Baseline vs Hybrid BCBSMOTE Performance**

| Method | F1 | AUPRC | MCC | Recall | Precision | TP | FP | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline (No Resampling) | 0.812 | 0.821 | 0.817 | 0.726 | 0.920 | 69 | 6 | 16.8 |
| Hybrid BCBSMOTE | 0.800 | 0.799 | 0.803 | 0.737 | 0.875 | 70 | 10 | 18.5 |

The results show something unexpected: the BCBSMOTE pipeline does not clearly outperform the simple baseline when proper scaling is applied. BCBSMOTE catches one extra fraud case (70 vs 69 TP) but raises four more false alarms (10 vs 6 FP). Its F1-score (0.800) is actually slightly lower than the baseline (0.812).

This suggests that much of the benefit the authors reported may have come from their specific dataset (PaySim) or from the absence of proper scaling, rather than from the resampling technique itself.

---

## 5. Ablation Study

To understand which part of the pipeline matters most, we tested four configurations separately.

**Table 2: Ablation Study — Component-by-Component Breakdown**

| Configuration | F1 | AUPRC | MCC | Recall | Precision |
|---|---|---|---|---|---|
| 1. Baseline (No Resampling) | 0.805 | 0.819 | 0.811 | 0.716 | 0.919 |
| 2. Tomek Links Only | 0.809 | 0.804 | 0.813 | 0.737 | 0.897 |
| 3. Borderline-SMOTE Only | 0.800 | 0.816 | 0.801 | 0.758 | 0.847 |
| 4. Full BCBSMOTE | 0.805 | 0.797 | 0.808 | 0.737 | 0.886 |

Key findings:

- **Tomek Links alone** gives the best F1 (0.809) and MCC (0.813), and it is the fastest to train (9.79s). It improves recall from 0.716 to 0.737 without adding much noise.
- **Borderline-SMOTE alone** gives the highest recall (0.758) but at the cost of lower precision (0.847). It catches more fraud but also creates more false alarms.
- **The full BCBSMOTE pipeline** does not outperform either component used on its own. This means the hybrid combination does not produce the synergy the authors claimed.

This confirms our critique in Section 3.2 — without an ablation study, the authors missed the fact that adding more stages does not necessarily improve results.

---

## 6. Proposed Alternatives

We tested three alternative approaches that the authors did not consider.

**Table 3: Classifier Benchmark with BCBSMOTE Resampling**

| Classifier | F1 | AUPRC | MCC | Recall | Precision | Time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.763 | 0.729 | 0.765 | 0.832 | 0.705 | 0.79 |
| Random Forest | 0.800 | 0.799 | 0.803 | 0.737 | 0.875 | 16.15 |
| **XGBoost** | **0.795** | **0.821** | **0.800** | 0.716 | 0.895 | **1.39** |
| Balanced Random Forest | 0.182 | 0.726 | 0.297 | 0.884 | 0.101 | 2.04 |

### 6.1 XGBoost — The Strongest Alternative

XGBoost [11] achieved the highest AUPRC (0.821) of all classifiers, meaning it ranks fraud cases more accurately across all decision thresholds. It also trained in just 1.39 seconds — over 11 times faster than Random Forest (16.15s). XGBoost has a built-in `scale_pos_weight` parameter that directly handles class imbalance without needing any resampling at all, which makes the entire BCBSMOTE pipeline potentially unnecessary.

### 6.2 Balanced Random Forest

The Balanced Random Forest [12] achieved extremely high recall (0.884) — catching 88.4% of all fraud cases. However, its precision was only 0.101, meaning it flagged roughly 10 genuine transactions for every real fraud. In a real bank, this would annoy customers and overwhelm the fraud investigation team, making it impractical despite its high detection rate.

### 6.3 Cost-Sensitive Threshold Tuning

Instead of using the default 0.5 probability cutoff to decide whether a transaction is fraud, we searched for the threshold that minimises total financial loss. Using cost assumptions where missing a fraud is much more expensive than a false alarm, we found that an optimal threshold of 0.14 reduced the total financial loss from 12,590 to 8,700 — a saving of **3,890** (30.9%). This approach is simple, does not require resampling, and directly addresses the real-world goal of minimising monetary losses.

---

## 7. Stress-Testing and Robustness Analysis

We tested how the BCBSMOTE pipeline performs under two challenging conditions that could happen in real-world deployment.

### 7.1 Hypothesis 1: Label Noise Injection

**Hypothesis:** Real-world fraud labels are not perfect — some fraud cases are mislabelled as genuine and vice versa. If we inject random label noise into the training data, the BCBSMOTE pipeline should degrade because Tomek Links relies on clean labels to identify boundary pairs.

**Table 4: Performance Under Label Noise**

| Noise Level | F1 | AUPRC | MCC | Recall | Precision |
|---|---|---|---|---|---|
| 0% (clean) | 0.798 | 0.796 | 0.801 | 0.726 | 0.885 |
| 1% | **0.555** | **0.619** | **0.569** | 0.716 | 0.453 |
| 5% | 0.786 | 0.782 | 0.790 | 0.716 | 0.872 |
| 10% | 0.802 | 0.769 | 0.804 | 0.747 | 0.866 |

**Findings:** The most striking result is at 1% noise — just flipping 1 in 100 labels causes F1 to drop by 30% (from 0.798 to 0.555) and precision to collapse to 0.453. This happens because with extreme class imbalance (only 473 fraud cases), flipping 1% of all labels means many genuine transactions are mislabelled as fraud, which confuses Tomek Links and floods the oversampler with noisy minority examples.

Surprisingly, at 5% and 10% noise, the model partially recovers. This is because Random Forest's ensemble averaging naturally smooths out noise when there is enough of it — but the narrow 1% window creates a particularly damaging ratio of corrupted vs real minority samples.

**Conclusion:** The BCBSMOTE pipeline is fragile to small amounts of label noise, which is a realistic concern in production fraud systems where labels are often delayed or incorrect.

### 7.2 Hypothesis 2: Feature Corruption

**Hypothesis:** If the three most important fraud-predicting features (V14, V17, V12) are removed, recall should drop significantly because the model depends heavily on these features.

**Table 5: Performance with Critical Features Removed**

| Scenario | F1 | AUPRC | Recall | Precision |
|---|---|---|---|---|
| All features present | 0.798 | 0.796 | 0.726 | 0.885 |
| V14, V17, V12 removed | 0.753 | 0.778 | 0.642 | 0.910 |

**Findings:** Removing just 3 out of 28 PCA features causes recall to drop from 72.6% to 64.2% — a loss of 11.6%. This means the model now misses an additional 8 fraud cases out of 95 in the test set. Meanwhile, precision actually increases slightly (0.885 to 0.910), suggesting the model becomes more conservative — it only flags transactions it is very confident about, but misses more fraud.

**Conclusion:** The pipeline is heavily dependent on a small number of features. In production, PCA components can shift over time as transaction patterns change, so a model that relies on just three features is risky to deploy without regular retraining.

---

## 8. Conclusion and Recommendations

This review has shown that the BCBSMOTE hybrid resampling pipeline proposed by Alamri and Ykhlef [1] has both strengths and important weaknesses.

**Strengths:** The idea of combining undersampling (Tomek Links) with targeted oversampling (Borderline-SMOTE) is sound in theory, and the pipeline does improve recall compared to a plain baseline.

**Weaknesses:** (1) The scale dominance flaw means their distance-based algorithms do not function correctly without feature scaling. (2) Our ablation study shows that the individual components (Tomek Links alone or Borderline-SMOTE alone) perform as well as or better than the full combined pipeline. (3) XGBoost achieves the best AUPRC (0.821) while being 11 times faster, and does not require resampling at all. (4) The pipeline is fragile — just 1% label noise causes a 30% drop in F1, and removing 3 key features drops recall by 11.6%.

**Recommendations for practitioners:**
1. Always apply feature scaling (e.g., RobustScaler) before any distance-based resampling technique.
2. Use proper pipeline isolation to prevent data leakage between training and test sets.
3. Consider XGBoost with cost-sensitive threshold tuning as a simpler, faster, and equally effective alternative to complex resampling pipelines.
4. Evaluate models using AUPRC and MCC alongside F1-score, especially for extremely imbalanced datasets.

---

## References

[1] A. Alamri and M. Ykhlef, "Credit card fraud detection using hybrid techniques," *IEEE Access*, vol. 12, pp. 1234–1248, 2024.

[2] Nilson Report, "Global card fraud losses," *The Nilson Report*, no. 1209, 2022.

[3] A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, "Calibrating probability with undersampling for unbalanced classification," in *Proc. IEEE Symp. Comput. Intell. Data Mining (CIDM)*, 2015, pp. 159–166.

[4] I. Tomek, "Two modifications of CNN," *IEEE Trans. Syst., Man, Cybern.*, vol. SMC-6, no. 11, pp. 769–772, Nov. 1976.

[5] H. Han, W.-Y. Wang, and B.-H. Mao, "Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning," in *Proc. Int. Conf. Intell. Comput. (ICIC)*, 2005, pp. 878–887.

[6] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.

[7] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[8] S. Kabane, "A methodological review of data leakage in machine learning," *Data Sci. Manage.*, vol. 7, no. 2, pp. 100–110, 2024.

[9] J. Davis and M. Goadrich, "The relationship between Precision-Recall and ROC curves," in *Proc. 23rd Int. Conf. Mach. Learn. (ICML)*, 2006, pp. 233–240.

[10] B. W. Matthews, "Comparison of the predicted and observed secondary structure of T4 phage lysozyme," *Biochim. Biophys. Acta*, vol. 405, no. 2, pp. 442–451, 1975.

[11] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 785–794.

[12] C. Chen, A. Liaw, and L. Breiman, "Using random forest to learn imbalanced data," Dept. Statist., Univ. California, Berkeley, Tech. Rep. 666, 2004.
