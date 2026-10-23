# Phase 3: Critical Review and Stress-Testing of Hybrid Resampling for Imbalanced Fraud Detection

Isuru Amarasinghe (KIC-BSCCSAI-251P-042) · Kandy Innovation Centre, NIBM · Machine Learning and Related Applications

Paper under review: M. Alamri and M. Ykhlef, “Hybrid Undersampling and Oversampling for Handling Imbalanced Credit Card Data,” *IEEE Access*, vol. 12, pp. 14050–14060, 2024 \[1\].

## Summary of Phases 1 and 2

**Phase 1 (topic and paper).** The paper proposes BCBSMOTE, a three-step method for very imbalanced data: Tomek links, BIRCH clustering and Borderline-SMOTE, followed by a Random Forest. The authors use the PaySim dataset \[2\]. It is a simulated mobile-money dataset with 6,362,620 transactions, of which 8,213 are fraud (about 0.13%). They split the data 80/20 first and resample the training set only. They compare BCBSMOTE with SMOTE, Borderline-SMOTE, ADASYN, SMOTE-Tomek and SMOTE-ENN, all with Random Forest, and report accuracy, precision, recall, F1 and AUPRC. Their best F1 is 85.20%. Note: my first Phase 1 text described the ULB credit card dataset by mistake, because I had read only the abstract. After reading the full paper I corrected this. The paper uses PaySim, and ULB is used in this work only as a second dataset.

**Phase 2 (baseline implementation).** I rebuilt the pipeline on PaySim and obtained F1 = 75.79% and recall = 69.23%. This is lower than the paper’s 85.20% F1 and 89.53% recall. I cannot name one exact cause. Likely reasons are different sampling and settings, and the fact that the paper does not fully describe its BIRCH step, so my BCBSMOTE is an approximation. I then repeated the experiments on a second, similar open dataset, the European cardholders dataset (OpenML ID 1597) \[3\]. It has 284,807 transactions and 492 frauds (0.172%). After removing duplicate rows (9,144 rows that duplicate another row when Time is ignored) there are 275,663 records with 473 frauds. On this dataset I scaled the features with RobustScaler, fitted on the training data only, and I did all resampling inside the training part only. The test set holds about 95 frauds. Results from this dataset cannot be compared directly with the paper’s PaySim numbers.

## a. Technical Summary

**Problem.** When fraud is about 0.1–0.2% of transactions, a model that always says “genuine” is about 99.8% accurate and finds no fraud. Accuracy is therefore not a useful measure here.

**Step 1: Tomek links.** A Tomek link is a pair of points from different classes that are each other’s nearest neighbour. The method removes the genuine point of each pair, which cleans the class border \[4\].

**Step 2: BIRCH and Borderline-SMOTE.** BIRCH \[5\] groups the fraud points into clusters, and very small clusters are treated as noise. Borderline-SMOTE \[6\] then finds “danger” fraud points. For a fraud point with k nearest neighbours, of which m′ are genuine, the point is in danger when k/2 ≤ m′ < k. New points are made between a danger point x and a nearby fraud point x′ using x\_new = x + λ(x′ − x), where λ is a random number between 0 and 1 \[7\].

**Step 3: Random Forest.** A Random Forest trains many decision trees on random samples of the data and combines their votes \[8\].

**What the paper reports.** Best F1 is 85.20% (recall 89.53%, precision 81.27%, AUPRC 72.77%), against 84.67% F1 for Borderline-SMOTE alone. In my implementation the BIRCH settings (branching factor 50, threshold 0.5) are my own choices, because the paper does not give them.

## b. Methodological Critique

The idea of cleaning the border and then adding fraud examples only where they are most useful is reasonable. The paper also does several things correctly: it splits before resampling, it says accuracy is unsuitable, and it reports AUPRC. These are the weaknesses I found.

### b1. One run, and a very small gain

The paper reports one split, with no repeated runs, no random seeds, no cross-validation and no standard deviation. Its gain over Borderline-SMOTE is 0.53 F1 points (85.20% against 84.67%). Without repeats, we cannot say this gap is more than random variation. My own runs show why this matters: the same baseline gave F1 = 0.812 in one table and 0.805 in another, so run-to-run noise is about 0.01 F1, as large as many gaps in this field.

### b2. The Tomek step does almost nothing

Tomek links removed 954 of 5,083,526 majority samples (about 0.02%). So the “hybrid” method is almost entirely oversampling.

### b3. No plain baseline and no ablation

The paper never shows a Random Forest with no resampling, and it never tests the stages one by one. Without these, we cannot tell whether resampling helps, or which stage helps.

### b4. Recall is lower than the competitors

The proposed method has recall of 89.53%, while SMOTE, ADASYN and the other baselines are at about 97–98%. Its F1 is higher only because its precision is higher. For fraud, the missed cases are often the most costly, so this trade-off needs discussion, and the paper offers none.

### b5. Missing measures

There is no MCC, and there is no decision-threshold or cost analysis. For a bank, the cost of a missed fraud is very different from the cost of a false alarm.

### b6. Table 5 looks mislabelled

The numbers give the reported recall (89.53%) and precision (81.27%) only if fraud is the positive class (TP = 1,471, FN = 172, FP = 339). The table labels appear to be swapped.

### b7. The dataset is not credit card data

PaySim simulates mobile-money transfers, yet the paper calls it synthetic credit card data. It is also simulated, so results may not carry over to real banks.

### b8. Scaling is not mentioned (a risk, not a proven error)

Tomek links, BIRCH and SMOTE all use Euclidean distance, so features with large values can control the result. In the ULB data, Amount ranges from 0 to 25,691 and Time has a standard deviation of about 47,488, while the V features are small. For example, a difference of 50 in Amount adds 2,500 to the squared distance, while a difference of 2 in each of the 28 V features adds only 28 × 4 = 112 in total. Amount alone would then be about 96% of the distance. The paper does not say whether it scaled its features, so I cannot say it did not. I did not run an unscaled comparison, so the effect of scaling on BCBSMOTE is a hypothesis that I leave for future work.

## c. Proposed Alternatives

All results below are from the ULB dataset, with scaling and resampling done inside the training part only. Each number comes from one run on a test set with about 95 frauds. A difference of about 0.01 in F1 is within run-to-run noise, so I do not treat small gaps as real.

### c1. Add the missing baseline

**Table 1. Plain Random Forest against BCBSMOTE.**

| Method | F1 | AUPRC | MCC | Recall | Precision | TP | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No resampling | 0.812 | 0.821 | 0.817 | 0.726 | 0.920 | 69 | 6 |
| BCBSMOTE | 0.800 | 0.799 | 0.803 | 0.737 | 0.875 | 70 | 10 |

The plain Random Forest is as good as or better than BCBSMOTE on F1, AUPRC, MCC and precision. BCBSMOTE finds one more fraud (70 against 69) but raises four more false alarms. On this dataset, BCBSMOTE gives no clear benefit. This does not prove the paper wrong, because the datasets differ.

### c2. Test each stage on its own

**Table 2. Ablation study.**

| Configuration | F1 | AUPRC | MCC | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| No resampling | 0.805 | 0.819 | 0.811 | 0.716 | 0.919 |
| Tomek links only | 0.809 | 0.804 | 0.813 | 0.737 | 0.897 |
| Borderline-SMOTE only | 0.800 | 0.816 | 0.801 | 0.758 | 0.847 |
| Full BCBSMOTE | 0.805 | 0.797 | 0.808 | 0.737 | 0.886 |

All four F1 values lie within 0.009 of each other, which is inside the run-to-run noise. The only clear pattern is a trade-off. Borderline-SMOTE raises recall (0.758) and lowers precision (0.847). I cannot say that any stage, or the full pipeline, is better than doing nothing. (The same pipeline gives F1 = 0.800 in Table 1 and 0.805 in Table 2, which again shows the noise.)

### c3. Try other classifiers

**Table 3. Classifiers with BCBSMOTE resampling.**

| Classifier | F1 | AUPRC | MCC | Recall | Precision | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.763 | 0.729 | 0.765 | 0.832 | 0.705 | 0.79 |
| Random Forest | 0.800 | 0.799 | 0.803 | 0.737 | 0.875 | 16.15 |
| XGBoost \[9\] | 0.795 | 0.821 | 0.800 | 0.716 | 0.895 | 1.39 |
| Balanced Random Forest \[10\] | 0.182 | 0.726 | 0.297 | 0.884 | 0.101 | 2.04 |

XGBoost has the best AUPRC (0.821 against 0.799, a small gap on 95 frauds), a similar F1 to Random Forest, and trains about 11 times faster. It is therefore an attractive alternative for speed, but I cannot claim it is more accurate. XGBoost also has a scale\_pos\_weight setting that handles imbalance without resampling, but I did not test that here. The Balanced Random Forest has high recall (0.884) but precision of only 0.101, about 9 false alarms for each real fraud. Its low F1 may also come from combining its own internal balancing with BCBSMOTE, so I do not draw a general conclusion from this one setting.

### c4. Choose the decision threshold by cost

Instead of the default 0.5 cut-off, I searched for the threshold with the lowest total cost, assuming a missed fraud costs 500 and a false alarm costs 10. A threshold of 0.14 reduced the total cost from 12,590 to 8,700, a saving of 3,890 (30.9%). A threshold must be chosen using training data only (for example, by cross-validation) and then applied once to the test set. If it is chosen by scanning the test set, the saving is optimistic. I therefore treat this figure as an indication of the possible saving, not a final result.

### c5. A better way to evaluate

I recommend repeating every experiment with five or more random seeds and reporting mean ± standard deviation. I also recommend always including a plain baseline, reporting AUPRC and MCC \[11\], \[12\], and choosing the threshold on training data.

## d. Stress-Testing

I tested three hypotheses about conditions that can happen in real use. Each test should be compared with a plain Random Forest under the same condition, so that any weakness can be blamed on the resampling and not on the classifier.

### d1. Label noise

**Hypothesis.** Real fraud labels are often late or wrong. BCBSMOTE should get worse as label noise grows, because Tomek links needs correct labels to find border pairs.

**Table 4. BCBSMOTE with random label noise in the training data.**

| Noise level | F1 | AUPRC | MCC | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| 0% | 0.798 | 0.796 | 0.801 | 0.726 | 0.885 |
| 1% | 0.555 | 0.619 | 0.569 | 0.716 | 0.453 |
| 5% | 0.786 | 0.782 | 0.790 | 0.716 | 0.872 |
| 10% | 0.802 | 0.769 | 0.804 | 0.747 | 0.866 |

**Result and honest reading.** F1 fell by 0.243 points at 1% noise, but it was back to 0.786 at 5% and 0.802 at 10%. This is not a normal pattern, because more noise should not give better results. The cause may be randomness in a single run or an error in how the noise was added. I do not claim that the pipeline is fragile to small amounts of noise. The result only shows that one run is not enough. To settle this, the test should be repeated with five or more seeds and a plain Random Forest as a control, and the method of flipping labels (training set only) should be stated clearly.

### d2. Removing important features

**Hypothesis.** Features can change or disappear in real use. If the three strongest features (V14, V17, V12) are removed, recall should drop.

**Table 5. BCBSMOTE with three features removed.**

| Scenario | F1 | AUPRC | Recall | Precision |
| --- | --- | --- | --- | --- |
| All features | 0.798 | 0.796 | 0.726 | 0.885 |
| V14, V17, V12 removed | 0.753 | 0.778 | 0.642 | 0.910 |

**Result.** Recall fell by 8.4 points (about 8 frauds out of 95), and precision rose slightly. This fits the hypothesis, but any model gets worse when its best features are removed. Because there is no plain Random Forest control, I cannot say that BCBSMOTE is more sensitive than other models, and the result comes from one run.

### d3. More dimensions (planned test)

**Hypothesis.** Tomek links, BIRCH and SMOTE all use distances, and in high dimensions distances become less meaningful \[13\]. If I add many useless features, BCBSMOTE should get worse faster than a plain Random Forest, which can ignore features that do not help.

**Method.** Add 20 and then 100 independent random features (standard normal) to the training and test data. Run plain Random Forest and BCBSMOTE with 5 seeds, resampling inside the training part only. Report mean ± standard deviation of F1 and AUPRC.

**Table 6. Effect of adding useless features (results to be added after the run).**

| Extra features | Plain RF F1 | BCBSMOTE F1 | Plain RF AUPRC | BCBSMOTE AUPRC |
| --- | --- | --- | --- | --- |
| 0 | to run | to run | to run | to run |
| 20 | to run | to run | to run | to run |
| 100 | to run | to run | to run | to run |

If BCBSMOTE falls much faster than the plain forest, the hypothesis is supported. If both fall together, it is not.

## Limitations

Most results come from one run on one dataset with about 95 frauds in the test set, so small differences are not reliable. The BCBSMOTE used here is an approximation of the paper’s method. ULB is real but old (two days in 2013), and PaySim is simulated, so neither is a perfect test.

## Conclusion

BCBSMOTE is a sensible idea, but the evidence for it is weaker than the paper suggests. The paper’s gain is small and comes from a single run. Tomek links remove almost nothing, there is no plain baseline, and recall is lower than that of the competing methods. On the ULB dataset, a plain Random Forest was as good as BCBSMOTE, and no single stage was clearly better. The stress tests point to possible weaknesses, but they need repeated runs and plain-model controls before any firm conclusion. My main recommendations are to always report a plain baseline, to repeat experiments over several seeds, to add MCC and cost-based thresholds, and to scale features before distance-based resampling.

## References

&#91;1\] M. Alamri and M. Ykhlef, “Hybrid undersampling and oversampling for handling imbalanced credit card data,” *IEEE Access*, vol. 12, pp. 14050–14060, 2024, doi: 10.1109/ACCESS.2024.3357091.

&#91;2\] E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, “PaySim: A financial mobile money simulator for fraud detection,” in *Proc. 28th European Modeling and Simulation Symp.*, 2016.

&#91;3\] A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, “Calibrating probability with undersampling for unbalanced classification,” in *Proc. IEEE Symp. Comput. Intell. Data Mining (CIDM)*, 2015, pp. 159–166.

&#91;4\] I. Tomek, “Two modifications of CNN,” *IEEE Trans. Syst., Man, Cybern.*, vol. SMC-6, no. 11, pp. 769–772, 1976.

&#91;5\] T. Zhang, R. Ramakrishnan, and M. Livny, “BIRCH: An efficient data clustering method for very large databases,” in *Proc. ACM SIGMOD*, 1996, pp. 103–114.

&#91;6\] H. Han, W.-Y. Wang, and B.-H. Mao, “Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning,” in *Proc. Int. Conf. Intell. Comput. (ICIC)*, 2005, pp. 878–887.

&#91;7\] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority over-sampling technique,” *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.

&#91;8\] L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

&#91;9\] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 785–794.

&#91;10\] C. Chen, A. Liaw, and L. Breiman, “Using random forest to learn imbalanced data,” Dept. Statist., Univ. California, Berkeley, Tech. Rep. 666, 2004.

&#91;11\] J. Davis and M. Goadrich, “The relationship between precision-recall and ROC curves,” in *Proc. 23rd Int. Conf. Mach. Learn. (ICML)*, 2006, pp. 233–240.

&#91;12\] D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,” *BMC Genomics*, vol. 21, art. 6, 2020.

&#91;13\] C. C. Aggarwal, A. Hinneburg, and D. A. Keim, “On the surprising behavior of distance metrics in high dimensional space,” in *Proc. Int. Conf. Database Theory (ICDT)*, 2001, pp. 420–434.
