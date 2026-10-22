"""
Model Training and Evaluation Module for Credit Card Fraud Detection.
Based on Section V.D (Random Forest Classifier) and Section IV/VI (Evaluation Metrics & Results)
from Alamri & Ykhlef (2024).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from .sampling import HybridTomekBCBSMOTE


class FraudDetectionModel:
    """
    Complete Credit Card Fraud Detection Model.
    Combines Hybrid Tomek Links + BCBSMOTE resampling with a Random Forest Classifier.
    """
    def __init__(self, sampler=None, n_estimators: int = 50, max_depth: int = 15,
                 n_jobs: int = -1, random_state: int = 42, threshold: float = 0.5):
        """
        Parameters:
        -----------
        sampler : object, optional
            Resampling object with a fit_resample method. Defaults to HybridTomekBCBSMOTE.
        n_estimators : int
            Number of decision trees in Random Forest.
        max_depth : int
            Maximum depth of individual trees.
        n_jobs : int
            Number of CPU cores for parallel execution.
        random_state : int
            Reproducibility seed.
        threshold : float
            Decision probability threshold for classifying a transaction as fraud.
        """
        if sampler is None:
            self.sampler = HybridTomekBCBSMOTE(random_state=random_state)
        else:
            self.sampler = sampler
            
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state
        )
        self.threshold = threshold
        self.is_fitted = False
        self.feature_names_ = None

    def fit(self, X_train, y_train, resample: bool = True):
        """
        Fits the resampling pipeline and trains the Random Forest classifier.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
            
        X_arr = np.asarray(X_train, dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.int32)
        
        if resample and self.sampler is not None:
            print("--- Starting Hybrid Data Resampling Pipeline ---")
            X_res, y_res = self.sampler.fit_resample(X_arr, y_arr)
            print("--- Resampling Completed ---")
        else:
            X_res, y_res = X_arr, y_arr
            
        print(f"\nTraining Random Forest Classifier on {len(X_res):,} samples with {self.classifier.n_estimators} trees...")
        self.classifier.fit(X_res, y_res)
        self.is_fitted = True
        print("Model training completed successfully.")
        return self

    def predict_proba(self, X):
        """
        Predicts fraud probabilities for given transactions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.classifier.predict_proba(X_arr)[:, 1]

    def predict(self, X, threshold: float = None):
        """
        Predicts binary fraud class (0 = Legitimate, 1 = Fraud).
        """
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def optimize_threshold(self, X_val, y_val, metric: str = 'f1', target_value: float = None):
        """
        Finds the decision threshold that maximizes F1 or achieves a target recall/precision.
        """
        probs = self.predict_proba(X_val)
        y_true = np.asarray(y_val, dtype=np.int32)
        
        thresholds = np.linspace(0.01, 0.99, 100)
        best_thresh = self.threshold
        best_score = -1
        
        for t in thresholds:
            preds = (probs >= t).astype(int)
            if metric == 'f1':
                score = f1_score(y_true, preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, preds, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, preds, zero_division=0)
            else:
                score = f1_score(y_true, preds, zero_division=0)
                
            if score > best_score:
                best_score = score
                best_thresh = t
                
        print(f"Optimized threshold for {metric.upper()}: {best_thresh:.4f} (Score: {best_score*100:.2f}%)")
        self.threshold = best_thresh
        return best_thresh

    def evaluate(self, X_test, y_test, threshold: float = None):
        """
        Evaluates the model against test data and computes all metrics from the paper.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing Accuracy, Precision, Recall, F1, AUPRC, ROC-AUC, and Confusion Matrix.
        """
        thresh = threshold if threshold is not None else self.threshold
        X_arr = np.asarray(X_test, dtype=np.float32)
        y_true = np.asarray(y_test, dtype=np.int32)
        
        probs = self.predict_proba(X_arr)
        preds = (probs >= thresh).astype(int)
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        auprc = average_precision_score(y_true, probs)
        roc_auc = roc_auc_score(y_true, probs)
        cm = confusion_matrix(y_true, preds)
        
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'AUPRC': auprc,
            'ROC_AUC': roc_auc,
            'Confusion_Matrix': cm,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'Threshold': thresh,
            'Probabilities': probs
        }
        
        print("\n================ MODEL PERFORMANCE EVALUATION ================")
        print(f"Accuracy  : {acc * 100:.2f}%")
        print(f"F1-Score  : {f1 * 100:.2f}%")
        print(f"Recall    : {rec * 100:.2f}%")
        print(f"Precision : {prec * 100:.2f}%")
        print(f"AUPRC     : {auprc * 100:.2f}%")
        print(f"ROC-AUC   : {roc_auc * 100:.2f}%")
        print("\nConfusion Matrix (Actual Rows x Predicted Cols):")
        print(f"  True Negatives  (Legitimate identified correctly) : {tn:,}")
        print(f"  False Positives (Legitimate incorrectly flagged)  : {fp:,}")
        print(f"  False Negatives (Fraud transactions missed)       : {fn:,}")
        print(f"  True Positives  (Fraud transactions detected)     : {tp:,}")
        print("==============================================================")
        return metrics

    def save(self, filepath: str):
        """
        Saves the trained model and associated metadata to a joblib file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            'classifier': self.classifier,
            'threshold': self.threshold,
            'feature_names': self.feature_names_,
            'is_fitted': self.is_fitted
        }
        joblib.dump(checkpoint, filepath)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """
        Loads a saved model checkpoint.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        checkpoint = joblib.load(filepath)
        model = cls()
        model.classifier = checkpoint['classifier']
        model.threshold = checkpoint['threshold']
        model.feature_names_ = checkpoint['feature_names']
        model.is_fitted = checkpoint['is_fitted']
        print(f"Model loaded successfully from: {filepath}")
        return model
