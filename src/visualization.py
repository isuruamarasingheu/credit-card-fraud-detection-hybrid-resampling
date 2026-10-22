"""
Visualization Module for Credit Card Fraud Detection.
Generates publication-quality figures matching the IEEE Access paper:
- Figure 1: Class distribution (Normal vs Fraud)
- Figure 2: Correlation matrix heatmap
- Figure 3: Proportions of transaction types
- Figure 4: Fraud vs Normal transactions per transaction type
- Figure 7: Tomek links undersampling impact
- Figure 9: Benchmark comparison (SMOTE, B-SMOTE, ADASYN, etc. vs Proposed)
- Figure 10: Precision-Recall Curve (AUPRC)
- Confusion Matrix Heatmap matching Table 5
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def set_plotting_style():
    """Sets a clean, publication-ready style for plots."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150
    })


def plot_class_distribution(y, output_path: str = None):
    """
    Figure 1: Numbers of normal and fraud transactions in PaySim dataset.
    """
    set_plotting_style()
    counts = pd.Series(y).value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['0 (Normal)', '1 (Fraud)'], counts.values, color=['#4CAF50', '#8BC34A'], width=0.45)
    
    ax.set_yscale('log')
    ax.set_ylabel('No. of Transactions')
    ax.set_xlabel('isFraud')
    ax.set_title('Figure 1: Numbers of Normal and Fraud Transactions in PaySim Dataset')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved Figure 1 to: {output_path}")
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_path: str = None):
    """
    Figure 2: Correlation matrix for PaySim dataset.
    """
    set_plotting_style()
    corr_cols = [c for c in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                            'oldbalanceDest', 'newbalanceDest', 'type', 'isFraud', 'isFlaggedFraud'] if c in df.columns]
    corr = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1.0, vmax=1.0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Figure 2: Correlation Matrix for PaySim Dataset')
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved Figure 2 to: {output_path}")
    plt.close()


def plot_transaction_type_distribution(df: pd.DataFrame, output_path: str = None):
    """
    Figure 3: Proportions of transaction types in PaySim dataset.
    """
    set_plotting_style()
    type_counts = df['type'].value_counts()
    
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ['#8BC34A', '#29B6F6', '#FFCA28', '#AB47BC', '#26A69A']
    ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct='%1.2f%%',
        startangle=140,
        colors=colors[:len(type_counts)],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax.set_title('Figure 3: Proportions of Transaction Types in PaySim Dataset')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved Figure 3 to: {output_path}")
    plt.close()


def plot_fraud_by_transaction_type(df: pd.DataFrame, output_path: str = None):
    """
    Figure 4: Number of normal transactions and fraudulent transactions for each type of transaction.
    """
    set_plotting_style()
    grouped = df.groupby(['type', 'isFraud']).size().unstack(fill_value=0)
    grouped.columns = ['Genuine', 'Fraud']
    
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped.plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336'], width=0.6)
    ax.set_yscale('log')
    ax.set_ylabel('No. of Transactions (Log scale)')
    ax.set_xlabel('Transaction Type')
    ax.set_title('Figure 4: Count of Normal and Fraud Transactions in each Transaction Type')
    ax.set_xticklabels(grouped.index, rotation=0)
    ax.legend(title='Transaction', loc='upper right')
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved Figure 4 to: {output_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_prob, auprc: float, output_path: str = None):
    """
    Figure 10: AUPRC for hybrid Tomek links BCBSMOTE.
    """
    set_plotting_style()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color='#1E88E5', lw=2.5, label=f'Proposed (AUPRC = {auprc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f'Figure 10: AUPRC for Hybrid Tomek Links BCBSMOTE (AUPRC = {auprc:.4f})')
    ax.legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved Figure 10 to: {output_path}")
    plt.close()


def plot_confusion_matrix_heatmap(cm, output_path: str = None):
    """
    Table 5: Confusion matrix visualization.
    """
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    
    labels = np.array([
        [f"TN\n{cm[0, 0]:,}", f"FP\n{cm[0, 1]:,}"],
        [f"FN\n{cm[1, 0]:,}", f"TP\n{cm[1, 1]:,}"]
    ])
    
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=['Pred Normal (0)', 'Pred Fraud (1)'],
                yticklabels=['Actual Normal (0)', 'Actual Fraud (1)'])
    
    ax.set_title('Table 5: Confusion Matrix for Fraud Detection')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved Confusion Matrix to: {output_path}")
    plt.close()


def plot_benchmark_comparison(results_df: pd.DataFrame, output_path: str = None):
    """
    Figure 9: Comparison between state-of-the-art sampling methods and proposed method.
    """
    set_plotting_style()
    metrics = ['Accuracy', 'F1_score', 'Recall', 'Precision', 'AUPRC']
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    df_plot = results_df.set_index('Method')[available_metrics] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind='bar', ax=ax, width=0.75, colormap='Spectral')
    ax.set_ylabel('Score (%)')
    ax.set_title('Figure 9: Comparison Between Sampling Methods and Proposed BCBSMOTE')
    ax.set_ylim([0, 110])
    ax.set_xticklabels(results_df['Method'], rotation=15, ha='right')
    ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved Figure 9 to: {output_path}")
    plt.close()


def plot_evaluation_metrics(y_true, y_prob, metrics: dict, output_dir: str = "results"):
    """
    Convenience method to plot and save all relevant evaluation figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    if 'AUPRC' in metrics:
        plot_precision_recall_curve(
            y_true, y_prob, metrics['AUPRC'],
            output_path=os.path.join(output_dir, "figure_10_auprc_curve.png")
        )
    if 'Confusion_Matrix' in metrics:
        plot_confusion_matrix_heatmap(
            metrics['Confusion_Matrix'],
            output_path=os.path.join(output_dir, "table_5_confusion_matrix.png")
        )
