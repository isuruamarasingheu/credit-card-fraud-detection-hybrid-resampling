"""
Data Preprocessing Module for Credit Card Fraud Detection.
Based on Section III (Dataset) & Section III.A/B (Dataset Preparation and Analysis)
from Alamri & Ykhlef (2024).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Features retained according to the paper:
# Dropped:
# - nameOrig, nameDest: Customer ID strings
# - isFlaggedFraud: Weak heuristic column (>200,000 USD transfer)
# - newbalanceOrig: Dropped due to extremely high correlation (~1.0) with oldbalanceOrg
# Retained:
# - step, type, amount, oldbalanceOrg, oldbalanceDest, newbalanceDest
FEATURE_COLUMNS = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'newbalanceDest']
TARGET_COLUMN = 'isFraud'


def load_raw_dataset(file_path: str, nrows: int = None, sample_size: int = None, random_state: int = 42) -> pd.DataFrame:
    """
    Loads the PaySim transaction dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the PaySim dataset CSV.
    nrows : int, optional
        Number of rows to read from the top of the CSV file.
    sample_size : int, optional
        If provided, randomly samples this number of transactions while maintaining class ratio.
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame
        Raw or sampled transaction DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    print(f"Loading dataset from: {file_path}...")
    if sample_size is not None and sample_size > 0:
        # Load in chunks to stratify sample if sample_size is specified
        df = pd.read_csv(file_path, nrows=nrows)
        if len(df) > sample_size:
            print(f"Subsampling {sample_size} records while maintaining class proportion...")
            df, _ = train_test_split(df, train_size=sample_size, stratify=df[TARGET_COLUMN], random_state=random_state)
            df = df.reset_index(drop=True)
    else:
        df = pd.read_csv(file_path, nrows=nrows)
        
    print(f"Loaded {len(df):,} transactions with {df.shape[1]} columns.")
    return df


def clean_and_verify_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning checks described in Section III.A:
    1. Checks for null/missing values (df.isnull()).
    2. Checks for duplicate rows (df.duplicated()).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input transaction DataFrame.
        
    Returns:
    --------
    pd.DataFrame
        Verified DataFrame.
    """
    print("\n--- Running Dataset Cleaning & Integrity Verification ---")
    null_count = df.isnull().sum().sum()
    print(f"Missing / Null values count: {null_count}")
    if null_count > 0:
        print("Imputing or dropping missing values...")
        df = df.dropna().reset_index(drop=True)
        
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows count: {dup_count}")
    if dup_count > 0:
        print("Removing duplicate rows...")
        df = df.drop_duplicates().reset_index(drop=True)
        
    normal_count = (df[TARGET_COLUMN] == 0).sum()
    fraud_count = (df[TARGET_COLUMN] == 1).sum()
    fraud_ratio = (fraud_count / len(df)) * 100
    print(f"Total transactions: {len(df):,}")
    print(f"Normal transactions (Class 0): {normal_count:,} ({100 - fraud_ratio:.4f}%)")
    print(f"Fraud transactions  (Class 1): {fraud_count:,} ({fraud_ratio:.4f}%)")
    print("--------------------------------------------------------\n")
    return df


def preprocess_features(df: pd.DataFrame, label_encoder: LabelEncoder = None):
    """
    Applies feature selection and categorical encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset.
    label_encoder : LabelEncoder, optional
        Pre-fitted LabelEncoder for 'type' column (used during inference).
        
    Returns:
    --------
    X : pd.DataFrame
        Preprocessed feature matrix.
    y : pd.Series or None
        Target labels (if 'isFraud' in df).
    label_encoder : LabelEncoder
        Fitted label encoder for 'type'.
    """
    df = df.copy()
    
    # Check if target column is present
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    
    # Encode categorical transaction 'type'
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['type'] = label_encoder.fit_transform(df['type'].astype(str))
    else:
        df['type'] = label_encoder.transform(df['type'].astype(str))
        
    # Drop collinear, identifier, and irrelevant columns
    # Dropping newbalanceOrig, nameOrig, nameDest, isFlaggedFraud
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_cols]
    
    return X, y, label_encoder


def prepare_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into 80% training and 20% test sets using stratified sampling
    as specified in Section V.A and Figure 6 of the paper.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Fraction of data for testing (default 0.20 = 20%).
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    print(f"Splitting dataset into {int((1-test_size)*100)}% Train and {int(test_size*100)}% Test (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print(f"Train set: {len(X_train):,} samples (Normal: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,})")
    print(f"Test set:  {len(X_test):,} samples (Normal: {(y_test == 0).sum():,}, Fraud: {(y_test == 1).sum():,})")
    return X_train, X_test, y_train, y_test


def load_and_preprocess_data(file_path: str, nrows: int = None, sample_size: int = None, test_size: float = 0.2, random_state: int = 42):
    """
    End-to-end data loading and preprocessing pipeline.
    """
    raw_df = load_raw_dataset(file_path, nrows=nrows, sample_size=sample_size, random_state=random_state)
    cleaned_df = clean_and_verify_dataset(raw_df)
    X, y, encoder = preprocess_features(cleaned_df)
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, encoder
