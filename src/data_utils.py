"""
Data handling utilities for HIGGS dataset analysis
Purpose: Implementing data loading, preprocessing, and transformation functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import sys

# Adding source directory to path
sys.path.append(str(Path(__file__).parent))
from config import *


def load_higgs_dataset(file_location=None, row_limit=None,
                       sample_count=None, show_info=True):
    """
    Loading HIGGS dataset from CSV file

    Args:
        file_location: Path to dataset file
        row_limit: Maximum rows to load
        sample_count: Number of samples for random sampling
        show_info: Whether to display loading information

    Returns:
        features: DataFrame containing feature columns
        labels: Series containing target variable
    """

    if file_location is None:
        file_location = DATA_FILE

    if show_info:
        print("Loading HIGGS dataset from file...")
        print(f"Source: {file_location}")

    # Determining number of rows to read
    if sample_count is not None:
        rows_to_load = sample_count
    elif row_limit is not None:
        rows_to_load = row_limit
    elif ENABLE_SAMPLING:
        rows_to_load = SAMPLE_ROWS
    else:
        rows_to_load = None

    if show_info and rows_to_load:
        print(f"Reading {rows_to_load:,} rows from dataset...")

    # Preparing column names
    col_names = [TARGET_COL] + ALL_FEATURES

    try:
        # Reading CSV data
        dataset = pd.read_csv(
            file_location,
            names=col_names,
            nrows=rows_to_load,
            dtype=np.float32
        )

        # Applying random sampling if needed
        if sample_count and len(dataset) > sample_count:
            dataset = dataset.sample(n=sample_count, random_state=SEED_VALUE)

        # Separating features and target
        features = dataset[ALL_FEATURES]
        labels = dataset[TARGET_COL].astype(int)

        if show_info:
            print("Dataset loaded successfully")
            print(f"Shape: {features.shape}")
            memory_mb = features.memory_usage(deep=True).sum() / 1e6
            print(f"Memory Usage: {memory_mb:.2f} MB")
            print("Class Distribution:")
            signal_count = (labels == 1).sum()
            background_count = (labels == 0).sum()
            signal_pct = (labels == 1).mean() * 100
            background_pct = (labels == 0).mean() * 100
            print(f"  Signal (1): {signal_count:,} ({signal_pct:.2f}%)")
            print(f"  Background (0): {background_count:,} ({background_pct:.2f}%)")
            imbalance = signal_count / background_count
            print(f"  Imbalance Ratio: {imbalance:.3f}:1")

        return features, labels

    except FileNotFoundError:
        print(f"Error: Cannot find file at {file_location}")
        print(f"Please verify HIGGS.csv exists in {RAW_DIR}")
        raise
    except Exception as error:
        print(f"Error during data loading: {error}")
        raise


def retrieve_feature_descriptions():
    """
    Retrieving feature names with detailed descriptions

    Returns:
        Dictionary mapping feature names to descriptions
    """

    feature_info = {
        # Lepton measurements
        'lepton_pT': 'Lepton transverse momentum component',
        'lepton_eta': 'Lepton pseudorapidity measurement',
        'lepton_phi': 'Lepton azimuthal angle',

        # Missing energy components
        'missing_energy_magnitude': 'Transverse missing energy magnitude',
        'missing_energy_phi': 'Missing energy azimuthal direction',

        # First jet properties
        'jet_1_pt': 'Leading jet transverse momentum',
        'jet_1_eta': 'Leading jet pseudorapidity',
        'jet_1_phi': 'Leading jet azimuthal angle',
        'jet_1_b_tag': 'Leading jet bottom quark tag',

        # Second jet properties
        'jet_2_pt': 'Second jet transverse momentum',
        'jet_2_eta': 'Second jet pseudorapidity',
        'jet_2_phi': 'Second jet azimuthal angle',
        'jet_2_b_tag': 'Second jet bottom quark tag',

        # Third jet properties
        'jet_3_pt': 'Third jet transverse momentum',
        'jet_3_eta': 'Third jet pseudorapidity',
        'jet_3_phi': 'Third jet azimuthal angle',
        'jet_3_b_tag': 'Third jet bottom quark tag',

        # Fourth jet properties
        'jet_4_pt': 'Fourth jet transverse momentum',
        'jet_4_eta': 'Fourth jet pseudorapidity',
        'jet_4_phi': 'Fourth jet azimuthal angle',
        'jet_4_b_tag': 'Fourth jet bottom quark tag',

        # Composite features
        'm_jj': 'Invariant mass calculated from two jets',
        'm_jjj': 'Invariant mass calculated from three jets',
        'm_lv': 'Invariant mass from lepton and missing energy',
        'm_jlv': 'Combined invariant mass measurement',
        'm_bb': 'Invariant mass from bottom-tagged jets',
        'm_wbb': 'Mass from W boson and two bottom jets',
        'm_wwbb': 'Mass from two W bosons and two bottom jets'
    }

    return feature_info


def perform_data_cleaning(features, labels, eliminate_outliers=False, sigma_threshold=5):
    """
    Performing data cleaning and validation operations

    Args:
        features: Feature DataFrame
        labels: Target Series
        eliminate_outliers: Whether to remove statistical outliers
        sigma_threshold: Standard deviation threshold for outlier removal

    Returns:
        cleaned_features: Cleaned feature DataFrame
        cleaned_labels: Cleaned target Series
    """

    print("Initiating data cleaning process...")

    original_count = len(features)

    # Checking for missing values
    missing_total = features.isnull().sum().sum()
    if missing_total > 0:
        print(f"Found {missing_total} missing values, removing affected rows...")
        valid_mask = ~features.isnull().any(axis=1)
        features = features[valid_mask]
        labels = labels[valid_mask]

    # Checking for infinite values
    infinite_mask = np.isinf(features).any(axis=1)
    if infinite_mask.sum() > 0:
        print(f"Found {infinite_mask.sum()} infinite values, removing affected rows...")
        features = features[~infinite_mask]
        labels = labels[~infinite_mask]

    # Removing outliers if requested
    if eliminate_outliers:
        print(f"Detecting outliers beyond {sigma_threshold} standard deviations...")
        z_scores = np.abs((features - features.mean()) / features.std())
        valid_data_mask = (z_scores < sigma_threshold).all(axis=1)
        outlier_count = (~valid_data_mask).sum()

        if outlier_count > 0:
            print(f"Removing {outlier_count} outlier observations")
            features = features[valid_data_mask]
            labels = labels[valid_data_mask]

    removed_count = original_count - len(features)
    if removed_count > 0:
        removal_pct = removed_count / original_count * 100
        print(f"Removed {removed_count} problematic rows ({removal_pct:.2f}%)")
    else:
        print("Data cleaning complete, no modifications needed")

    return features.reset_index(drop=True), labels.reset_index(drop=True)


def apply_feature_scaling(train_features, test_features=None, scaler_save_path=None):
    """
    Applying standardization to feature columns

    Args:
        train_features: Training feature set
        test_features: Testing feature set (optional)
        scaler_save_path: Path for saving scaler object

    Returns:
        scaled_train: Standardized training features
        scaled_test: Standardized test features (if provided)
        scaler_object: Fitted StandardScaler instance
    """

    print("Applying feature standardization...")

    # Initializing scaler
    scaler_object = StandardScaler()

    # Fitting scaler on training data
    scaled_train = scaler_object.fit_transform(train_features)
    print("Scaler fitted on training data")

    # Transforming test data if provided
    if test_features is not None:
        scaled_test = scaler_object.transform(test_features)
        print("Test data transformed using training statistics")

    # Saving scaler if path provided
    if scaler_save_path:
        Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_object, scaler_save_path)
        print(f"Scaler saved to {scaler_save_path}")

    if test_features is not None:
        return scaled_train, scaled_test, scaler_object
    else:
        return scaled_train, scaler_object


def generate_train_test_split(features, labels, validation_proportion=None,
                               seed=None, use_stratification=True):
    """
    Generating stratified train-test data splits

    Args:
        features: Feature matrix
        labels: Target vector
        validation_proportion: Proportion for validation set
        seed: Random seed for reproducibility
        use_stratification: Whether to maintain class proportions

    Returns:
        train_x, test_x, train_y, test_y: Split datasets
    """

    if validation_proportion is None:
        validation_proportion = VALIDATION_SPLIT

    if seed is None:
        seed = SEED_VALUE

    print("Generating train-test split...")
    print(f"Validation proportion: {validation_proportion*100:.0f}%")
    print(f"Stratification: {'Enabled' if use_stratification else 'Disabled'}")

    train_x, test_x, train_y, test_y = train_test_split(
        features, labels,
        test_size=validation_proportion,
        random_state=seed,
        stratify=labels if use_stratification else None
    )

    print("Split operation completed")
    print(f"Training set: {len(train_x):,} observations")
    print(f"Testing set: {len(test_x):,} observations")

    # Verifying class balance in splits
    train_positive = (train_y == 1).sum()
    test_positive = (test_y == 1).sum()
    train_pos_pct = train_positive / len(train_y) * 100
    test_pos_pct = test_positive / len(test_y) * 100

    print(f"Training positive class: {train_pos_pct:.2f}%")
    print(f"Testing positive class: {test_pos_pct:.2f}%")

    return train_x, test_x, train_y, test_y


def compute_imbalance_metric(label_array):
    """
    Computing class imbalance ratio metric

    Args:
        label_array: Array of class labels

    Returns:
        Ratio of minority to majority class
    """

    classes, class_counts = np.unique(label_array, return_counts=True)

    if len(classes) != 2:
        raise ValueError("Function requires binary classification problem")

    minority_size = class_counts.min()
    majority_size = class_counts.max()

    imbalance_ratio = minority_size / majority_size

    return imbalance_ratio


def persist_processed_datasets(train_x, test_x, train_y, test_y, file_prefix="higgs"):
    """
    Persisting processed datasets to disk

    Args:
        train_x, test_x, train_y, test_y: Datasets to save
        file_prefix: Prefix for output filenames
    """

    print("Saving processed datasets to disk...")

    # Creating DataFrames with labels
    train_data = pd.DataFrame(train_x, columns=ALL_FEATURES)
    train_data[TARGET_COL] = train_y

    test_data = pd.DataFrame(test_x, columns=ALL_FEATURES)
    test_data[TARGET_COL] = test_y

    # Defining output paths
    train_output = PROCESSED_DIR / f"{file_prefix}_train.csv"
    test_output = PROCESSED_DIR / f"{file_prefix}_test.csv"

    # Writing to CSV
    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)

    print(f"Datasets saved to {PROCESSED_DIR}")
    print(f"Training file: {train_output.name}")
    print(f"Testing file: {test_output.name}")


def retrieve_processed_datasets(file_prefix="higgs"):
    """
    Retrieving previously saved processed datasets

    Args:
        file_prefix: Prefix of saved filenames

    Returns:
        train_x, test_x, train_y, test_y: Loaded datasets
    """

    train_input = PROCESSED_DIR / f"{file_prefix}_train.csv"
    test_input = PROCESSED_DIR / f"{file_prefix}_test.csv"

    print(f"Loading processed data from {PROCESSED_DIR}...")

    train_data = pd.read_csv(train_input)
    test_data = pd.read_csv(test_input)

    train_x = train_data[ALL_FEATURES]
    train_y = train_data[TARGET_COL]
    test_x = test_data[ALL_FEATURES]
    test_y = test_data[TARGET_COL]

    print("Datasets loaded successfully")
    print(f"Training shape: {train_x.shape}")
    print(f"Testing shape: {test_x.shape}")

    return train_x, test_x, train_y, test_y

def load_resampled_data(method_name, data_dir=None, file_prefix='higgs'):
    """
    Loading resampled dataset from disk
    
    Args:
        method_name: Name of resampling method
        data_dir: Directory containing resampled data
        file_prefix: Prefix of saved filename
    
    Returns:
        features: Feature DataFrame
        labels: Target Series
    """
    
    if data_dir is None:
        data_dir = RESAMPLED_DIR
    
    # Constructing filename
    filename = f"{file_prefix}_{method_name}_resampled.csv"
    file_path = Path(data_dir) / filename
    
    print(f"Loading resampled data: {method_name}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Resampled dataset not found: {file_path}")
    
    # Reading CSV
    dataset = pd.read_csv(file_path)
    
    # Separating features and target
    features = dataset[ALL_FEATURES]
    labels = dataset[TARGET_COL]
    
    print(f"  Loaded {len(features):,} samples")
    print(f"  Features shape: {features.shape}")
    
    return features, labels


if __name__ == "__main__":
    print("=" * 70)
    print("DATA UTILITIES MODULE TEST")
    print("=" * 70)

    print("\nTest 1: Loading dataset sample...")
    X, y = load_higgs_dataset(row_limit=10000)

    print("\nTest 2: Retrieving feature descriptions...")
    desc = retrieve_feature_descriptions()
    print(f"Retrieved {len(desc)} feature descriptions")

    print("\nTest 3: Cleaning data...")
    X_clean, y_clean = perform_data_cleaning(X, y)

    print("\nTest 4: Creating train-test split...")
    X_train, X_test, y_train, y_test = generate_train_test_split(X_clean, y_clean)

    print("\nTest 5: Scaling features...")
    X_train_scaled, X_test_scaled, scaler = apply_feature_scaling(X_train, X_test)

    print("\nTest 6: Computing imbalance ratio...")
    ratio = compute_imbalance_metric(y)
    print(f"Imbalance ratio: {ratio:.3f}:1")

    print("ALL TESTS COMPLETED SUCCESSFULLY")
