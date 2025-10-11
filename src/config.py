"""
Configuration module for Beyond SMOTE evaluation framework
Purpose: Defining all project constants, paths, and hyperparameters
"""

import os
from pathlib import Path

# PATH CONFIGURATION

# Establishing base directory structure
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT_DIR / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
RESAMPLED_DIR = DATA_ROOT / "resampled"

# Setting up output directories
OUTPUT_ROOT = ROOT_DIR / "results"
FIGURE_OUTPUT = OUTPUT_ROOT / "figures"
TABLE_OUTPUT = OUTPUT_ROOT / "tables"
METRIC_OUTPUT = OUTPUT_ROOT / "metrics"
LOG_OUTPUT = OUTPUT_ROOT / "logs"

# Configuring model storage locations
MODEL_ROOT = ROOT_DIR / "models"
SKLEARN_STORAGE = MODEL_ROOT / "sklearn"
PYTORCH_STORAGE = MODEL_ROOT / "pytorch"
SCALER_STORAGE = MODEL_ROOT / "scalers"

# Creating directories if not existing
for dir_path in [DATA_ROOT, RAW_DIR, PROCESSED_DIR, RESAMPLED_DIR,
                 OUTPUT_ROOT, FIGURE_OUTPUT, TABLE_OUTPUT, METRIC_OUTPUT, LOG_OUTPUT,
                 MODEL_ROOT, SKLEARN_STORAGE, PYTORCH_STORAGE, SCALER_STORAGE]:
    dir_path.mkdir(parents=True, exist_ok=True)

# DATASET SPECIFICATIONS

# Defining dataset properties
DATASET_ID = "HIGGS"
DATA_FILE = RAW_DIR / "HIGGS.csv"

# Specifying dataset dimensions
TOTAL_ROWS = 11000000
FEATURE_COUNT = 28
CLASS_COUNT = 2

# Naming all features in order
ALL_FEATURES = [
    # Physics low-level measurements (21 features)
    'lepton_pT', 'lepton_eta', 'lepton_phi',
    'missing_energy_magnitude', 'missing_energy_phi',
    'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag',
    'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b_tag',
    'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b_tag',
    'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b_tag',
    # High-level derived features (7 features)
    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
]

# Naming target variable
TARGET_COL = 'label'

# EXPERIMENTAL DESIGN PARAMETERS

# Setting reproducibility seed
SEED_VALUE = 42

# Defining data split proportions
VALIDATION_SPLIT = 0.2
TRAINING_SPLIT = 0.8

# Configuring cross-validation
FOLD_COUNT = 5
SPLIT_METHOD = "stratified"

# Managing memory constraints through sampling
ENABLE_SAMPLING = True
SAMPLE_ROWS = 1000000
SAMPLE_SEED = SEED_VALUE

# MODEL HYPERPARAMETER CONFIGURATIONS

# Configuring logistic regression
LOGIT_CONFIG = {
    'max_iter': 1000,
    'random_state': SEED_VALUE,
    'n_jobs': -1,
    'solver': 'lbfgs'
}

# Configuring random forest
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': SEED_VALUE,
    'n_jobs': -1,
    'verbose': 0
}

# Configuring gradient boosting
XGB_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED_VALUE,
    'n_jobs': -1,
    'verbosity': 0,
    'eval_metric': 'logloss'
}

# Configuring support vector machine
SVM_CONFIG = {
    'kernel': 'rbf',
    'probability': True,
    'random_state': SEED_VALUE,
    'max_iter': 1000
}

# Configuring neural network (sklearn)
MLP_CONFIG = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 200,
    'random_state': SEED_VALUE,
    'early_stopping': True,
    'validation_fraction': 0.1
}

# Configuring deep learning (PyTorch)
DL_CONFIG = {
    'architecture': [256, 128, 64, 32],
    'dropout_vals': [0.3, 0.3, 0.2],
    'learn_rate': 0.001,
    'batch_count': 512,
    'epoch_count': 50,
    'patience': 5,
    'compute_device': 'cpu'
}

# RESAMPLING TECHNIQUE PARAMETERS

# Configuring SMOTE oversampling
SMOTE_CONFIG = {
    'k_neighbors': 5,
    'random_state': SEED_VALUE,
    'n_jobs': -1
}

# Configuring borderline SMOTE
BSMOTE_CONFIG = {
    'k_neighbors': 5,
    'random_state': SEED_VALUE,
    'n_jobs': -1,
    'kind': 'borderline-1'
}

# Configuring ADASYN
ADASYN_CONFIG = {
    'n_neighbors': 5,
    'random_state': SEED_VALUE,
    'n_jobs': -1
}

# Configuring random oversampling
ROS_CONFIG = {
    'random_state': SEED_VALUE
}

# Configuring random undersampling
RUS_CONFIG = {
    'random_state': SEED_VALUE
}

# Configuring Tomek links
TOMEK_CONFIG = {
    'n_jobs': -1
}

# Configuring NearMiss
NMISS_CONFIG = {
    'version': 1,
    'n_neighbors': 3,
    'n_jobs': -1
}

# Configuring SMOTE-Tomek hybrid
SMTOMEK_CONFIG = {
    'random_state': SEED_VALUE,
    'n_jobs': -1
}

# Configuring SMOTE-ENN hybrid
SMENN_CONFIG = {
    'random_state': SEED_VALUE,
    'n_jobs': -1
}


# EVALUATION METRIC SELECTION

# Defining metrics for assessment
ASSESSMENT_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc_roc',
    'auc_pr',
    'g_mean',
    'mcc'
]

# VISUALIZATION PREFERENCES

# Setting figure properties
DPI_VALUE = 300
FILE_FORMAT = 'png'
DEFAULT_FIGSIZE = (10, 6)

# Choosing color schemes
PALETTE_CHOICE = 'Set2'
HEAT_COLORMAP = 'YlOrRd'

# Selecting plot style
STYLE_CHOICE = 'seaborn-v0_8-darkgrid'

# COMPUTATIONAL RESOURCE MANAGEMENT

# Enabling parallel processing
PARALLEL_JOBS = -1

# Setting memory constraints
RAM_LIMIT_GB = 16

# Controlling output verbosity
VERBOSE_LEVEL = 1

# LOGGING CONFIGURATION

# Setting logging parameters
LOG_SEVERITY = "INFO"
LOG_TEMPLATE = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DESTINATION = LOG_OUTPUT / "experiment.log"

# UTILITY FUNCTIONS

def compute_class_weights(target_array):
    """
    Computing balanced class weights for imbalanced datasets

    Args:
        target_array: Array containing class labels

    Returns:
        Dictionary mapping class labels to their weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    unique_classes = np.unique(target_array)
    weight_values = compute_class_weight('balanced',
                                         classes=unique_classes,
                                         y=target_array)
    return dict(zip(unique_classes, weight_values))


def display_configuration():
    """Displaying current configuration settings"""
    separator = "=" * 70
    print(separator)
    print("PROJECT CONFIGURATION SUMMARY")
    print(separator)
    print("\nData Locations:")
    print(f"  Dataset Path: {DATA_FILE}")
    print(f"  Processing Directory: {PROCESSED_DIR}")
    print(f"  Output Directory: {OUTPUT_ROOT}")

    print("\nExperiment Parameters:")
    print(f"  Random Seed: {SEED_VALUE}")
    print(f"  Cross-Validation Folds: {FOLD_COUNT}")
    print(f"  Validation Split: {VALIDATION_SPLIT}")
    print(f"  Sample Size: {SAMPLE_ROWS if ENABLE_SAMPLING else 'Full Dataset'}")

    print("\nComputation Settings:")
    print(f"  Parallel Jobs: {PARALLEL_JOBS}")
    print(f"  Verbosity Level: {VERBOSE_LEVEL}")
    print(f"  PyTorch Device: {DL_CONFIG['compute_device']}")

    print(separator)


if __name__ == "__main__":
    display_configuration()
    print("\nConfiguration module loaded successfully")
    print(f"Total features defined: {len(ALL_FEATURES)}")
    print("Model configurations: 7")
    print("Resampling configurations: 11")
