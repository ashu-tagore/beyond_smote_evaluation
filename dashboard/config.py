"""
Configuration file for Higgs Discovery Dashboard
Defined paths and settings for loading saved analysis results
"""

import os
from pathlib import Path

# Defined base project directory
BASE_DIR = Path(__file__).parent.parent

# Defined data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Defined results directories
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

# Defined model directories
MODELS_DIR = BASE_DIR / "models"

# Defined dashboard settings
DASHBOARD_TITLE = "Higgs Boson Discovery Analysis Dashboard"
PAGE_ICON = ":atom_symbol:"
LAYOUT = "wide"

# Defined color scheme for visualizations
COLORS = {
    "signal": "#FF6B6B",
    "background": "#4ECDC4",
    "primary": "#1A535C",
    "secondary": "#FFE66D",
    "success": "#95E1D3",
    "warning": "#F38181",
    "info": "#3D5A80"
}

# Defined model names mapping
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
    "gradient_boosting": "Gradient Boosting"
}

# Defined metric names for display
METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC"
}

# Defined physics feature names
PHYSICS_FEATURES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b_tag",
    "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b_tag",
    "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b_tag",
    "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b_tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
]

# Defined page configuration
PAGES = [
    "Overview",
    "Data Explorer",
    "Feature Analysis",
    "Model Comparison",
    "Physics Discoveries",
    "Recommendations"
]

# Created function to ensure directories exist
def ensure_directories():
    """
    Ensured all required directories exist
    Created directories if they do not exist
    """
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        RESULTS_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR,
        MODELS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Created function to get file paths
def get_result_path(category, filename):
    """
    Retrieved full path for a result file

    Parameters:
        category: Type of result (figures, metrics, tables)
        filename: Name of the file

    Returns:
        Full path to the file
    """
    category_dirs = {
        "figures": FIGURES_DIR,
        "metrics": METRICS_DIR,
        "tables": TABLES_DIR,
        "models": MODELS_DIR
    }

    return category_dirs.get(category, RESULTS_DIR) / filename
