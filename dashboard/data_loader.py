"""
Data loading utilities for Higgs Discovery Dashboard
Loaded saved results from notebook analysis
"""

import json
import pandas as pd
import numpy as np
from config import (
    METRICS_DIR, FIGURES_DIR, TABLES_DIR,
    PROCESSED_DATA_DIR
)

class ResultsLoader:
    """
    Loaded and managed analysis results from saved files
    """

    def __init__(self):
        """
        Initialized the results loader
        Set up paths to result files
        """
        self.metrics_dir = METRICS_DIR
        self.figures_dir = FIGURES_DIR
        self.tables_dir = TABLES_DIR
        self.data_dir = PROCESSED_DATA_DIR

    def load_model_metrics(self):
        """
        Loaded model performance metrics from JSON file

        Returns:
            Dictionary containing metrics for all models
        """
        metrics_file = self.metrics_dir / "model_metrics.json"

        if not metrics_file.exists():
            # Returned dummy data if file not found
            return self._get_dummy_metrics()

        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return metrics

    def load_feature_importance(self):
        """
        Loaded feature importance scores from JSON file

        Returns:
            Dictionary with feature names and importance scores
        """
        importance_file = self.metrics_dir / "feature_importance.json"

        if not importance_file.exists():
            return self._get_dummy_feature_importance()

        with open(importance_file, 'r', encoding='utf-8') as f:
            importance = json.load(f)

        return importance

    def load_statistical_significance(self):
        """
        Loaded statistical significance results

        Returns:
            Dictionary with discovery significance calculations
        """
        stats_file = self.metrics_dir / "statistical_significance.json"

        if not stats_file.exists():
            return {"significance": 40.78, "p_value": 1e-200}

        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        return stats

    def load_confusion_matrices(self):
        """
        Loaded confusion matrices for all models

        Returns:
            Dictionary with confusion matrix for each model
        """
        cm_file = self.metrics_dir / "confusion_matrices.json"

        if not cm_file.exists():
            return self._get_dummy_confusion_matrices()

        with open(cm_file, 'r', encoding='utf-8') as f:
            matrices = json.load(f)

        return matrices

    def load_roc_curves(self):
        """
        Loaded ROC curve data for all models

        Returns:
            Dictionary with FPR, TPR, and AUC for each model
        """
        roc_file = self.metrics_dir / "roc_curves.json"

        if not roc_file.exists():
            return self._get_dummy_roc_data()

        with open(roc_file, 'r', encoding='utf-8') as f:
            roc_data = json.load(f)

        return roc_data

    def load_data_summary(self):
        """
        Loaded dataset summary statistics

        Returns:
            DataFrame with dataset statistics
        """
        summary_file = self.data_dir / "data_summary.csv"

        if not summary_file.exists():
            return self._get_dummy_data_summary()

        summary_df = pd.read_csv(summary_file)
        return summary_df

    def load_feature_distributions(self):
        """
        Loaded feature distribution data

        Returns:
            DataFrame with feature statistics by class
        """
        dist_file = self.data_dir / "feature_distributions.csv"

        if not dist_file.exists():
            return self._get_dummy_feature_distributions()

        distributions = pd.read_csv(dist_file)
        return distributions

    def load_predictions(self):
        """
        Loaded model predictions

        Returns:
            DataFrame with predictions from all models
        """
        pred_file = self.data_dir / "signal_predictions.csv"

        if not pred_file.exists():
            return self._get_dummy_predictions()

        predictions = pd.read_csv(pred_file)
        return predictions

    def load_invariant_mass_data(self):
        """
        Loaded invariant mass calculation results

        Returns:
            Dictionary with mass histogram data
        """
        mass_file = self.metrics_dir / "invariant_mass.json"

        if not mass_file.exists():
            return self._get_dummy_mass_data()

        with open(mass_file, 'r', encoding='utf-8') as f:
            mass_data = json.load(f)

        return mass_data

    # Dummy data methods for testing when actual results not available

    def _get_dummy_metrics(self):
        """Generated dummy metrics for testing"""
        return {
            "logistic_regression": {
                "accuracy": 0.856,
                "precision": 0.842,
                "recall": 0.861,
                "f1_score": 0.851,
                "roc_auc": 0.923
            },
            "random_forest": {
                "accuracy": 0.889,
                "precision": 0.881,
                "recall": 0.893,
                "f1_score": 0.887,
                "roc_auc": 0.956
            },
            "xgboost": {
                "accuracy": 0.912,
                "precision": 0.908,
                "recall": 0.915,
                "f1_score": 0.911,
                "roc_auc": 0.971
            },
            "neural_network": {
                "accuracy": 0.901,
                "precision": 0.896,
                "recall": 0.905,
                "f1_score": 0.900,
                "roc_auc": 0.963
            }
        }

    def _get_dummy_feature_importance(self):
        """Generated dummy feature importance"""
        features = ["m_wwbb", "m_wbb", "m_bb", "jet_1_pt", "jet_2_pt"]
        importance = [0.15, 0.12, 0.10, 0.08, 0.07]
        return dict(zip(features, importance))

    def _get_dummy_confusion_matrices(self):
        """Generated dummy confusion matrices"""
        return {
            "xgboost": [[8500, 1500], [1200, 8800]]
        }

    def _get_dummy_roc_data(self):
        """Generated dummy ROC curve data"""
        fpr = np.linspace(0, 1, 100).tolist()
        tpr = (1 - np.exp(-5 * np.array(fpr))).tolist()
        return {
            "xgboost": {"fpr": fpr, "tpr": tpr, "auc": 0.971}
        }

    def _get_dummy_data_summary(self):
        """Generated dummy data summary"""
        return pd.DataFrame({
            "metric": ["Total Events", "Signal Events", "Background Events", "Signal Ratio"],
            "value": [11000000, 5500000, 5500000, 0.50]
        })

    def _get_dummy_feature_distributions(self):
        """Generated dummy feature distributions"""
        return pd.DataFrame({
            "feature": ["lepton_pT", "missing_energy_magnitude", "jet_1_pt"],
            "signal_mean": [45.2, 78.3, 112.5],
            "background_mean": [38.7, 65.1, 98.3]
        })

    def _get_dummy_predictions(self):
        """Generated dummy predictions"""
        return pd.DataFrame({
            "true_label": [1, 0, 1, 0],
            "xgboost_pred": [0.92, 0.15, 0.88, 0.23]
        })

    def _get_dummy_mass_data(self):
        """Generated dummy invariant mass data"""
        masses = np.linspace(100, 150, 50).tolist()
        counts = (np.exp(-((np.array(masses) - 125)**2) / 20) * 1000).tolist()
        return {"masses": masses, "counts": counts, "peak": 125.0}

# Created singleton instance
results_loader = ResultsLoader()
