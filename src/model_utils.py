"""
Model utilities for training and evaluating machine learning models
Purpose: Implementing model initialization, training, evaluation, and tuning functions
"""

# Standard library imports
import sys
import time
from pathlib import Path

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Importing XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# Adding project modules to path
sys.path.append(str(Path(__file__).parent))

# Local imports
from config import (
    SEED_VALUE,
    LOGIT_CONFIG,
    RF_CONFIG,
    XGB_CONFIG,
    SVM_CONFIG,
    MLP_CONFIG,
    PARALLEL_JOBS,
    SKLEARN_STORAGE
)


def initialize_all_models(use_class_weights=False, class_weight_dict=None):
    """
    Initializing dictionary containing all machine learning model instances

    Args:
        use_class_weights: Whether to apply class weights for imbalanced learning
        class_weight_dict: Custom class weight dictionary (if None, uses 'balanced')

    Returns:
        Dictionary mapping model names to configured model instances
    """

    # Determining class weight parameter
    if use_class_weights:
        weight_param = class_weight_dict if class_weight_dict else 'balanced'
    else:
        weight_param = None

    # Initializing model dictionary
    models = {}

    # Configuring Logistic Regression
    models['logistic_regression'] = LogisticRegression(
        max_iter=LOGIT_CONFIG['max_iter'],
        random_state=LOGIT_CONFIG['random_state'],
        n_jobs=LOGIT_CONFIG['n_jobs'],
        solver=LOGIT_CONFIG['solver'],
        class_weight=weight_param
    )

    # Configuring Random Forest
    models['random_forest'] = RandomForestClassifier(
        n_estimators=RF_CONFIG['n_estimators'],
        max_depth=RF_CONFIG['max_depth'],
        min_samples_split=RF_CONFIG['min_samples_split'],
        min_samples_leaf=RF_CONFIG['min_samples_leaf'],
        random_state=RF_CONFIG['random_state'],
        n_jobs=RF_CONFIG['n_jobs'],
        verbose=RF_CONFIG['verbose'],
        class_weight=weight_param
    )

    # Configuring XGBoost if available
    if XGBOOST_AVAILABLE:
        # Computing scale_pos_weight for XGBoost class imbalance handling
        if use_class_weights and class_weight_dict:
            # Extracting weights if provided
            scale_pos_weight = class_weight_dict.get(0, 1) / class_weight_dict.get(1, 1)
        else:
            scale_pos_weight = 1

        models['xgboost'] = XGBClassifier(
            n_estimators=XGB_CONFIG['n_estimators'],
            max_depth=XGB_CONFIG['max_depth'],
            learning_rate=XGB_CONFIG['learning_rate'],
            subsample=XGB_CONFIG['subsample'],
            colsample_bytree=XGB_CONFIG['colsample_bytree'],
            random_state=XGB_CONFIG['random_state'],
            n_jobs=XGB_CONFIG['n_jobs'],
            verbosity=XGB_CONFIG['verbosity'],
            eval_metric=XGB_CONFIG['eval_metric'],
            scale_pos_weight=scale_pos_weight if use_class_weights else 1
        )

    # Configuring Support Vector Machine
    models['svm'] = SVC(
        kernel=SVM_CONFIG['kernel'],
        probability=SVM_CONFIG['probability'],
        random_state=SVM_CONFIG['random_state'],
        max_iter=SVM_CONFIG['max_iter'],
        class_weight=weight_param
    )

    # Configuring Multi-Layer Perceptron
    models['mlp'] = MLPClassifier(
        hidden_layer_sizes=MLP_CONFIG['hidden_layer_sizes'],
        activation=MLP_CONFIG['activation'],
        solver=MLP_CONFIG['solver'],
        max_iter=MLP_CONFIG['max_iter'],
        random_state=MLP_CONFIG['random_state'],
        early_stopping=MLP_CONFIG['early_stopping'],
        validation_fraction=MLP_CONFIG['validation_fraction']
    )

    return models


def train_single_model(model, X_train, y_train, model_name='model', verbose=True):
    """
    Training a single machine learning model on provided data

    Args:
        model: Scikit-learn compatible model instance
        X_train: Training feature matrix
        y_train: Training target vector
        model_name: Name of model for logging
        verbose: Whether to print training information

    Returns:
        Tuple containing (trained_model, training_time)
    """

    if verbose:
        print(f"Training {model_name}...")

    # Recording start time
    start_time = time.time()

    # Fitting model
    model.fit(X_train, y_train)

    # Computing training time
    training_time = time.time() - start_time

    if verbose:
        print(f"Training completed in {training_time:.2f} seconds")

    return model, training_time


def evaluate_model_predictions(model, X_test, y_test, model_name='model', verbose=True):
    """
    Generating predictions and prediction probabilities from trained model

    Args:
        model: Trained model instance
        X_test: Test feature matrix
        y_test: Test target vector
        model_name: Name of model for logging
        verbose: Whether to print evaluation information

    Returns:
        Dictionary containing predictions and probabilities
    """

    if verbose:
        print(f"Evaluating {model_name}...")

    # Recording prediction time
    start_time = time.time()

    # Generating predictions
    y_pred = model.predict(X_test)

    # Generating prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    else:
        y_proba = None
        if verbose:
            print(f"Note: {model_name} does not support probability predictions")

    # Computing prediction time
    prediction_time = time.time() - start_time

    results = {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'prediction_time': prediction_time
    }

    if verbose:
        print(f"Prediction completed in {prediction_time:.4f} seconds")

    return results


def perform_cross_validation(model, X, y, cv_folds=5, scoring_metrics=None,
                             model_name='model', verbose=True):
    """
    Performing stratified k-fold cross-validation on model

    Args:
        model: Scikit-learn compatible model instance
        X: Feature matrix
        y: Target vector
        cv_folds: Number of cross-validation folds
        scoring_metrics: List of metric names to compute
        model_name: Name of model for logging
        verbose: Whether to print CV information

    Returns:
        Dictionary containing cross-validation results
    """

    if verbose:
        print(f"Performing {cv_folds}-fold cross-validation for {model_name}...")

    # Setting default scoring metrics if not provided
    if scoring_metrics is None:
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Creating stratified k-fold splitter
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=SEED_VALUE)

    # Recording start time
    start_time = time.time()

    # Performing cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv_splitter,
        scoring=scoring_metrics,
        return_train_score=True,
        n_jobs=PARALLEL_JOBS,
        verbose=0
    )

    # Computing total CV time
    cv_time = time.time() - start_time

    if verbose:
        print(f"Cross-validation completed in {cv_time:.2f} seconds")
        print(f"Results across {cv_folds} folds:")
        for metric in scoring_metrics:
            test_key = f'test_{metric}'
            if test_key in cv_results:
                mean_score = cv_results[test_key].mean()
                std_score = cv_results[test_key].std()
                print(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")

    # Adding CV time to results
    cv_results['cv_time'] = cv_time

    return cv_results


def tune_model_hyperparameters(model_type, X, y, param_grid, cv_folds=5,
                               scoring_metric='f1', verbose=True):
    """
    Performing grid search for hyperparameter tuning

    Args:
        model_type: String indicating model type ('logistic', 'random_forest', etc.)
        X: Feature matrix
        y: Target vector
        param_grid: Dictionary of parameters to search
        cv_folds: Number of cross-validation folds
        scoring_metric: Metric to optimize
        verbose: Whether to print tuning information

    Returns:
        Tuple containing (best_model, best_params, best_score, grid_search_results)
    """

    if verbose:
        print(f"Tuning hyperparameters for {model_type}...")
        print(f"Parameter grid: {param_grid}")
        print(f"Optimizing for: {scoring_metric}")

    # Getting base model
    models = initialize_all_models()

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    base_model = models[model_type]

    # Creating stratified k-fold splitter
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=SEED_VALUE)

    # Initializing grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=scoring_metric,
        n_jobs=PARALLEL_JOBS,
        verbose=2 if verbose else 0,
        return_train_score=True
    )

    # Recording start time
    start_time = time.time()

    # Performing grid search
    grid_search.fit(X, y)

    # Computing tuning time
    tuning_time = time.time() - start_time

    if verbose:
        print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best {scoring_metric} score: {grid_search.best_score_:.4f}")

    return (grid_search.best_estimator_,
            grid_search.best_params_,
            grid_search.best_score_,
            grid_search.cv_results_)


def extract_feature_importance(model, feature_names=None, top_n=20):
    """
    Extracting feature importance from trained model

    Args:
        model: Trained model with feature importance attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame containing features and their importance scores
    """

    # Checking if model has feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Using absolute coefficient values for linear models
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not have feature importance or coefficients")
        return None

    # Creating feature names if not provided
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]

    # Creating DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sorting by importance
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Returning top N features
    return importance_df.head(top_n)


def save_trained_model(model, model_name, save_directory=None):
    """
    Saving trained model to disk using joblib

    Args:
        model: Trained model instance
        model_name: Name for saved model file
        save_directory: Directory to save model (default: SKLEARN_STORAGE)

    Returns:
        Path to saved model file
    """

    if save_directory is None:
        save_directory = SKLEARN_STORAGE

    # Creating directory if not existing
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Creating filename
    model_filename = f"{model_name}.pkl"
    model_path = Path(save_directory) / model_filename

    # Saving model
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    return model_path


def load_trained_model(model_name, load_directory=None):
    """
    Loading trained model from disk

    Args:
        model_name: Name of saved model file
        load_directory: Directory containing model (default: SKLEARN_STORAGE)

    Returns:
        Loaded model instance
    """

    if load_directory is None:
        load_directory = SKLEARN_STORAGE

    # Creating filename
    model_filename = f"{model_name}.pkl"
    model_path = Path(load_directory) / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Loading model
    model = joblib.load(model_path)

    print(f"Model loaded from {model_path}")

    return model


def create_model_comparison_table(results_dict):
    """
    Creating comparison table from multiple model results

    Args:
        results_dict: Dictionary mapping model names to their CV results

    Returns:
        DataFrame containing comparison metrics for all models
    """

    comparison_data = []

    for model_name, cv_results in results_dict.items():
        # Extracting mean scores for each metric
        row = {'model': model_name}

        for key, values in cv_results.items():
            if key.startswith('test_'):
                metric_name = key.replace('test_', '')
                row[f'{metric_name}_mean'] = values.mean()
                row[f'{metric_name}_std'] = values.std()

        comparison_data.append(row)

    # Creating DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Sorting by F1 score if available
    if 'f1_mean' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('f1_mean', ascending=False)

    return comparison_df


def train_all_models(X_train, y_train, X_test, y_test, use_class_weights=False,
                     class_weight_dict=None, verbose=True):
    """
    Training all available models and collecting results

    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Test feature matrix
        y_test: Test target vector (used in evaluation)
        use_class_weights: Whether to apply class weights
        class_weight_dict: Custom class weight dictionary
        verbose: Whether to print training information

    Returns:
        Dictionary containing trained models and their results
    """

    # Initializing all models
    all_models = initialize_all_models(use_class_weights, class_weight_dict)

    results = {}

    for model_name, model_instance in all_models.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training: {model_name}")
            print(f"{'='*70}")

        # Training model
        trained_model, train_time = train_single_model(
            model_instance, X_train, y_train, model_name, verbose
        )

        # Evaluating model (y_test is used here)
        eval_results = evaluate_model_predictions(
            trained_model, X_test, y_test, model_name, verbose
        )

        # Storing results
        results[model_name] = {
            'model': trained_model,
            'training_time': train_time,
            'prediction_time': eval_results['prediction_time'],
            'y_pred': eval_results['y_pred'],
            'y_proba': eval_results['y_proba']
        }

    return results


def perform_cv_all_models(X, y, cv_folds=5, scoring_metrics=None,
                          use_class_weights=False, verbose=True):
    """
    Performing cross-validation on all models

    Args:
        X: Feature matrix
        y: Target vector
        cv_folds: Number of cross-validation folds
        scoring_metrics: List of metrics to compute
        use_class_weights: Whether to apply class weights
        verbose: Whether to print CV information

    Returns:
        Dictionary containing CV results for all models
    """

    # Initializing all models
    all_models = initialize_all_models(use_class_weights)

    cv_results_all = {}

    for model_name, model_instance in all_models.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Cross-validating: {model_name}")
            print(f"{'='*70}")

        # Performing cross-validation
        model_cv_results = perform_cross_validation(
            model_instance, X, y, cv_folds, scoring_metrics, model_name, verbose
        )

        cv_results_all[model_name] = model_cv_results

    return cv_results_all


if __name__ == "__main__":
    print("="*70)
    print("MODEL UTILITIES MODULE TEST")
    print("="*70)

    # Generating synthetic data for testing
    from sklearn.datasets import make_classification

    print("\nGenerating synthetic dataset...")
    X_synthetic, y_synthetic = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=SEED_VALUE
    )

    print(f"Dataset shape: {X_synthetic.shape}")
    print(f"Class distribution: {np.bincount(y_synthetic)}")

    print("\nTest 1: Initializing all models...")
    test_models = initialize_all_models()
    print(f"Initialized {len(test_models)} models")

    print("\nTest 2: Training single model (Logistic Regression)...")
    lr_model, lr_train_time = train_single_model(
        test_models['logistic_regression'],
        X_synthetic[:800],
        y_synthetic[:800],
        'Logistic Regression'
    )

    print("\nTest 3: Evaluating model...")
    lr_eval_results = evaluate_model_predictions(
        lr_model,
        X_synthetic[800:],
        y_synthetic[800:],
        'Logistic Regression'
    )

    print("\nTest 4: Performing cross-validation...")
    rf_cv_results = perform_cross_validation(
        test_models['random_forest'],
        X_synthetic,
        y_synthetic,
        cv_folds=3,
        model_name='Random Forest'
    )

    print("\nTest 5: Extracting feature importance...")
    rf_model, _ = train_single_model(
        test_models['random_forest'],
        X_synthetic,
        y_synthetic,
        'Random Forest',
        verbose=False
    )
    rf_importance_df = extract_feature_importance(rf_model, top_n=10)
    print(rf_importance_df)

    print("\nALL TESTS COMPLETED SUCCESSFULLY")
