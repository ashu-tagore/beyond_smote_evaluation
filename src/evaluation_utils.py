"""
Evaluation utilities for model performance assessment
Purpose: Implementing comprehensive metrics, confusion matrices, and statistical tests
"""

# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef
)

# Adding project modules to path (must be before local imports)
sys.path.append(str(Path(__file__).parent))

# Local imports (after path modification)
from config import SEED_VALUE, DPI_VALUE


def calculate_all_metrics(y_true, y_pred, y_proba=None):
    """
    Calculating comprehensive classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class (optional)

    Returns:
        Dictionary containing all calculated metrics
    """

    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    # Calculating specificity and sensitivity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculating G-Mean (geometric mean of sensitivity and specificity)
    metrics['g_mean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])

    # Probability-based metrics (if probabilities provided)
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = np.nan

        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics['auc_pr'] = np.nan
    else:
        metrics['auc_roc'] = np.nan
        metrics['auc_pr'] = np.nan

    # Confusion matrix components
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)

    return metrics


def create_confusion_matrix(y_true, y_pred, class_names=None, normalize=False,
                            save_path=None, title='Confusion Matrix'):
    """
    Creating and visualizing confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes (default: ['Background', 'Signal'])
        normalize: Whether to normalize confusion matrix
        save_path: Path to save figure
        title: Title for plot

    Returns:
        Confusion matrix array
    """

    # Computing confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalizing if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Setting default class names
    if class_names is None:
        class_names = ['Background (0)', 'Signal (1)']

    # Creating figure
    plt.figure(figsize=(8, 6))

    # Creating heatmap
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

    return cm


def plot_roc_curve(y_true, y_proba, model_name='Model', save_path=None):
    """
    Plotting Receiver Operating Characteristic (ROC) curve

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        model_name: Name of model for legend
        save_path: Path to save figure

    Returns:
        Tuple containing (fpr, tpr, thresholds, auc_score)
    """

    # Computing ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    # Creating figure
    plt.figure(figsize=(8, 6))

    # Plotting ROC curve
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')

    # Plotting diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.show()

    return fpr, tpr, thresholds, auc_score


def plot_precision_recall_curve(y_true, y_proba, model_name='Model', save_path=None):
    """
    Plotting Precision-Recall curve

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        model_name: Name of model for legend
        save_path: Path to save figure

    Returns:
        Tuple containing (precision, recall, thresholds, avg_precision)
    """

    # Computing precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    # Creating figure
    plt.figure(figsize=(8, 6))

    # Plotting PR curve
    plt.plot(recall, precision, linewidth=2,
             label=f'{model_name} (AP = {avg_precision:.3f})')

    # Plotting baseline (proportion of positive class)
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                label=f'Baseline (Positive Rate = {baseline:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")

    plt.show()

    return precision, recall, thresholds, avg_precision


def compare_multiple_roc_curves(results_dict, save_path=None):
    """
    Plotting multiple ROC curves on same figure for comparison

    Args:
        results_dict: Dictionary mapping model names to (y_true, y_proba) tuples
        save_path: Path to save figure

    Returns:
        Dictionary mapping model names to AUC scores
    """

    plt.figure(figsize=(10, 8))

    auc_scores = {}

    # Plotting ROC curve for each model
    for model_name, (y_true, y_proba) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        auc_scores[model_name] = auc_score

        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')

    # Plotting diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"ROC comparison saved to {save_path}")

    plt.show()

    return auc_scores


def generate_classification_report(y_true, y_pred, class_names=None,
                                   save_path=None):
    """
    Generating detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes
        save_path: Path to save report as text file

    Returns:
        Classification report as string
    """

    # Setting default class names
    if class_names is None:
        class_names = ['Background', 'Signal']

    # Generating report
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(report)

    # Saving report if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")

    return report


def perform_statistical_significance_test(scores_1, scores_2, test_type='paired_t',
                                          alpha=0.05):
    """
    Performing statistical test to compare two sets of scores

    Args:
        scores_1: First set of scores (e.g., cross-validation scores)
        scores_2: Second set of scores
        test_type: Type of test ('paired_t' or 'wilcoxon')
        alpha: Significance level

    Returns:
        Dictionary containing test results
    """

    # Converting to numpy arrays
    scores_1 = np.array(scores_1)
    scores_2 = np.array(scores_2)

    # Performing selected test
    if test_type == 'paired_t':
        statistic, p_value = stats.ttest_rel(scores_1, scores_2)
        test_name = 'Paired t-test'
    elif test_type == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(scores_1, scores_2)
        test_name = 'Wilcoxon signed-rank test'
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    # Computing effect size (Cohen's d)
    mean_diff = scores_1.mean() - scores_2.mean()
    pooled_std = np.sqrt((scores_1.std()**2 + scores_2.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Determining significance
    is_significant = p_value < alpha

    results = {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'is_significant': is_significant,
        'mean_diff': mean_diff,
        'cohens_d': cohens_d,
        'scores_1_mean': scores_1.mean(),
        'scores_1_std': scores_1.std(),
        'scores_2_mean': scores_2.mean(),
        'scores_2_std': scores_2.std()
    }

    return results


def print_statistical_test_results(results):
    """
    Printing formatted statistical test results

    Args:
        results: Dictionary from perform_statistical_significance_test()
    """

    print("="*70)
    print(f"STATISTICAL SIGNIFICANCE TEST: {results['test_name']}")
    print("="*70)

    print(f"\nTest Statistic: {results['statistic']:.4f}")
    print(f"P-value: {results['p_value']:.6f}")
    print(f"Alpha: {results['alpha']}")
    print(f"Significant: {'YES' if results['is_significant'] else 'NO'}")

    print(f"\nScores 1: {results['scores_1_mean']:.4f} (+/- {results['scores_1_std']:.4f})")
    print(f"Scores 2: {results['scores_2_mean']:.4f} (+/- {results['scores_2_std']:.4f})")
    print(f"Mean Difference: {results['mean_diff']:.4f}")
    print(f"Cohen's d (Effect Size): {results['cohens_d']:.4f}")

    # Interpreting effect size
    abs_d = abs(results['cohens_d'])
    if abs_d < 0.2:
        effect = "negligible"
    elif abs_d < 0.5:
        effect = "small"
    elif abs_d < 0.8:
        effect = "medium"
    else:
        effect = "large"

    print(f"Effect Size Interpretation: {effect}")
    print("="*70)


def compare_all_methods_statistical(results_df, metric_column='f1_score',
                                    baseline_method='baseline', alpha=0.05):
    """
    Comparing all methods against baseline using statistical tests

    Args:
        results_df: DataFrame with columns for method, fold, and metrics
        metric_column: Name of metric column to compare
        baseline_method: Name of baseline method
        alpha: Significance level

    Returns:
        DataFrame containing comparison results
    """

    print(f"Comparing all methods against {baseline_method}...")

    # Extracting baseline scores
    baseline_scores = results_df[results_df['method'] == baseline_method][metric_column].values

    if len(baseline_scores) == 0:
        raise ValueError(f"Baseline method '{baseline_method}' not found in results")

    # Initializing results list
    comparison_results = []

    # Getting unique methods
    methods = results_df['method'].unique()

    for method in methods:
        if method == baseline_method:
            continue

        # Extracting method scores
        method_scores = results_df[results_df['method'] == method][metric_column].values

        if len(method_scores) == 0:
            continue

        # Performing statistical test
        test_results = perform_statistical_significance_test(
            method_scores, baseline_scores, test_type='paired_t', alpha=alpha
        )

        # Storing results
        comparison_results.append({
            'method': method,
            'mean_score': test_results['scores_1_mean'],
            'std_score': test_results['scores_1_std'],
            'improvement': test_results['mean_diff'],
            'p_value': test_results['p_value'],
            'significant': test_results['is_significant'],
            'cohens_d': test_results['cohens_d']
        })

    # Creating DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values('mean_score', ascending=False)

    return comparison_df


def create_metrics_summary_table(results_dict):
    """
    Creating summary table from multiple evaluation results

    Args:
        results_dict: Dictionary mapping model names to metrics dictionaries

    Returns:
        DataFrame containing summary of all metrics
    """

    summary_data = []

    for model_name, metrics in results_dict.items():
        row = {'model': model_name}
        row.update(metrics)
        summary_data.append(row)

    # Creating DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Ordering columns
    priority_cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score',
                     'auc_roc', 'auc_pr', 'g_mean', 'mcc']

    other_cols = [col for col in summary_df.columns if col not in priority_cols]
    ordered_cols = [col for col in priority_cols if col in summary_df.columns] + other_cols

    summary_df = summary_df[ordered_cols]

    return summary_df


def aggregate_cv_results(cv_results_dict, metric_names=None):
    """
    Aggregating cross-validation results across multiple models

    Args:
        cv_results_dict: Dictionary mapping model names to CV results
        metric_names: List of metric names to include

    Returns:
        DataFrame containing aggregated CV statistics
    """

    if metric_names is None:
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    aggregated_data = []

    for model_name, cv_results in cv_results_dict.items():
        row = {'model': model_name}

        for metric in metric_names:
            test_key = f'test_{metric}'

            if test_key in cv_results:
                scores = cv_results[test_key]
                row[f'{metric}_mean'] = scores.mean()
                row[f'{metric}_std'] = scores.std()
                row[f'{metric}_min'] = scores.min()
                row[f'{metric}_max'] = scores.max()

        aggregated_data.append(row)

    # Creating DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)

    return aggregated_df


if __name__ == "__main__":
    print("="*70)
    print("EVALUATION UTILITIES MODULE TEST")
    print("="*70)

    # Generating synthetic predictions for testing
    np.random.seed(SEED_VALUE)

    n_samples = 1000
    y_true_test = np.random.randint(0, 2, n_samples)
    y_pred_test = y_true_test.copy()

    # Introducing some errors
    error_indices = np.random.choice(n_samples, size=100, replace=False)
    y_pred_test[error_indices] = 1 - y_pred_test[error_indices]

    # Generating probabilities
    y_proba_test = np.random.rand(n_samples)

    print("\nTest 1: Calculating all metrics...")
    metrics_test = calculate_all_metrics(y_true_test, y_pred_test, y_proba_test)
    for metric_name, metric_value in metrics_test.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("\nTest 2: Creating confusion matrix...")
    cm_test = create_confusion_matrix(y_true_test, y_pred_test)

    print("\nTest 3: Plotting ROC curve...")
    plot_roc_curve(y_true_test, y_proba_test, model_name='Test Model')

    print("\nTest 4: Plotting Precision-Recall curve...")
    plot_precision_recall_curve(y_true_test, y_proba_test, model_name='Test Model')

    print("\nTest 5: Statistical significance test...")
    scores_a = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
    scores_b = np.array([0.82, 0.83, 0.81, 0.84, 0.80])
    stat_results = perform_statistical_significance_test(scores_a, scores_b)
    print_statistical_test_results(stat_results)

    print("\nALL TESTS COMPLETED SUCCESSFULLY")
