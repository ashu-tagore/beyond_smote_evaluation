"""
Visualization utilities for data analysis and results presentation
Purpose: Implementing plotting functions for EDA, model comparison, and results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Adding project modules to path
sys.path.append(str(Path(__file__).parent))
from config import *


def plot_class_distribution(labels, class_names=None, save_path=None,
                            figure_title='Class Distribution'):
    """
    Plotting class distribution as pie chart and bar chart

    Args:
        labels: Array of class labels
        class_names: Names for classes
        save_path: Path to save figure
        figure_title: Title for plot
    """

    # Setting default class names
    if class_names is None:
        class_names = ['Background (0)', 'Signal (1)']

    # Computing class counts
    unique, counts = np.unique(labels, return_counts=True)

    # Creating subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    colors = ['skyblue', 'lightcoral']
    axes[0].pie(counts, labels=class_names, autopct='%1.2f%%',
                colors=colors, startangle=90)
    axes[0].set_title('Class Distribution (Pie Chart)')

    # Bar chart
    axes[1].bar(class_names, counts, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Class Distribution (Bar Chart)')
    axes[1].grid(axis='y', alpha=0.3)

    # Adding count labels on bars
    for i, count in enumerate(counts):
        axes[1].text(i, count, f'{count:,}', ha='center', va='bottom')

    # Adding imbalance ratio
    ratio = counts.min() / counts.max()
    fig.suptitle(f'{figure_title}\nImbalance Ratio: {ratio:.3f}:1',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")

    plt.show()


def plot_feature_distributions(data, features=None, target_col=None,
                               save_path=None, max_features=9):
    """
    Plotting distributions of multiple features using histograms

    Args:
        data: DataFrame containing features
        features: List of feature names to plot
        target_col: Target column for color coding
        save_path: Path to save figure
        max_features: Maximum number of features to plot
    """

    if features is None:
        features = ALL_FEATURES[:max_features]

    # Limiting to max_features
    features = features[:max_features]

    # Determining grid size
    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    # Creating figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    # Plotting each feature
    for idx, feature in enumerate(features):
        ax = axes[idx]

        if target_col is not None and target_col in data.columns:
            # Plotting by class
            for class_val in sorted(data[target_col].unique()):
                subset = data[data[target_col] == class_val][feature]
                ax.hist(subset, bins=30, alpha=0.6,
                       label=f'Class {class_val}', edgecolor='black')
            ax.legend()
        else:
            # Plotting without class separation
            ax.hist(data[feature], bins=30, alpha=0.7,
                   color='steelblue', edgecolor='black')

        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution: {feature}')
        ax.grid(axis='y', alpha=0.3)

    # Hiding unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Feature distributions saved to {save_path}")

    plt.show()


def plot_correlation_heatmap(data, features=None, save_path=None,
                             figure_title='Feature Correlation Matrix'):
    """
    Creating correlation heatmap for features

    Args:
        data: DataFrame containing features
        features: List of features to include
        save_path: Path to save figure
        figure_title: Title for plot
    """

    if features is None:
        features = ALL_FEATURES

    # Computing correlation matrix
    correlation_matrix = data[features].corr()

    # Creating figure
    plt.figure(figsize=(14, 12))

    # Creating heatmap
    sns.heatmap(correlation_matrix, annot=False, cmap=HEAT_COLORMAP,
                center=0, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})

    plt.title(figure_title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")

    plt.show()


def plot_feature_boxplots(data, features=None, target_col=None,
                          save_path=None, max_features=9):
    """
    Creating box plots for outlier detection

    Args:
        data: DataFrame containing features
        features: List of feature names
        target_col: Target column for grouping
        save_path: Path to save figure
        max_features: Maximum features to plot
    """

    if features is None:
        features = ALL_FEATURES[:max_features]

    # Limiting features
    features = features[:max_features]

    # Determining grid size
    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    # Creating figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    # Plotting box plots
    for idx, feature in enumerate(features):
        ax = axes[idx]

        if target_col is not None and target_col in data.columns:
            # Box plot by class
            data.boxplot(column=feature, by=target_col, ax=ax)
            ax.set_title(f'{feature}')
            ax.set_xlabel('Class')
        else:
            # Single box plot
            ax.boxplot(data[feature].dropna())
            ax.set_title(f'{feature}')
            ax.set_xticklabels([''])

        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)

    # Hiding unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Feature Box Plots for Outlier Detection',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Box plots saved to {save_path}")

    plt.show()


def create_comparison_barplot(results_df, metric_col='f1_score',
                              group_col='model', save_path=None,
                              figure_title='Model Performance Comparison'):
    """
    Creating bar plot comparing performance across groups

    Args:
        results_df: DataFrame with results
        metric_col: Column containing metric values
        group_col: Column for grouping bars
        save_path: Path to save figure
        figure_title: Title for plot
    """

    # Creating figure
    plt.figure(figsize=(12, 6))

    # Creating bar plot
    if group_col in results_df.columns:
        # Grouped bar plot
        groups = results_df[group_col].unique()
        x_positions = np.arange(len(groups))

        plt.bar(x_positions, results_df.groupby(group_col)[metric_col].mean(),
                color='steelblue', alpha=0.7, edgecolor='black')
        plt.xticks(x_positions, groups, rotation=45, ha='right')
    else:
        # Simple bar plot
        plt.bar(range(len(results_df)), results_df[metric_col],
                color='steelblue', alpha=0.7, edgecolor='black')

    plt.xlabel(group_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    plt.title(figure_title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Comparison bar plot saved to {save_path}")

    plt.show()


def plot_performance_heatmap(results_df, row_col='resampling_method',
                            col_col='model', value_col='f1_score',
                            save_path=None):
    """
    Creating heatmap showing performance across methods and models

    Args:
        results_df: DataFrame with experimental results
        row_col: Column for rows
        col_col: Column for columns
        value_col: Column containing values
        save_path: Path to save figure
    """

    # Pivoting data for heatmap
    heatmap_data = results_df.pivot_table(
        values=value_col,
        index=row_col,
        columns=col_col,
        aggfunc='mean'
    )

    # Creating figure
    plt.figure(figsize=(12, 8))

    # Creating heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap=HEAT_COLORMAP,
                cbar_kws={'label': value_col.replace('_', ' ').title()},
                linewidths=0.5)

    plt.title(f'{value_col.replace("_", " ").title()} Heatmap',
              fontsize=14, fontweight='bold')
    plt.xlabel(col_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(row_col.replace('_', ' ').title(), fontsize=12)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Performance heatmap saved to {save_path}")

    plt.show()


def plot_learning_curves(train_scores, val_scores, epochs=None,
                         metric_name='Loss', save_path=None):
    """
    Plotting training and validation curves

    Args:
        train_scores: Array of training scores
        val_scores: Array of validation scores
        epochs: Array of epoch numbers
        metric_name: Name of metric being plotted
        save_path: Path to save figure
    """

    if epochs is None:
        epochs = np.arange(1, len(train_scores) + 1)

    # Creating figure
    plt.figure(figsize=(10, 6))

    # Plotting curves
    plt.plot(epochs, train_scores, marker='o', label=f'Training {metric_name}',
             linewidth=2, markersize=4)
    plt.plot(epochs, val_scores, marker='s', label=f'Validation {metric_name}',
             linewidth=2, markersize=4)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Learning Curves: {metric_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Learning curves saved to {save_path}")

    plt.show()


def plot_metric_comparison_across_folds(cv_results, metric_name='f1',
                                        save_path=None):
    """
    Plotting metric values across cross-validation folds

    Args:
        cv_results: Dictionary from cross_validate
        metric_name: Metric to visualize
        save_path: Path to save figure
    """

    test_key = f'test_{metric_name}'

    if test_key not in cv_results:
        print(f"Metric {metric_name} not found in results")
        return

    scores = cv_results[test_key]
    folds = np.arange(1, len(scores) + 1)

    # Creating figure
    plt.figure(figsize=(10, 6))

    # Plotting scores
    plt.plot(folds, scores, marker='o', linewidth=2, markersize=8,
             color='steelblue', label='Fold Scores')

    # Adding mean line
    mean_score = scores.mean()
    plt.axhline(y=mean_score, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_score:.4f}')

    # Adding confidence interval
    std_score = scores.std()
    plt.fill_between(folds, mean_score - std_score, mean_score + std_score,
                     alpha=0.2, color='red')

    plt.xlabel('Fold Number', fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.title(f'{metric_name.upper()} Across Cross-Validation Folds',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(folds)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"CV fold comparison saved to {save_path}")

    plt.show()


def plot_feature_importance_horizontal(importance_df, top_n=15,
                                       save_path=None):
    """
    Creating horizontal bar plot for feature importance

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save figure
    """

    # Selecting top features
    top_features = importance_df.head(top_n).sort_values('importance')

    # Creating figure
    plt.figure(figsize=(10, 8))

    # Creating horizontal bar plot
    plt.barh(top_features['feature'], top_features['importance'],
             color='steelblue', alpha=0.7, edgecolor='black')

    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features',
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    plt.show()


def create_pairplot_subset(data, features_subset=None, target_col=None,
                           save_path=None):
    """
    Creating pairwise scatter plots for feature relationships

    Args:
        data: DataFrame containing features
        features_subset: List of features to include
        target_col: Target column for color coding
        save_path: Path to save figure
    """

    if features_subset is None:
        # Selecting first 5 features by default
        features_subset = ALL_FEATURES[:5]

    # Preparing data for pairplot
    if target_col and target_col in data.columns:
        plot_data = data[features_subset + [target_col]]
        pairplot = sns.pairplot(plot_data, hue=target_col, diag_kind='hist',
                                plot_kws={'alpha': 0.6}, height=2.5)
    else:
        plot_data = data[features_subset]
        pairplot = sns.pairplot(plot_data, diag_kind='hist',
                                plot_kws={'alpha': 0.6}, height=2.5)

    pairplot.fig.suptitle('Pairwise Feature Relationships',
                          y=1.02, fontsize=14, fontweight='bold')

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Pairplot saved to {save_path}")

    plt.show()


def plot_computational_cost_comparison(results_df, time_col='training_time',
                                      group_col='resampling_method',
                                      save_path=None):
    """
    Visualizing computational cost across methods

    Args:
        results_df: DataFrame with timing information
        time_col: Column containing time values
        group_col: Column for grouping
        save_path: Path to save figure
    """

    # Creating figure
    plt.figure(figsize=(12, 6))

    # Aggregating by group
    time_summary = results_df.groupby(group_col)[time_col].mean().sort_values()

    # Creating bar plot
    plt.bar(range(len(time_summary)), time_summary.values,
            color='coral', alpha=0.7, edgecolor='black')
    plt.xticks(range(len(time_summary)), time_summary.index,
               rotation=45, ha='right')

    plt.xlabel(group_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Average Time (seconds)', fontsize=12)
    plt.title('Computational Cost Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Computational cost plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("VISUALIZATION UTILITIES MODULE TEST")
    print("=" * 70)

    # Generating synthetic data for testing
    np.random.seed(SEED_VALUE)

    # Creating sample dataset
    n_samples = 1000
    test_data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples) * 2,
        'feature3': np.random.randn(n_samples) * 0.5,
        'label': np.random.randint(0, 2, n_samples)
    })

    print("\nTest 1: Class distribution plot...")
    plot_class_distribution(test_data['label'])

    print("\nTest 2: Feature distributions...")
    plot_feature_distributions(test_data,
                               features=['feature1', 'feature2', 'feature3'],
                               target_col='label')

    print("\nTest 3: Correlation heatmap...")
    plot_correlation_heatmap(test_data,
                            features=['feature1', 'feature2', 'feature3'])

    print("\nTest 4: Box plots...")
    plot_feature_boxplots(test_data,
                         features=['feature1', 'feature2', 'feature3'],
                         target_col='label')

    print("ALL VISUALIZATION TESTS COMPLETED")