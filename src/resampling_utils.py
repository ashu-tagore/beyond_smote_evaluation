"""
Resampling techniques implementation for imbalanced classification
Purpose: Implementing various oversampling, undersampling, and hybrid methods
"""

# Standard library imports
import sys
from pathlib import Path
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    ADASYN
)
from imblearn.under_sampling import (
    RandomUnderSampler,
    TomekLinks,
    NearMiss
)
from imblearn.combine import (
    SMOTEENN,
    SMOTETomek
)

# Local imports
sys.path.append(str(Path(__file__).parent))
from config import (
    SEED_VALUE,
    ALL_FEATURES,
    TARGET_COL,
    RESAMPLED_DIR,
    DPI_VALUE,
    ROS_CONFIG,
    RUS_CONFIG
)


def initialize_resampling_methods():
    """
    Initializing dictionary containing all resampling technique instances

    Returns:
        Dictionary mapping method names to configured resampler objects
    """

    # Configuring methods without n_jobs parameter to avoid errors
    smote_config = {'k_neighbors': 5, 'random_state': SEED_VALUE}
    bsmote_config = {'k_neighbors': 5, 'random_state': SEED_VALUE, 'kind': 'borderline-1'}
    adasyn_config = {'n_neighbors': 5, 'random_state': SEED_VALUE}
    tomek_config = {}
    nmiss_config = {'version': 1, 'n_neighbors': 3}
    smtomek_config = {'random_state': SEED_VALUE}
    smenn_config = {'random_state': SEED_VALUE}

    resampling_techniques = {
        'baseline': None,

        # Oversampling techniques
        'random_over': RandomOverSampler(**ROS_CONFIG),
        'smote': SMOTE(**smote_config),
        'borderline_smote': BorderlineSMOTE(**bsmote_config),
        'adasyn': ADASYN(**adasyn_config),

        # Undersampling techniques
        'random_under': RandomUnderSampler(**RUS_CONFIG),
        'tomek': TomekLinks(**tomek_config),
        'nearmiss': NearMiss(**nmiss_config),

        # Hybrid techniques
        'smote_tomek': SMOTETomek(**smtomek_config),
        'smote_enn': SMOTEENN(**smenn_config),

        # Algorithm-level approach
        'class_weight': 'use_weights'
    }

    return resampling_techniques


def execute_resampling(features, labels, technique='smote', display_info=True):
    """
    Executing specified resampling technique on dataset

    Args:
        features: Feature matrix (numpy array or DataFrame)
        labels: Target vector (numpy array or Series)
        technique: Name of resampling method to apply
        display_info: Whether to print resampling statistics

    Returns:
        resampled_features: Transformed feature matrix
        resampled_labels: Transformed target vector
    """

    if display_info:
        print(f"Applying resampling technique: {technique}")
        original_dist = Counter(labels)
        print(f"Original distribution: {dict(original_dist)}")

    # Getting resampling method
    available_methods = initialize_resampling_methods()

    if technique not in available_methods:
        available = list(available_methods.keys())
        raise ValueError(f"Unknown technique: {technique}. Available: {available}")

    resampler = available_methods[technique]

    # Handling special cases
    if technique == 'baseline':
        if display_info:
            print("Baseline selected - returning original data without modifications")
        return features, labels

    if technique == 'class_weight':
        if display_info:
            print("Class weighting selected - no data resampling performed")
            print("Note: Use compute_class_weights() for weight calculation")
        return features, labels

    # Applying resampling transformation
    try:
        resampled_features, resampled_labels = resampler.fit_resample(features, labels)

        if display_info:
            new_dist = Counter(resampled_labels)
            print(f"Resampled distribution: {dict(new_dist)}")

            size_change = len(resampled_labels) - len(labels)
            pct_change = (size_change / len(labels)) * 100
            print(f"Dataset size change: {size_change:+,} observations ({pct_change:+.2f}%)")

            # Computing new imbalance ratio
            counts = np.array(list(new_dist.values()))
            new_ratio = counts.min() / counts.max()
            print(f"New imbalance ratio: {new_ratio:.3f}:1")

        return resampled_features, resampled_labels

    except (ValueError, RuntimeError) as error:
        print(f"Error during resampling: {error}")
        raise


def measure_imbalance_ratio(label_vector):
    """
    Measuring degree of class imbalance in dataset

    Args:
        label_vector: Array of class labels

    Returns:
        Tuple containing (minority_count, majority_count, imbalance_ratio)
    """

    label_distribution = Counter(label_vector)

    if len(label_distribution) != 2:
        raise ValueError("Function supports binary classification only")

    class_counts = sorted(label_distribution.values())
    minority_samples = class_counts[0]
    majority_samples = class_counts[1]

    imbalance_metric = minority_samples / majority_samples

    return minority_samples, majority_samples, imbalance_metric


def visualize_resampling_impact(original_features, original_labels,
                                resampled_features, resampled_labels,
                                technique_name='Unknown',
                                save_path=None):
    """
    Visualizing data distribution before and after resampling

    Args:
        original_features: Original feature matrix
        original_labels: Original target vector
        resampled_features: Resampled feature matrix
        resampled_labels: Resampled target vector
        technique_name: Name of applied technique
        save_path: Optional path for saving figure
    """

    print(f"Creating visualization for {technique_name} resampling...")

    # Applying PCA for 2D visualization
    pca_transformer = PCA(n_components=2, random_state=SEED_VALUE)

    # Transforming original data
    original_reduced = pca_transformer.fit_transform(original_features)

    # Transforming resampled data using same PCA
    resampled_reduced = pca_transformer.transform(resampled_features)

    # Creating subplot figure
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plotting original distribution
    axes[0].scatter(original_reduced[original_labels == 0, 0],
                    original_reduced[original_labels == 0, 1],
                    c='blue', label='Class 0', alpha=0.5, s=10)
    axes[0].scatter(original_reduced[original_labels == 1, 0],
                    original_reduced[original_labels == 1, 1],
                    c='red', label='Class 1', alpha=0.5, s=10)
    axes[0].set_title('Original Data Distribution')
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Adding original class counts
    orig_0 = (original_labels == 0).sum()
    orig_1 = (original_labels == 1).sum()
    axes[0].text(0.02, 0.98, f'Class 0: {orig_0:,}\nClass 1: {orig_1:,}',
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plotting resampled distribution
    axes[1].scatter(resampled_reduced[resampled_labels == 0, 0],
                    resampled_reduced[resampled_labels == 0, 1],
                    c='blue', label='Class 0', alpha=0.5, s=10)
    axes[1].scatter(resampled_reduced[resampled_labels == 1, 0],
                    resampled_reduced[resampled_labels == 1, 1],
                    c='red', label='Class 1', alpha=0.5, s=10)
    axes[1].set_title(f'After {technique_name} Resampling')
    axes[1].set_xlabel('First Principal Component')
    axes[1].set_ylabel('Second Principal Component')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Adding resampled class counts
    resamp_0 = (resampled_labels == 0).sum()
    resamp_1 = (resampled_labels == 1).sum()
    axes[1].text(0.02, 0.98, f'Class 0: {resamp_0:,}\nClass 1: {resamp_1:,}',
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Saving figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI_VALUE, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()

    print("Visualization complete")


def compare_resampling_techniques(features, labels, techniques_to_compare=None):
    """
    Comparing multiple resampling techniques on same dataset

    Args:
        features: Feature matrix
        labels: Target vector
        techniques_to_compare: List of technique names (None for all)

    Returns:
        Dictionary containing results for each technique
    """

    print("Initiating comparison of resampling techniques...")

    if techniques_to_compare is None:
        available_methods = initialize_resampling_methods()
        techniques_to_compare = [k for k in available_methods.keys()
                                if k not in ['baseline', 'class_weight']]

    comparison_results = {}

    for technique in techniques_to_compare:
        print(f"\nEvaluating technique: {technique}")
        print("-" * 50)

        try:
            _, resampled_y = execute_resampling(features, labels,
                                                technique,
                                                display_info=True)

            # Storing results
            comparison_results[technique] = {
                'original_size': len(labels),
                'resampled_size': len(resampled_y),
                'size_change': len(resampled_y) - len(labels),
                'original_distribution': dict(Counter(labels)),
                'resampled_distribution': dict(Counter(resampled_y)),
                'execution_status': 'success'
            }

        except (ValueError, RuntimeError) as error:
            print(f"Failed to apply {technique}: {error}")
            comparison_results[technique] = {
                'execution_status': 'failed',
                'error_message': str(error)
            }

    print("\n" + "=" * 50)
    print("Comparison completed")
    print("=" * 50)

    return comparison_results


def generate_resampling_summary(comparison_results):
    """
    Generating summary table from comparison results

    Args:
        comparison_results: Dictionary from compare_resampling_techniques()

    Returns:
        DataFrame containing summary statistics
    """

    summary_data = []

    for technique, result_data in comparison_results.items():
        if result_data['execution_status'] == 'success':
            row = {
                'Technique': technique,
                'Original Size': result_data['original_size'],
                'Resampled Size': result_data['resampled_size'],
                'Size Change': result_data['size_change'],
                'Change %': (result_data['size_change'] / result_data['original_size']) * 100,
                'Class 0': result_data['resampled_distribution'].get(0, 0),
                'Class 1': result_data['resampled_distribution'].get(1, 0),
            }
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    return summary_df


def persist_resampled_data(resampled_features, resampled_labels,
                           technique_name, file_prefix='higgs'):
    """
    Persisting resampled dataset to disk

    Args:
        resampled_features: Resampled feature matrix
        resampled_labels: Resampled target vector
        technique_name: Name of resampling technique
        file_prefix: Prefix for output filename
    """

    print(f"Saving resampled data for {technique_name}...")

    # Creating DataFrame
    if isinstance(resampled_features, pd.DataFrame):
        output_data = resampled_features.copy()
    else:
        output_data = pd.DataFrame(resampled_features, columns=ALL_FEATURES)

    output_data[TARGET_COL] = resampled_labels

    # Defining output path
    filename = f"{file_prefix}_{technique_name}_resampled.csv"
    output_path = RESAMPLED_DIR / filename

    # Writing to CSV
    output_data.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")
    print(f"Observations: {len(output_data):,}")
    print(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")


def retrieve_resampled_data(technique_name, file_prefix='higgs'):
    """
    Retrieving previously saved resampled dataset

    Args:
        technique_name: Name of resampling technique
        file_prefix: Prefix of saved filename

    Returns:
        features, labels: Loaded resampled data
    """

    filename = f"{file_prefix}_{technique_name}_resampled.csv"
    input_path = RESAMPLED_DIR / filename

    print(f"Loading resampled data from {input_path}...")

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find {filename} in {RESAMPLED_DIR}")

    dataset = pd.read_csv(input_path)

    features = dataset[ALL_FEATURES]
    labels = dataset[TARGET_COL]

    print(f"Loaded {len(features):,} observations")

    return features, labels


def calculate_sampling_strategy(original_labels, target_ratio=1.0):
    """
    Calculating custom sampling strategy for specific imbalance ratio

    Args:
        original_labels: Original target vector
        target_ratio: Desired minority/majority ratio

    Returns:
        Dictionary specifying number of samples per class
    """

    label_counts = Counter(original_labels)

    if len(label_counts) != 2:
        raise ValueError("Binary classification required")

    # Identifying minority and majority classes
    sorted_classes = sorted(label_counts.items(), key=lambda x: x[1])
    minority_class, minority_count = sorted_classes[0]
    majority_class, majority_count = sorted_classes[1]

    # Computing target counts
    target_minority = int(majority_count * target_ratio)

    strategy = {
        minority_class: target_minority,
        majority_class: majority_count
    }

    print("Calculated sampling strategy:")
    print(f"  Minority class {minority_class}: {minority_count} -> {target_minority}")
    print(f"  Majority class {majority_class}: {majority_count} (unchanged)")
    print(f"  Target ratio: {target_ratio:.3f}:1")

    return strategy


if __name__ == "__main__":
    print("=" * 70)
    print("RESAMPLING UTILITIES MODULE TEST")
    print("=" * 70)

    # Generating synthetic imbalanced dataset for testing
    from sklearn.datasets import make_classification

    print("\nGenerating synthetic imbalanced dataset...")
    X_synthetic, y_synthetic = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=SEED_VALUE
    )

    print(f"Generated dataset: {X_synthetic.shape}")
    print(f"Original distribution: {dict(Counter(y_synthetic))}")

    print("\nTest 1: Measuring imbalance...")
    min_count, maj_count, ratio = measure_imbalance_ratio(y_synthetic)
    print(f"Minority: {min_count}, Majority: {maj_count}, Ratio: {ratio:.3f}")

    print("\nTest 2: Applying SMOTE...")
    X_smote, y_smote = execute_resampling(X_synthetic, y_synthetic,
                                          technique='smote')

    print("\nTest 3: Comparing multiple techniques...")
    techniques_list = ['smote', 'adasyn', 'random_over']
    technique_results = compare_resampling_techniques(X_synthetic, y_synthetic,
                                                      techniques_list)

    print("\nTest 4: Generating summary...")
    summary = generate_resampling_summary(technique_results)
    print(summary)

    print("\nTest 5: Visualizing SMOTE impact...")
    visualize_resampling_impact(X_synthetic, y_synthetic,
                               X_smote, y_smote,
                               technique_name='SMOTE')

    print("ALL TESTS COMPLETED SUCCESSFULLY")
