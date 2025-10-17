"""
Resampling utilities for handling class imbalance
Purpose: Implementing various resampling techniques (SMOTE, ROS, RUS, etc.)
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

# Resampling libraries
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
from imblearn.combine import SMOTETomek, SMOTEENN

# Adding project modules to path
sys.path.append(str(Path(__file__).parent))

# Local imports
from config import (
    SEED_VALUE,
    ALL_FEATURES,
    TARGET_COL,
    RESAMPLED_DIR,
    DPI_VALUE
)


def initialize_resampling_methods():
    """
    Initializing dictionary containing all resampling method instances
    
    Returns:
        Dictionary mapping method names to configured resampling objects
    """
    
    resampling_techniques = {
        # Baseline - no resampling
        'baseline': 'no_resampling',
        
        # Oversampling methods
        'random_over': RandomOverSampler(
            sampling_strategy='auto',
            random_state=SEED_VALUE
        ),
        
        'smote': SMOTE(
            sampling_strategy='auto',
            k_neighbors=5,
            random_state=SEED_VALUE
        ),
        
        'borderline_smote': BorderlineSMOTE(
            sampling_strategy='auto',
            k_neighbors=5,
            random_state=SEED_VALUE
        ),
        
        'adasyn': ADASYN(
            sampling_strategy='auto',
            n_neighbors=5,
            random_state=SEED_VALUE
        ),
        
        # Undersampling methods
        'random_under': RandomUnderSampler(
            sampling_strategy='auto',
            random_state=SEED_VALUE
        ),
        
        'tomek': TomekLinks(
            sampling_strategy='auto'
        ),
        
        'nearmiss': NearMiss(
            sampling_strategy='auto',
            version=1,
            n_neighbors=3
        ),
        
        # Combination methods
        'smote_tomek': SMOTETomek(
            sampling_strategy='auto',
            random_state=SEED_VALUE
        ),
        
        'smote_enn': SMOTEENN(
            sampling_strategy='auto',
            random_state=SEED_VALUE
        ),
        
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
    
    # Creating directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    # Calculating target minority count
    target_minority = int(majority_count * target_ratio)
    
    sampling_strategy = {
        minority_class: target_minority,
        majority_class: majority_count
    }
    
    return sampling_strategy


# ============================================================================
# NOTEBOOK 03 COMPATIBILITY FUNCTIONS
# Purpose: Wrapper functions and missing functionality for Notebook 03
# ============================================================================


def apply_resampling_technique(X, y, method_name, resampler_dict=None):
    """
    Wrapper function for execute_resampling with notebook-compatible interface
    Includes automatic name mapping for notebook compatibility
    
    Args:
        X: Feature matrix (DataFrame or array)
        y: Target vector (Series or array)
        method_name: Name of resampling method (notebook naming convention)
        resampler_dict: Dictionary of resampling methods (optional, not used)
    
    Returns:
        X_resampled: Resampled feature matrix
        y_resampled: Resampled target vector
    """
    # Name mapping dictionary: notebook names -> actual function names
    name_mapping = {
        'random_oversampling': 'random_over',
        'smote': 'smote',
        'borderline_smote': 'borderline_smote',
        'adasyn': 'adasyn',
        'random_undersampling': 'random_under',
        'tomek_links': 'tomek',
        'nearmiss': 'nearmiss',
        'smote_tomek': 'smote_tomek',
        'smote_enn': 'smote_enn',
        'baseline': 'baseline',
        'class_weighting': 'class_weight',
        'class_weight': 'class_weight'
    }
    
    # Translate method name if needed
    actual_method_name = name_mapping.get(method_name, method_name)
    
    # Call the existing execute_resampling function
    return execute_resampling(
        features=X,
        labels=y,
        technique=actual_method_name,
        display_info=True
    )


def save_resampled_dataset(X_resampled, y_resampled, method_name, output_dir):
    """
    Wrapper function for persist_resampled_data with notebook-compatible interface
    
    Args:
        X_resampled: Resampled feature matrix
        y_resampled: Resampled target vector
        method_name: Name of resampling method
        output_dir: Output directory (handled automatically)
    
    Returns:
        None (saves to disk)
    """
    persist_resampled_data(
        resampled_features=X_resampled,
        resampled_labels=y_resampled,
        technique_name=method_name,
        file_prefix='higgs'
    )


def compute_resampling_statistics(X_original, y_original, X_resampled, 
                                  y_resampled, method_name, resampling_time):
    """
    Computing comprehensive statistics for resampling method comparison
    
    Args:
        X_original: Original feature matrix before resampling
        y_original: Original target vector before resampling
        X_resampled: Resampled feature matrix
        y_resampled: Resampled target vector
        method_name: Name of the resampling method applied
        resampling_time: Time taken to perform resampling (seconds)
    
    Returns:
        dict: Dictionary containing comprehensive resampling statistics
    """
    from collections import Counter
    
    # Computing class distributions
    original_dist = Counter(y_original)
    resampled_dist = Counter(y_resampled)
    
    # Extracting sample counts
    n_original = len(y_original)
    n_resampled = len(y_resampled)
    
    # Getting class counts from original data
    if len(original_dist) >= 2:
        original_class_counts = sorted(original_dist.values())
        minority_original = original_class_counts[0]
        majority_original = original_class_counts[-1]
    else:
        minority_original = list(original_dist.values())[0]
        majority_original = minority_original
    
    # Getting class counts from resampled data
    if len(resampled_dist) >= 2:
        resampled_class_counts = sorted(resampled_dist.values())
        minority_resampled = resampled_class_counts[0]
        majority_resampled = resampled_class_counts[-1]
    else:
        minority_resampled = list(resampled_dist.values())[0]
        majority_resampled = minority_resampled
    
    # Computing imbalance ratios (avoiding division by zero)
    original_ratio = minority_original / majority_original if majority_original > 0 else 1.0
    resampled_ratio = minority_resampled / majority_resampled if majority_resampled > 0 else 1.0
    
    # Computing size changes
    size_change = n_resampled - n_original
    size_change_pct = (size_change / n_original) * 100 if n_original > 0 else 0.0
    
    # Computing balance improvement
    balance_improvement = resampled_ratio - original_ratio
    
    # Creating comprehensive statistics dictionary
    stats = {
        'method': method_name,
        'n_samples': int(n_resampled),
        'original_samples': int(n_original),
        'size_change': int(size_change),
        'size_change_pct': float(size_change_pct),
        'n_minority': int(minority_resampled),
        'n_majority': int(majority_resampled),
        'imbalance_ratio': float(resampled_ratio),
        'original_ratio': float(original_ratio),
        'balance_improvement': float(balance_improvement),
        'resampling_time': float(resampling_time),
        'minority_increase': int(minority_resampled - minority_original),
        'majority_change': int(majority_resampled - majority_original),
        'perfectly_balanced': bool(abs(resampled_ratio - 1.0) < 0.01)
    }
    
    return stats