"""
Utility functions for Higgs Discovery Dashboard
Provided helper functions for formatting, plotting, and calculations
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from config import COLORS, METRIC_DISPLAY_NAMES, MODEL_NAMES

def format_number(value, decimal_places=2):
    """
    Formatted number for display

    Parameters:
        value: Number to format
        decimal_places: Number of decimal places

    Returns:
        Formatted string
    """
    if isinstance(value, (int, float)):
        if value >= 1_000_000:
            return f"{value/1_000_000:.{decimal_places}f}M"
        elif value >= 1_000:
            return f"{value/1_000:.{decimal_places}f}K"
        else:
            return f"{value:.{decimal_places}f}"
    return str(value)

def format_percentage(value, decimal_places=2):
    """
    Formatted value as percentage

    Parameters:
        value: Value to format (0-1 range)
        decimal_places: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """
    Created styled metric card

    Parameters:
        title: Metric name
        value: Metric value
        delta: Change value (optional)
        delta_color: Color for delta (normal, inverse, off)
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color
    )

def create_comparison_bar_chart(data_dict, title, xlabel, ylabel):
    """
    Created horizontal bar chart for comparing values

    Parameters:
        data_dict: Dictionary with labels as keys and values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Plotly figure object
    """
    df = pd.DataFrame({
        ylabel: list(data_dict.keys()),
        xlabel: list(data_dict.values())
    })

    fig = px.bar(
        df,
        x=xlabel,
        y=ylabel,
        orientation='h',
        title=title,
        color=xlabel,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

def create_confusion_matrix_plot(cm, labels=None):
    """
    Created confusion matrix heatmap

    Parameters:
        cm: Confusion matrix (2D array)
        labels: Class labels

    Returns:
        Plotly figure object
    """
    if labels is None:
        labels = ['Background', 'Signal']

    # Converted to numpy array if needed
    cm_array = np.array(cm)

    # Created heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm_array,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

def create_roc_curve(fpr, tpr, auc_score, model_name):
    """
    Created ROC curve plot

    Parameters:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        model_name: Name of the model

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Added ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.3f})',
        line=dict(color=COLORS['primary'], width=2)
    ))

    # Added diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))

    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )

    return fig

def create_feature_importance_chart(importance_dict, top_n=10):
    """
    Created feature importance bar chart

    Parameters:
        importance_dict: Dictionary with features and importance scores
        top_n: Number of top features to display

    Returns:
        Plotly figure object
    """
    # Sorted by importance and selected top N
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    features, scores = zip(*sorted_features)

    fig = go.Figure(go.Bar(
        x=list(scores),
        y=list(features),
        orientation='h',
        marker=dict(color=list(scores), colorscale='Viridis')
    ))

    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    return fig

def create_distribution_plot(data, feature_name, signal_data=None, background_data=None):
    """
    Created distribution plot comparing signal and background

    Parameters:
        data: Full dataset (if signal and background not separate)
        feature_name: Name of feature to plot
        signal_data: Signal class data (optional)
        background_data: Background class data (optional)

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    if signal_data is not None and background_data is not None:
        # Plotted separate distributions
        fig.add_trace(go.Histogram(
            x=signal_data,
            name='Signal',
            opacity=0.7,
            marker_color=COLORS['signal'],
            nbinsx=50
        ))

        fig.add_trace(go.Histogram(
            x=background_data,
            name='Background',
            opacity=0.7,
            marker_color=COLORS['background'],
            nbinsx=50
        ))
    else:
        # Plotted single distribution
        fig.add_trace(go.Histogram(
            x=data,
            name=feature_name,
            marker_color=COLORS['primary'],
            nbinsx=50
        ))

    fig.update_layout(
        title=f'Distribution of {feature_name}',
        xaxis_title=feature_name,
        yaxis_title='Count',
        barmode='overlay',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

def create_metrics_heatmap(metrics_df):
    """
    Created heatmap of model metrics

    Parameters:
        metrics_df: DataFrame with models as rows, metrics as columns

    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=metrics_df.values,
        x=metrics_df.columns,
        y=metrics_df.index,
        colorscale='RdYlGn',
        text=metrics_df.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 10},
        showscale=True
    ))

    fig.update_layout(
        title='Model Performance Heatmap',
        xaxis_title='Metrics',
        yaxis_title='Models',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

def calculate_statistical_significance(signal_count, background_count):
    """
    Calculated statistical significance of discovery

    Parameters:
        signal_count: Number of signal events
        background_count: Number of background events

    Returns:
        Significance in standard deviations (sigma)
    """
    if background_count == 0:
        return 0

    # Used simplified significance formula
    # s / sqrt(b) where s=signal, b=background
    significance = signal_count / np.sqrt(background_count)

    return significance

def get_model_display_name(model_key):
    """
    Retrieved display name for model

    Parameters:
        model_key: Internal model key

    Returns:
        Formatted display name
    """
    return MODEL_NAMES.get(model_key, model_key.replace('_', ' ').title())

def get_metric_display_name(metric_key):
    """
    Retrieved display name for metric

    Parameters:
        metric_key: Internal metric key

    Returns:
        Formatted display name
    """
    return METRIC_DISPLAY_NAMES.get(metric_key, metric_key.replace('_', ' ').title())

def apply_custom_css():
    """
    Applied custom CSS styling to dashboard
    """
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        h1 {
            color: #1A535C;
            padding-bottom: 1rem;
        }
        h2 {
            color: #4ECDC4;
            padding-top: 1rem;
        }
        h3 {
            color: #FF6B6B;
        }
        </style>
        """, unsafe_allow_html=True)
