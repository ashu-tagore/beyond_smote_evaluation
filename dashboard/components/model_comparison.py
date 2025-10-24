"""
Model Comparison page for Higgs Discovery Dashboard
Displayed model performance metrics and comparisons
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import results_loader
from utils import (
    create_metrics_heatmap,
    create_roc_curve,
    create_confusion_matrix_plot,
    get_model_display_name
)
from config import COLORS


def show_model_comparison_page():
    """
    Displayed model comparison page
    Showed performance metrics across different models
    """
    st.header("Model Comparison - Performance Analysis")

    # Loaded model performance data
    model_metrics = results_loader.load_model_metrics()
    confusion_matrices = results_loader.load_confusion_matrices()
    roc_data = results_loader.load_roc_curves()

    # Created metrics comparison section
    st.subheader("Model Performance Metrics")

    # Converted metrics to dataframe for heatmap
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df.index = [get_model_display_name(idx) for idx in metrics_df.index]

    # Displayed metrics table
    st.dataframe(
        metrics_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True
    )

    st.markdown("---")

    # Created performance heatmap
    st.subheader("Performance Heatmap")

    fig = create_metrics_heatmap(metrics_df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Created model selection for detailed view
    st.subheader("Detailed Model Analysis")

    model_names = list(model_metrics.keys())
    display_names = [get_model_display_name(name) for name in model_names]

    selected_display_name = st.selectbox(
        "Select Model for Detailed Analysis",
        display_names
    )

    # Found original model name
    selected_model = model_names[display_names.index(selected_display_name)]

    # Displayed selected model metrics
    selected_metrics = model_metrics[selected_model]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{selected_metrics['accuracy']:.4f}")

    with col2:
        st.metric("Precision", f"{selected_metrics['precision']:.4f}")

    with col3:
        st.metric("Recall", f"{selected_metrics['recall']:.4f}")

    with col4:
        st.metric("F1-Score", f"{selected_metrics['f1_score']:.4f}")

    with col5:
        st.metric("ROC-AUC", f"{selected_metrics['roc_auc']:.4f}")

    st.markdown("---")

    # Created two columns for visualizations
    col_left, col_right = st.columns(2)

    with col_left:
        # Displayed confusion matrix if available
        st.subheader("Confusion Matrix")

        if selected_model in confusion_matrices:
            cm = confusion_matrices[selected_model]
            fig_cm = create_confusion_matrix_plot(cm)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix not available for this model")

    with col_right:
        # Displayed ROC curve if available
        st.subheader("ROC Curve")

        if selected_model in roc_data:
            roc_info = roc_data[selected_model]
            fig_roc = create_roc_curve(
                roc_info['fpr'],
                roc_info['tpr'],
                roc_info['auc'],
                selected_display_name
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curve not available for this model")

    st.markdown("---")

    # Added model comparison insights
    st.subheader("Model Comparison Insights")

    # Found best model for each metric
    best_accuracy_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
    best_f1_model = max(model_metrics.items(), key=lambda x: x[1]['f1_score'])
    best_auc_model = max(model_metrics.items(), key=lambda x: x[1]['roc_auc'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success(f"""
        **Best Accuracy**

        {get_model_display_name(best_accuracy_model[0])}

        Score: {best_accuracy_model[1]['accuracy']:.4f}
        """)

    with col2:
        st.success(f"""
        **Best F1-Score**

        {get_model_display_name(best_f1_model[0])}

        Score: {best_f1_model[1]['f1_score']:.4f}
        """)

    with col3:
        st.success(f"""
        **Best ROC-AUC**

        {get_model_display_name(best_auc_model[0])}

        Score: {best_auc_model[1]['roc_auc']:.4f}
        """)

    st.markdown("---")

    # Added performance summary
    st.subheader("Performance Summary")

    st.info("""
    **Key Findings from Model Comparison:**

    1. **XGBoost leads overall:** Achieves highest F1-score and ROC-AUC
    2. **Neural Network competitive:** Close second in most metrics
    3. **Random Forest robust:** Good balance of precision and recall
    4. **Logistic Regression baseline:** Simple but effective, fast training
    5. **Ensemble potential:** Combining models could improve performance

    **Recommendation:** Deploy XGBoost for production use, with neural network as backup.
    """)
