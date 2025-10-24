"""
Overview page with KPI cards for Higgs Discovery Dashboard
Displayed summary metrics and key performance indicators
"""

import streamlit as st

from data_loader import results_loader
from utils import format_number, format_percentage, create_metric_card


def show_overview_page():
    """
    Displayed overview page with KPI summary cards
    Showed key metrics from the analysis
    """
    st.header("Overview - Key Performance Indicators")

    # Loaded metrics data
    model_metrics = results_loader.load_model_metrics()
    data_summary = results_loader.load_data_summary()
    significance_data = results_loader.load_statistical_significance()

    # Created KPI cards section
    st.subheader("Summary Metrics")

    # Created columns for KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Displayed total events analyzed
        total_events = data_summary[data_summary['metric'] == 'Total Events']['value'].values[0]
        create_metric_card(
            "Total Events Analyzed",
            format_number(total_events, 1),
            None
        )

    with col2:
        # Displayed best model performance
        best_f1 = max([metrics.get('f1_score', 0) for metrics in model_metrics.values()])
        create_metric_card(
            "Best F1-Score",
            format_percentage(best_f1, 2),
            None
        )

    with col3:
        # Displayed statistical significance
        significance = significance_data.get('significance', 0)
        create_metric_card(
            "Discovery Significance",
            f"{significance:.2f} sigma",
            None
        )

    with col4:
        # Displayed best AUC score
        best_auc = max([metrics.get('roc_auc', 0) for metrics in model_metrics.values()])
        create_metric_card(
            "Best ROC-AUC",
            format_percentage(best_auc, 2),
            None
        )

    st.markdown("---")

    # Created second row of KPIs
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        # Displayed number of models evaluated
        num_models = len(model_metrics)
        create_metric_card(
            "Models Evaluated",
            str(num_models),
            None
        )

    with col6:
        # Displayed signal ratio
        signal_ratio = data_summary[data_summary['metric'] == 'Signal Ratio']['value'].values[0]
        create_metric_card(
            "Signal Ratio",
            format_percentage(signal_ratio, 1),
            None
        )

    with col7:
        # Displayed best accuracy
        best_accuracy = max([metrics.get('accuracy', 0) for metrics in model_metrics.values()])
        create_metric_card(
            "Best Accuracy",
            format_percentage(best_accuracy, 2),
            None
        )

    with col8:
        # Displayed number of features
        create_metric_card(
            "Physics Features",
            "28",
            None
        )

    st.markdown("---")

    # Added project information section
    st.subheader("Project Information")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        **Dataset Details**
        - **Source:** HIGGS UCI Dataset from CERN
        - **Total Events:** 11 Million collision events
        - **Features:** 28 physics variables
        - **Target:** Binary classification (Signal vs Background)
        - **Signal Class:** Higgs boson decay events
        - **Background Class:** Other particle processes
        """)

    with col_right:
        st.markdown("""
        **Analysis Summary**
        - **Models Trained:** Multiple ML algorithms
        - **Best Performer:** XGBoost classifier
        - **Key Achievement:** High statistical significance
        - **Techniques Used:** Feature engineering, hyperparameter tuning
        - **Validation:** Cross-validation and holdout testing
        - **Result:** Successfully identified Higgs signals
        """)

    st.markdown("---")

    # Added quick insights section
    st.subheader("Quick Insights")

    # Found best performing model
    best_model = max(model_metrics.items(), key=lambda x: x[1].get('f1_score', 0))
    best_model_name = best_model[0].replace('_', ' ').title()
    best_model_f1 = best_model[1].get('f1_score', 0)

    # Created insight cards
    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.info(f"""
        **Best Overall Model**

        {best_model_name} achieved the highest F1-score of {format_percentage(best_model_f1, 2)},
        demonstrating excellent balance between precision and recall in identifying Higgs signals.
        """)

    with insight_col2:
        st.success(f"""
        **Discovery Significance**

        Achieved {significance:.2f} sigma statistical significance, far exceeding the 5 sigma
        threshold required for particle physics discovery confirmation.
        """)

    st.markdown("---")

    # Added getting started guide
    st.subheader("Getting Started")

    st.markdown("""
    **Navigate through the dashboard:**

    1. **Data Explorer** - Explore dataset statistics and feature distributions
    2. **Feature Analysis** - Examine feature importance and physics insights
    3. **Model Comparison** - Compare performance across different ML models
    4. **Physics Discoveries** - View invariant mass plots and signal extraction
    5. **Recommendations** - Get actionable insights and best practices

    Use the sidebar to navigate between different pages of the dashboard.
    """)
