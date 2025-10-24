"""
Main Streamlit application for Higgs Boson Discovery Dashboard
Provided navigation and page routing for the dashboard
"""

import sys
from pathlib import Path

import streamlit as st

# Added parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Imported component pages
from components.kpi_cards import show_overview_page
from components.data_overview import show_data_explorer_page
from components.feature_analysis import show_feature_analysis_page
from components.model_comparison import show_model_comparison_page
from components.physics_insights import show_physics_discoveries_page
from components.recommendations import show_recommendations_page

# Imported configuration and utilities
from config import DASHBOARD_TITLE, PAGE_ICON, LAYOUT, PAGES
from utils import apply_custom_css


def main():
    """
    Main application function
    Set up page configuration and navigation
    """
    # Configured page settings
    st.set_page_config(
        page_title=DASHBOARD_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )

    # Applied custom styling
    apply_custom_css()

    # Created sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        st.markdown("---")

        # Created page selection radio buttons
        selected_page = st.radio(
            "Select Page",
            PAGES,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Added project information in sidebar
        st.subheader("Project Info")
        st.markdown("""
        **Higgs Boson Discovery**

        Analysis of particle collision data from CERN to identify Higgs boson signals.

        **Dataset:** HIGGS UCI Dataset
        **Events:** 11 Million
        **Features:** 28 Physics Variables
        """)

        st.markdown("---")

        # Added navigation help
        st.subheader("Quick Guide")
        st.markdown("""
        1. **Overview** - Summary metrics and KPIs
        2. **Data Explorer** - Dataset statistics
        3. **Feature Analysis** - Feature importance
        4. **Model Comparison** - Performance metrics
        5. **Physics Discoveries** - Scientific insights
        6. **Recommendations** - Best practices
        """)

    # Displayed main title
    st.title(DASHBOARD_TITLE)
    st.markdown("---")

    # Routed to selected page
    if selected_page == "Overview":
        show_overview_page()
    elif selected_page == "Data Explorer":
        show_data_explorer_page()
    elif selected_page == "Feature Analysis":
        show_feature_analysis_page()
    elif selected_page == "Model Comparison":
        show_model_comparison_page()
    elif selected_page == "Physics Discoveries":
        show_physics_discoveries_page()
    elif selected_page == "Recommendations":
        show_recommendations_page()

    # Added footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>Higgs Boson Discovery Analysis Dashboard | Data Science Final Project</p>
        <p>Built with Streamlit | Analysis using pandas, scikit-learn, and XGBoost</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
