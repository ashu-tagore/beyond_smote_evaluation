"""
Feature Analysis page for Higgs Discovery Dashboard
Displayed feature importance and physics insights
"""

import streamlit as st
import pandas as pd

from data_loader import results_loader
from utils import create_feature_importance_chart, create_comparison_bar_chart


def show_feature_analysis_page():
    """
    Displayed feature analysis page
    Showed feature importance and correlations
    """
    st.header("Feature Analysis - Physics Insights")

    # Loaded feature importance data
    feature_importance = results_loader.load_feature_importance()

    # Displayed feature importance section
    st.subheader("Feature Importance")

    st.markdown("""
    Identified the most important features for distinguishing Higgs signals from background events.
    Higher importance scores indicate features that contribute more to model predictions.
    """)

    # Created feature importance chart
    fig = create_feature_importance_chart(feature_importance, top_n=10)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Added feature importance table
    st.subheader("Top Features Ranked")

    # Converted to dataframe and sorted
    importance_df = pd.DataFrame(
        list(feature_importance.items()),
        columns=['Feature', 'Importance Score']
    ).sort_values('Importance Score', ascending=False).reset_index(drop=True)

    # Displayed top 15 features
    st.dataframe(
        importance_df.head(15),
        use_container_width=True,
        hide_index=False
    )

    st.markdown("---")

    # Added physics interpretation section
    st.subheader("Physics Interpretation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Most Important Features:**

        The top-ranked features are invariant mass combinations:
        - **m_wwbb:** Mass of W bosons and b-jets system
        - **m_wbb:** Mass of W boson and b-jets
        - **m_bb:** Mass of b-jet pair

        These features directly relate to Higgs decay signatures.
        """)

    with col2:
        st.markdown("""
        **Physics Significance:**

        - Invariant mass features capture the decay kinematics
        - Jet features identify b-quarks from Higgs decay
        - Missing energy indicates neutrinos
        - Combined, these features reconstruct the Higgs event
        """)

    st.markdown("---")

    # Added feature selection recommendations
    st.subheader("Feature Selection Insights")

    st.info("""
    **Key Findings from Feature Analysis:**

    1. **Invariant mass features dominate:** The top 3 features are all mass combinations
    2. **Jet momentum matters:** High-pT jets are strong discriminators
    3. **B-tagging is critical:** Identification of b-quarks improves signal purity
    4. **Angular features help:** Phi and eta distributions differ between classes
    5. **Missing energy essential:** Neutrinos from W decay leave energy signature

    **Recommendation:** Focus on invariant mass reconstruction and jet identification
    for optimal Higgs signal extraction.
    """)

    st.markdown("---")

    # Added feature correlation insights
    st.subheader("Feature Correlations")

    st.markdown("""
    **Correlation Patterns Observed:**

    - **Strong correlations:** Invariant mass features show expected physical correlations
    - **Jet features:** Leading jets have moderate correlations with each other
    - **Independent features:** Lepton and missing energy features relatively independent
    - **Engineered features:** Mass combinations effectively capture decay topology

    Feature engineering successfully created discriminative variables while maintaining
    physical interpretability.
    """)

    st.markdown("---")

    # Added comparison with physics theory
    st.subheader("Alignment with Physics Theory")

    st.success("""
    **Feature Importance Matches Theoretical Expectations:**

    The data-driven feature importance rankings align well with physics theory:

    1. Higgs boson decays to WW and bb final states
    2. Invariant mass reconstruction identifies these decay channels
    3. B-jet tagging confirms b-quark presence
    4. Missing energy indicates neutrino production

    This agreement validates both the analysis approach and the theoretical understanding
    of Higgs production and decay mechanisms.
    """)
