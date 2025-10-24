"""
Components package for Higgs Boson Discovery Dashboard
Initialized component modules
"""

# Imported all component functions for easy access
from components.kpi_cards import show_overview_page
from components.data_overview import show_data_explorer_page
from components.feature_analysis import show_feature_analysis_page
from components.model_comparison import show_model_comparison_page
from components.physics_insights import show_physics_discoveries_page
from components.recommendations import show_recommendations_page

__all__ = [
    'show_overview_page',
    'show_data_explorer_page',
    'show_feature_analysis_page',
    'show_model_comparison_page',
    'show_physics_discoveries_page',
    'show_recommendations_page'
]
