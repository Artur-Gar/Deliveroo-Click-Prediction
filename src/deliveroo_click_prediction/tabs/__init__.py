"""Tab rendering modules."""

from deliveroo_click_prediction.tabs.batch_predictions import render_batch_predictions_tab
from deliveroo_click_prediction.tabs.business_insights import render_business_insights_tab
from deliveroo_click_prediction.tabs.model_metrics import render_model_metrics_tab
from deliveroo_click_prediction.tabs.single_prediction import render_single_prediction_tab

__all__ = [
    "render_batch_predictions_tab",
    "render_business_insights_tab",
    "render_model_metrics_tab",
    "render_single_prediction_tab",
]
