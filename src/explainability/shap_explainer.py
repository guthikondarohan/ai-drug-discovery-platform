"""
Explainable AI (XAI) Module using SHAP

Provides explanations for model predictions showing which features
contribute most to predictions.
"""

import shap
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import io
import base64


class MolecularExplainer:
    """
    Generate explanations for molecular activity predictions using SHAP.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: Trained PyTorch model
            feature_names: List of feature names
        """
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.explainer = None
    
    def _predict_wrapper(self, X):
        """Wrapper for model prediction compatible with SHAP."""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)
            return probs.numpy()
    
    def create_explainer(self, background_data: np.ndarray):
        """
        Create SHAP explainer with background data.
        
        Args:
            background_data: Background dataset for SHAP (e.g., 100 samples)
        """
        self.explainer = shap.KernelExplainer(
            self._predict_wrapper,
            background_data
        )
    
    def explain_prediction(
        self,
        features: np.ndarray,
        nsamples: int = 100
    ) -> shap.Explanation:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            features: Feature vector for molecule
            nsamples: Number of samples for SHAP algorithm
        
        Returns:
            SHAP Explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")
        
        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(features, nsamples=nsamples)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0] if isinstance(shap_values, list) else shap_values,
            base_values=self.explainer.expected_value,
            data=features[0],
            feature_names=self.feature_names
        )
        
        return explanation
    
    def get_feature_importance(self, explanation: shap.Explanation) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            explanation: SHAP explanation
        
        Returns:
            DataFrame with features ranked by absolute SHAP value
        """
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP Value': explanation.values,
            'Actual Value': explanation.data,
            'Abs SHAP': np.abs(explanation.values)
        })
        
        # Sort by absolute SHAP value
        importance_df = importance_df.sort_values('Abs SHAP', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df
    
    def plot_waterfall(self, explanation: shap.Explanation, max_display: int = 15) -> str:
        """
        Create waterfall plot showing feature contributions.
        
        Args:
            explanation: SHAP explanation
            max_display: Maximum features to display
        
        Returns:
            Base64 encoded PNG image
        """
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    
    def plot_force(self, explanation: shap.Explanation) -> str:
        """
        Create force plot showing prediction explanation.
        
        Args:
            explanation: SHAP explanation
        
        Returns:
            HTML string with interactive plot
        """
        force_plot = shap.plots.force(
            explanation.base_values,
            explanation.values,
            explanation.data,
            feature_names=self.feature_names,
            matplotlib=False
        )
        
        return shap.getjs() + force_plot.html()
    
    def create_feature_importance_chart(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10
    ) -> go.Figure:
        """
        Create interactive feature importance bar chart.
        
        Args:
            importance_df: Feature importance DataFrame
            top_n: Number of top features to show
        
        Returns:
            Plotly figure
        """
        df_top = importance_df.head(top_n)
        
        # Color positive/negative contributions differently
        colors = ['#10a37f' if v > 0 else '#ef4444' for v in df_top['SHAP Value']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=df_top['SHAP Value'],
                y=df_top['Feature'],
                orientation='h',
                marker_color=colors,
                text=df_top['SHAP Value'].round(3),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Feature Contributions (SHAP Values)",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Feature",
            yaxis={'categoryorder': 'total ascending'},
            height=400 + top_n * 20,
            showlegend=False
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def generate_explanation_text(
        self,
        importance_df: pd.DataFrame,
        prediction: float,
        top_n: int = 5
    ) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            importance_df: Feature importance DataFrame
            prediction: Model prediction (probability)
            top_n: Number of top features to mention
        
        Returns:
            Explanation text
        """
        pred_label = "ACTIVE" if prediction > 0.5 else "INACTIVE"
        confidence = max(prediction, 1 - prediction) * 100
        
        explanation = f"**Prediction: {pred_label}** (Confidence: {confidence:.1f}%)\n\n"
        explanation += "**Key Contributing Factors:**\n\n"
        
        df_top = importance_df.head(top_n)
        
        for idx, row in df_top.iterrows():
            feature = row['Feature']
            shap_value = row['SHAP Value']
            actual_value = row['Actual Value']
            
            impact = "increases" if shap_value > 0 else "decreases"
            
            explanation += f"{idx + 1}. **{feature}** = {actual_value:.3f}  \n"
            explanation += f"   → This feature {impact} the likelihood of activity "
            explanation += f"(SHAP: {shap_value:+.3f})\n\n"
        
        return explanation


def explain_molecule_prediction(
    model,
    features: np.ndarray,
    feature_names: List[str],
    background_data: np.ndarray,
    prediction: float
) -> Dict:
    """
    Convenience function to generate full explanation for a molecule.
    
    Args:
        model: Trained model
        features: Feature vector
        feature_names: List of feature names
        background_data: Background dataset for SHAP
        prediction: Model prediction
    
    Returns:
        Dictionary with explanation results
    """
    explainer = MolecularExplainer(model, feature_names)
    explainer.create_explainer(background_data)
    
    explanation = explainer.explain_prediction(features)
    importance_df = explainer.get_feature_importance(explanation)
    
    results = {
        'importance_df': importance_df,
        'importance_chart': explainer.create_feature_importance_chart(importance_df),
        'waterfall_plot': explainer.plot_waterfall(explanation),
        'explanation_text': explainer.generate_explanation_text(
            importance_df,
            prediction
        )
    }
    
    return results
