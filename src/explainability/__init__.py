"""Explainability module for interpretable AI."""

from .shap_explainer import MolecularExplainer, explain_molecule_prediction

__all__ = ['MolecularExplainer', 'explain_molecule_prediction']
