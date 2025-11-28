"""
AI Drug Discovery Platform - Clean Working UI
Simple, beautiful, and functional.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import featurize_smiles
from models_fixed import SimpleTabularClassifier
from prediction_database import PredictionDatabase
from name_to_smiles import name_to_smiles
from molecular_visualization import smiles_to_image
from data_fusion import DataFusion, analyze_csv_type

# Page config
st.set_page_config(
    page_title="AI Drug Discovery Pro",
    page_icon="💊",
    layout="wide"
)

# Simple working CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a2e;
    }
    
    h1, h2, h3 {
        color: #eee;
    }
    
    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #5f3dc4;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'db' not in st.session_state:
    st.session_state.db = PredictionDatabase()

@st.cache_resource
def load_model():
    model_path = Path("results/model.pt")
    if model_path.exists():
        model = SimpleTabularClassifier(input_dim=13, hidden_dim=64, output_dim=1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        model.eval()
        return model
    return None

if st.session_state.model is None:
    st.session_state.model = load_model()

# Header
st.title("💊 AI Drug Discovery Platform")
st.markdown("### Predict molecular activity with AI")

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Predict", "🔀 Data Fusion", "📊 Analytics"])

with tab1:
    st.subheader("Molecular Activity Predictor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mol_input = st.text_input("Enter SMILES or chemical name:", placeholder="e.g., aspirin, CCO, caffeine")
        
        if st.button("🔬 Predict", type="primary"):
            if mol_input and st.session_state.model:
                smiles = name_to_smiles(mol_input)
                
                if smiles != mol_input:
                    st.info(f"Converted '{mol_input}' to SMILES: `{smiles}`")
                
                with st.spinner("Analyzing..."):
                    features_dict = featurize_smiles(smiles)
                    if features_dict:
                        features = np.array(list(features_dict.values()), dtype=np.float32)
                        
                        with torch.no_grad():
                            x = torch.tensor(features).unsqueeze(0)
                            logit = st.session_state.model(x)
                            prob = torch.sigmoid(logit).item()
                        
                        prediction = "Active" if prob > 0.5 else "Inactive"
                        
                        # Save to database
                        st.session_state.db.add_prediction(
                            smiles=smiles,
                            prediction=prediction,
                            probability=prob,
                            confidence=max(prob, 1-prob),
                            features=features_dict
                        )
                        
                        # Show results
                        with col2:
                            st.markdown("### Results")
                            
                            if prediction == "Active":
                                st.success("🟢 ACTIVE")
                            else:
                                st.error("🔴 INACTIVE")
                            
                            st.metric("Probability", f"{prob*100:.1f}%")
                            st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")
                            
                            # Show structure
                            img_data = smiles_to_image(smiles)
                            if img_data:
                                st.image(img_data, caption="2D Structure", use_column_width=True)
                    else:
                        st.error("Invalid molecule")
    
    with col2:
        if not mol_input:
            st.info("👈 Enter a molecule to see predictions")

with tab2:
    st.subheader("Multi-File Data Fusion")
    
    uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        fusion = DataFusion()
        for file in uploaded_files:
            df = pd.read_csv(file)
            fusion.add_file(file.name, df)
        
        if st.button("🚀 Merge Files"):
            merged, info = fusion.smart_merge()
            st.success("✅ Files merged successfully!")
            st.dataframe(merged.head(20))
            
            # Download button
            csv = merged.to_csv(index=False)
            st.download_button(
                "📥 Download Merged Data",
                csv,
                f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

with tab3:
    st.subheader("Analytics Dashboard")
    
    stats = st.session_state.db.get_prediction_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", stats['total_predictions'])
    col2.metric("Active", stats['active_count'])
    col3.metric("Inactive", stats['inactive_count'])
    col4.metric("Avg Confidence", f"{stats['average_confidence']*100:.0f}%")
    
    # Charts
    recent = st.session_state.db.get_recent_predictions(100)
    if recent:
        df = pd.DataFrame(recent)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='prediction', title="Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='confidence', title="Confidence Levels")
            st.plotly_chart(fig, use_container_width=True)

# Sidebar
with st.sidebar:
    st.title("📊 Stats")
    
    if st.session_state.model:
        st.success("✅ Model Ready")
    else:
        st.error("❌ Model Not Found")
    
    stats = st.session_state.db.get_prediction_stats()
    st.metric("Predictions", stats['total_predictions'])
    st.metric("Active", stats['active_count'])

st.markdown("---")
st.caption("🚀 AI Drug Discovery Platform | Powered by Advanced ML")
