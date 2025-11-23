"""
Streamlit Web Application for AI Drug Discovery
Real-time molecule activity prediction with interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import featurize_smiles
from models_fixed import SimpleTabularClassifier
from utils import set_seed

# Page configuration
st.set_page_config(
    page_title="AI Drug Discovery Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    h1 {
        background: linear-gradient(120deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = Path("results/model.pt")
    if not model_path.exists():
        return None, 0
    
    # Detect input dimension from processed training data
    train_path = Path("data/processed/molecules_train.csv")
    if train_path.exists():
        df = pd.read_csv(train_path)
        feature_cols = [c for c in df.columns if c not in ['smiles', 'label', 'id']]
        input_dim = len(feature_cols)
    else:
        input_dim = 3  # Fallback dimension
    
    try:
        model = SimpleTabularClassifier(input_dim=input_dim, hidden_dim=64, output_dim=1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, input_dim
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, 0

def predict_molecule(smiles, model, expected_features=None):
    """Predict activity for a single molecule"""
    try:
        # Featurize
        features_dict = featurize_smiles(smiles)
        if not features_dict:
            st.error("Invalid SMILES notation")
            return None, None
        
        # Convert to array
        features = np.array(list(features_dict.values()), dtype=np.float32)
        
        # Predict
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logit = model(x)
            prob = torch.sigmoid(logit).item()
        
        return prob, features_dict
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    # Header
    st.title("💊 AI Drug Discovery Platform")
    st.markdown("### Real-time Molecular Activity Prediction")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/treatment.png", width=80)
        st.markdown("## 🔬 Model Info")
        
        model, input_dim = load_model()
        if model:
            st.success("✅ Model Loaded")
            feature_type = "RDKit Descriptors" if input_dim > 3 else "Basic Features"
            st.info(f"**Architecture:** 3-Layer Neural Network\n\n**Input Features:** {input_dim} ({feature_type})\n\n**Status:** Ready for Predictions")
        else:
            st.error("❌ No Model Found")
            st.warning("Train a model first:\n```\npython -m src.main --stage all\n```")
        
        st.markdown("---")
        st.markdown("## 📊 Quick Stats")
        
        # Load training data if available
        train_path = Path("data/processed/molecules_train.csv")
        if train_path.exists():
            df = pd.read_csv(train_path)
            st.metric("Training Samples", len(df))
            st.metric("Active Molecules", f"{(df['label'].sum() / len(df) * 100):.1f}%")
        
        st.markdown("---")
        st.markdown("### 🧪 Sample SMILES")
        st.code("CCO", language="text")
        st.caption("Ethanol")
        st.code("CC(C)Cc1ccc(cc1)C(C)C(O)=O", language="text")
        st.caption("Ibuprofen")
    
    # Main content
    if model is None:
        st.warning("⚠️ Please train a model first before making predictions.")
        st.code(r".\venv\Scripts\python -m src.main --stage all --epochs 10", language="bash")
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Batch Analysis", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.markdown("## Single Molecule Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            smiles_input = st.text_input(
                "Enter SMILES notation:",
                value="CCO",
                help="Simplified Molecular Input Line Entry System notation"
            )
            
            predict_button = st.button("🚀 Predict Activity", type="primary", use_container_width=True)
        
        if predict_button and smiles_input:
            with st.spinner("Analyzing molecule..."):
                time.sleep(0.5)  # Dramatic effect
                prob, features = predict_molecule(smiles_input, model)
            
            if prob is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction result
                    activity_class = "ACTIVE" if prob > 0.5 else "INACTIVE"
                    confidence = prob if prob > 0.5 else (1 - prob)
                    
                    if activity_class == "ACTIVE":
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>🎯 {activity_class}</h2>
                            <h1>{prob*100:.1f}%</h1>
                            <p>Activity Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-card">
                            <h2>⚪ {activity_class}</h2>
                            <h1>{prob*100:.1f}%</h1>
                            <p>Activity Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence meter
                    st.markdown("### Confidence Score")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        title={'text': "Model Confidence"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "lightblue"},
                                {'range': [75, 100], 'color': "royalblue"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Feature visualization
                    st.markdown("### Molecular Features")
                    
                    feature_df = pd.DataFrame({
                        'Feature': list(features.keys()),
                        'Value': list(features.values())
                    })
                    
                    # Show top features in chart
                    top_features = feature_df.nlargest(8, 'Value') if len(feature_df) > 8 else feature_df
                    
                    fig = px.bar(
                        top_features,
                        x='Feature',
                        y='Value',
                        color='Value',
                        color_continuous_scale='viridis',
                        title=f'Top Molecular Features ({len(features)} total)'
                    )
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature table
                    st.dataframe(feature_df, use_container_width=True, hide_index=True)
                    
                    # SMILES info
                    st.info(f"**Input SMILES:** `{smiles_input}`")
    
    with tab2:
        st.markdown("## Batch Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with SMILES column",
            type=['csv'],
            help="CSV must contain a 'smiles' column"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            if 'smiles' not in df.columns:
                st.error("CSV must contain a 'smiles' column")
            else:
                st.success(f"✅ Loaded {len(df)} molecules")
                
                if st.button("🔬 Analyze All Molecules", type="primary"):
                    progress_bar = st.progress(0)
                    predictions = []
                    
                    for i, smiles in enumerate(df['smiles']):
                        prob, _ = predict_molecule(smiles, model)
                        predictions.append(prob if prob is not None else np.nan)
                        progress_bar.progress((i + 1) / len(df))
                    
                    df['activity_probability'] = predictions
                    df['prediction'] = ['Active' if p > 0.5 else 'Inactive' for p in predictions]
                    
                    # Results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Molecules", len(df))
                    with col2:
                        active_count = (df['prediction'] == 'Active').sum()
                        st.metric("Active Predictions", active_count)
                    with col3:
                        st.metric("Active %", f"{active_count/len(df)*100:.1f}%")
                    
                    # Distribution plot
                    fig = px.histogram(
                        df,
                        x='activity_probability',
                        color='prediction',
                        title='Activity Probability Distribution',
                        nbins=30,
                        color_discrete_map={'Active': '#f5576c', 'Inactive': '#4facfe'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.markdown("### Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    with tab3:
        st.markdown("## Model Performance")
        
        # Try to load training history
        train_path = Path("data/processed/molecules_train.csv")
        test_path = Path("data/processed/molecules_test.csv")
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", len(train_df))
            with col2:
                st.metric("Test Samples", len(test_df))
            with col3:
                st.metric("Total Features", 3)
            
            # Class distribution
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Training',
                x=['Inactive', 'Active'],
                y=[len(train_df) - train_df['label'].sum(), train_df['label'].sum()],
                marker_color='#667eea'
            ))
            fig.add_trace(go.Bar(
                name='Test',
                x=['Inactive', 'Active'],
                y=[len(test_df) - test_df['label'].sum(), test_df['label'].sum()],
                marker_color='#764ba2'
            ))
            fig.update_layout(title='Class Distribution', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature statistics
            st.markdown("### Feature Statistics")
            feature_cols = [col for col in train_df.columns if col not in ['smiles', 'label', 'id']]
            if feature_cols:
                stats_df = train_df[feature_cols].describe()
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("📊 No training data found. Run preprocessing first.")
    
    with tab4:
        st.markdown("## About This Platform")
        
        st.markdown("""
        ### 🎯 Purpose
        This AI-powered platform predicts molecular biological activity using machine learning.
        It's designed for rapid screening of drug candidates in early discovery stages.
        
        ### 🧠 Model Architecture
        - **Type:** Feed-Forward Neural Network
        - **Layers:** 3 fully connected layers with dropout
        - **Input:** Molecular features (SMILES-based)
        - **Output:** Binary activity prediction (Active/Inactive)
        
        ### 📊 Features Used
        
        **With RDKit (✅ Installed):**
        - Molecular Weight (MolWt)
        - Lipophilicity (LogP)
        - Hydrogen Bond Donors/Acceptors
        - Topological Polar Surface Area (TPSA)
        - Rotatable Bonds, Ring Counts
        - Aromatic/Aliphatic/Saturated Rings
        - Heteroatoms, Valence Electrons
        - Fraction of sp³ Carbons
        
        **Without RDKit (Fallback):**
        - SMILES string length
        - Number of digits in SMILES
        - Number of letters in SMILES
        
        ### 🚀 Getting Started
        1. **Train Model:** Run `python -m src.main --stage all`
        2. **Launch App:** Run `streamlit run app.py`
        3. **Make Predictions:** Enter SMILES notation or upload CSV
        
        ### 📝 SMILES Format
        SMILES (Simplified Molecular Input Line Entry System) is a notation for representing
        molecular structures as text strings.
        
        **Examples:**
        - `CCO` - Ethanol
        - `CC(=O)OC1=CC=CC=C1C(=O)O` - Aspirin
        - `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` - Caffeine
        
        ### 🔗 Resources
        - [SMILES Tutorial](http://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)
        - [RDKit Documentation](https://www.rdkit.org/docs/)
        - [Project GitHub](#)
        
        ---
        
        **Version:** 1.0.0 | **Built with:** Streamlit, PyTorch, Plotly
        """)

if __name__ == "__main__":
    set_seed(42)
    main()
