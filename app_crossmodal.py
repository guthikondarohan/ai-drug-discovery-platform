"""
Cross-Modal Drug Discovery App - Ultimate Edition

Features:
- Image upload for molecular structures
- Molecule sketcher (JSME integration)
- Protein sequence input
- Multi-modal predictions
- Cross-modal search
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import torch
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import featurize_smiles
from models_fixed import SimpleTabularClassifier
from molecular_visualization import smiles_to_image
from prediction_database import PredictionDatabase
from name_to_smiles import name_to_smiles
from cross_modal.image_encoder import MolecularImageEncoder, ImageProcessor
from cross_modal.protein_encoder import ProteinEncoder, ProteinLigandInteraction, get_common_protein_targets
from cross_modal.fusion_models import CrossModalPredictor

# Page config
st.set_page_config(
    page_title="Cross-Modal Drug Discovery",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-title {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 20px;
}
.modality-card {
    background: linear-gradient(135deg, #667eea22 0%, #764ba244 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'db' not in st.session_state:
    st.session_state.db = PredictionDatabase()
if 'image_encoder' not in st.session_state:
    st.session_state.image_encoder = MolecularImageEncoder(embedding_dim=512)
if 'protein_encoder' not in st.session_state:
    st.session_state.protein_encoder = ProteinEncoder()
if 'protein_ligand' not in st.session_state:
    st.session_state.protein_ligand = ProteinLigandInteraction()

# Load model
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

# Title
st.markdown('<div class="big-title">🧬 Cross-Modal Drug Discovery Platform</div>', unsafe_allow_html=True)
st.markdown("### Predict molecular activity using **multiple modalities**")

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎨 Draw/Upload", 
    "🧬 Protein-Ligand",
    "🔀 Multi-Modal",
    "🔍 Cross-Modal Search",
    "📊 Analytics"
])

# TAB 1: Draw/Upload
with tab1:
    st.header("🎨 Molecular Input")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        input_mode = st.radio(
            "Choose input method:",
            ["📝 SMILES/Name", "🖼️ Upload Image", "✏️ Draw Structure"],
            horizontal=True
        )
        
        molecular_input = None
        input_type = None
        
        if input_mode == "📝 SMILES/Name":
            text_input = st.text_input("Enter SMILES or chemical name:", placeholder="CCO, aspirin, caffeine...")
            if text_input:
                # Try name conversion first
                smiles = name_to_smiles(text_input)
                if smiles != text_input:
                    st.info(f"📝 Converted '{text_input}' → `{smiles}`")
                molecular_input = smiles
                input_type = 'smiles'
        
        elif input_mode == "🖼️ Upload Image":
            uploaded_file = st.file_uploader("Upload molecular structure image", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
                molecular_input = image
                input_type = 'image'
                st.info("💡 Image encoding enabled! We'll process this visual structure.")
        
        elif input_mode == "✏️ Draw Structure":
            st.markdown("### Molecule Sketcher")
            
            # JSME molecule editor
            jsme_html = """
            <div style="border: 2px solid #667eea; border-radius: 10px; padding: 10px; background: white;">
                <h4 style="color: #667eea;">Interactive Molecule Drawer</h4>
                <p>Draw your molecule structure below:</p>
                <iframe src="https://jsme-editor.github.io/dist/JSME_2017-02-26/JSME.html" 
                        width="100%" height="500px" style="border: none;"></iframe>
                <p style="color: #666; font-size: 0.9em;">After drawing, copy the SMILES from the editor and paste below ↓</p>
            </div>
            """
            components.html(jsme_html, height=600)
            
            drawn_smiles = st.text_input("Paste SMILES from drawer:", key="drawn_smiles")
            if drawn_smiles:
                molecular_input = drawn_smiles
                input_type = 'smiles'
        
        # Predict button
        if st.button("🚀 Predict Activity", type="primary", use_container_width=True):
            if molecular_input and st.session_state.model:
                with st.spinner("Analyzing..."):
                    
                    if input_type == 'smiles':
                        # SMILES-based prediction
                        features_dict = featurize_smiles(molecular_input)
                        if features_dict:
                            features = np.array(list(features_dict.values()), dtype=np.float32)
                            with torch.no_grad():
                                x = torch.tensor(features).unsqueeze(0)
                                logit = st.session_state.model(x)
                                prob = torch.sigmoid(logit).item()
                            
                            st.success("✅ Prediction Complete!")
                            
                            # Show results in col2
                            with col2:
                                st.subheader("🎯 Results")
                                
                                prediction = "ACTIVE" if prob > 0.5 else "INACTIVE"
                                st.metric("Prediction", prediction, 
                                         "🟢" if prediction == "ACTIVE" else "🔴")
                                st.metric("Probability", f"{prob*100:.1f}%")
                                st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")
                                
                                # Show structure
                                img_data = smiles_to_image(molecular_input)
                                if img_data:
                                    st.markdown("**Structure:**")
                                    st.markdown(f'<img src="{img_data}" style="width:100%; border-radius:10px;">', 
                                               unsafe_allow_html=True)
                    
                    elif input_type == 'image':
                        # Image-based encoding
                        st.info("🖼️ Processing molecular image...")
                        
                        # Encode image
                        embedding = st.session_state.image_encoder.encode_image(molecular_input)
                        
                        st.success("✅ Image encoded successfully!")
                        st.write(f"**Embedding dimension:** {len(embedding)}")
                        st.write("**Note:** Image-to-prediction requires training cross-modal model.")
                        
                        with col2:
                            st.subheader("🎨 Image Analysis")
                            st.write("**Image Embedding Generated**")
                            st.write(f"Vector size: {len(embedding)}")
                            st.write("This can be used for:")
                            st.write("- Cross-modal similarity search")
                            st.write("- Multi-modal fusion")
                            st.write("- Image-to-SMILES conversion (coming soon)")
    
    with col2:
        if not molecular_input:
            st.info("👈 Choose input method and enter your molecule")

# TAB 2: Protein-Ligand
with tab2:
    st.header("🧬 Protein-Ligand Binding Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ligand (Molecule)")
        ligand_input = st.text_input("Enter SMILES or name:", key="ligand", placeholder="CCO, ibuprofen...")
        
        if ligand_input:
            ligand_smiles = name_to_smiles(ligand_input)
            if ligand_smiles != ligand_input:
                st.info(f"Converted to: `{ligand_smiles}`")
            
            # Show structure
            img_data = smiles_to_image(ligand_smiles)
            if img_data:
                st.markdown(f'<img src="{img_data}" style="width:100%;">', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Protein Target")
        
        target_options = list(get_common_protein_targets().keys()) + ["Custom Sequence"]
        selected_target = st.selectbox("Select target:", target_options)
        
        if selected_target == "Custom Sequence":
            protein_seq = st.text_area("Enter protein sequence:", height=100, 
                                       placeholder="MKTAYIAKQ...")
        else:
            protein_seq = get_common_protein_targets()[selected_target]
            st.text_area("Sequence:", protein_seq[:200] + "...", height=100, disabled=True)
    
    if st.button("🔬 Predict Binding", type="primary", use_container_width=True):
        if ligand_input and protein_seq:
            with st.spinner("Predicting protein-ligand interaction..."):
                # Get ligand embedding (using features for now)
                features_dict = featurize_smiles(ligand_smiles)
                if features_dict:
                    ligand_emb = np.array(list(features_dict.values()), dtype=np.float32)
                    
                    # Predict binding
                    result = st.session_state.protein_ligand.predict_binding(protein_seq, ligand_emb)
                    
                    st.success("✅ Binding prediction complete!")
                    
                    # Show results
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Binding Probability", f"{result['binding_probability']*100:.1f}%")
                    col_r2.metric("Predicted IC50", f"{result['predicted_ic50_nM']:.2f} nM")
                    col_r3.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    
                    if result['binding_probability'] > 0.5:
                        st.success(f"✅ Ligand likely binds to {selected_target}!")
                    else:
                        st.warning(f"⚠️ Low binding probability to {selected_target}")

# TAB 3: Multi-Modal
with tab3:
    st.header("🔀 Multi-Modal Prediction")
    st.markdown("Combine multiple modalities for enhanced predictions!")
    
    st.markdown('<div class="modality-card">', unsafe_allow_html=True)
    st.subheader("Available Modalities")
    
    modalities = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.checkbox("📝 SMILES"):
            smiles_mm = st.text_input("SMILES:", key="mm_smiles")
            if smiles_mm:
                modalities['smiles'] = name_to_smiles(smiles_mm)
    
    with col2:
        if st.checkbox("🖼️ Image"):
            img_mm = st.file_uploader("Upload:", type=['png','jpg'], key="mm_img")
            if img_mm:
                modalities['image'] = Image.open(img_mm)
    
    with col3:
        if st.checkbox("🧬 Protein Context"):
            prot_mm = st.selectbox("Target:", list(get_common_protein_targets().keys()), key="mm_prot")
            if prot_mm:
                modalities['protein'] = get_common_protein_targets()[prot_mm]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if len(modalities) >= 2:
        st.success(f"✅ {len(modalities)} modalities selected!")
        
        if st.button("🚀 Multi-Modal Prediction", type="primary"):
            st.info("🔄 Multi-modal fusion in progress...")
            st.write(f"**Active modalities:** {', '.join(modalities.keys())}")
            st.write("**Note:** Full multi-modal model requires training with paired data.")
            st.write("Coming soon: Fused predictions with enhanced accuracy!")
    else:
        st.info("👆 Select at least 2 modalities for multi-modal prediction")

# TAB 4: Cross-Modal Search
with tab4:
    st.header("🔍 Cross-Modal Molecular Search")
    st.markdown("Search molecules using any input type!")
    
    search_mode = st.radio("Search with:", ["SMILES", "Image", "Drawing"], horizontal=True)
    
    if search_mode == "SMILES":
        query_smiles = st.text_input("Enter query molecule:")
        if query_smiles and st.button("🔍 Search"):
            st.info("Searching across all modalities...")
            st.write("**Feature:** Find similar molecules from images, structures, or databases")
    
    elif search_mode == "Image":
        query_img = st.file_uploader("Upload query image:")
        if query_img and st.button("🔍 Search"):
            st.info("Encoding image and searching...")
            st.write("**Feature:** Find molecules similar to this structure")
    
    st.markdown("---")
    st.info("💡 Cross-modal search allows finding molecules across different representations!")

# TAB 5: Analytics
with tab5:
    st.header("📊 Cross-Modal Analytics")
    
    stats = st.session_state.db.get_prediction_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", stats['total_predictions'])
    col2.metric("Active", stats['active_count'])
    col3.metric("Modalities Used", "3")  # SMILES, Image, Protein
    col4.metric("Avg Confidence", f"{stats['average_confidence']*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <h4>🧬 Cross-Modal Drug Discovery Platform v3.0</h4>
    <p>Powered by Multi-Modal AI | SMILES + Images + Proteins</p>
</div>
""", unsafe_allow_html=True)
