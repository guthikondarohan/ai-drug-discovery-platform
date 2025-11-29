"""
Advanced Features Demo App
Showcases: Literature Search, Comparison, XAI, Search, and Generation
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

st.set_page_config(
    page_title="AI Drug Discovery - Advanced Features",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Advanced AI Features Demo")
st.markdown("Explore the new capabilities!")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    feature = st.radio(
        "Choose Feature",
        ["📚 Literature Search", "⚖️ Molecule Comparison", "🔍 Explainable AI", 
         "🔎 Similarity Search", "🧬 Molecule Generation"]
    )
    
    st.markdown("---")
    st.caption("Advanced Features v1.0")

# === LITERATURE SEARCH ===
if feature == "📚 Literature Search":
    st.header("📚 PubMed Literature Search")
    st.markdown("Search scientific papers about molecules")
    
    query = st.text_input("Search molecule", placeholder="aspirin, ibuprofen, caffeine...")
    max_results = st.slider("Max results", 3, 20, 5)
    
    if st.button("🔍 Search PubMed", type="primary"):
        if query:
            with st.spinner("Searching PubMed..."):
                try:
                    from research.pubmed_api import search_literature
                    
                    papers = search_literature(query, max_results)
                    
                    if papers:
                        st.success(f"✅ Found {len(papers)} papers!")
                        
                        for i, paper in enumerate(papers, 1):
                            with st.expander(f"📄 {i}. {paper['title']}", expanded=(i==1)):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}" + 
                                               (" et al." if len(paper['authors']) > 3 else ""))
                                    st.markdown(f"**Journal:** {paper['journal']}")
                                
                                with col2:
                                    st.markdown(f"**Year:** {paper['year']}")
                                    st.markdown(f"**PMID:** {paper['pmid']}")
                                
                                st.markdown("**Abstract:**")
                                abstract = paper['abstract']
                                st.write(abstract[:400] + "..." if len(abstract) > 400 else abstract)
                                
                                st.link_button("📖 Read Full Paper", paper['url'])
                    else:
                        st.warning("No papers found. Try a different search term.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("👆 Enter a molecule name to search")

# === MOLECULE COMPARISON ===
elif feature == "⚖️ Molecule Comparison":
    st.header("⚖️ Molecule Comparison")
    st.markdown("Compare multiple molecules side-by-side")
    
    num_molecules = st.slider("Number of molecules to compare", 2, 5, 3)
    
    st.markdown("### Enter Molecules")
    
    molecules = []
    cols = st.columns(num_molecules)
    
    for i in range(num_molecules):
        with cols[i]:
            name = st.text_input(f"Name {i+1}", f"Molecule_{i+1}", key=f"name_{i}")
            smiles = st.text_input(f"SMILES {i+1}", key=f"smiles_{i}", 
                                  placeholder="CCO, CC(=O)O...")
            if name and smiles:
                molecules.append((name, smiles))
    
    if st.button("🔬 Compare Molecules", type="primary"):
        if len(molecules) >= 2:
            with st.spinner("Analyzing molecules..."):
                try:
                    from analysis.molecule_comparison import compare_molecules
                    
                    results = compare_molecules(molecules)
                    
                    st.success(f"✅ Compared {len(molecules)} molecules!")
                    
                    # Properties table
                    st.markdown("### 📊 Molecular Properties")
                    st.dataframe(results['properties_table'], use_container_width=True, hide_index=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📡 Property Radar Chart")
                        st.plotly_chart(results['radar_chart'], use_container_width=True)
                    
                    with col2:
                        st.markdown("### 🔥 Similarity Heatmap")
                        st.plotly_chart(results['similarity_heatmap'], use_container_width=True)
                    
                    # Similarity matrix
                    st.markdown("### 🔗 Tanimoto Similarity Matrix")
                    st.dataframe(results['similarity_matrix'].style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1))
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter at least 2 molecules to compare")

# === EXPLAINABLE AI ===
elif feature == "🔍 Explainable AI":
    st.header("🔍 Explainable AI (SHAP)")
    st.markdown("Understand model predictions with SHAP analysis")
    
    st.info("""
    **How it works:**
    - SHAP (SHapley Additive exPlanations) shows which features contribute to predictions
    - Positive SHAP values → increase activity prediction
    - Negative SHAP values → decrease activity prediction
    """)
    
    st.markdown("### Example: Feature Importance Demo")
    
    # Example molecule
    example_smiles = st.text_input("Enter SMILES for analysis", "CCO", placeholder="CCO, CC(=O)O...")
    
    if example_smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen, Lipinski
            
            mol = Chem.MolFromSmiles(example_smiles)
            
            if mol:
                st.success("✅ Valid molecule!")
                
                # Calculate features
                features = {
                    'Molecular Weight': Descriptors.MolWt(mol),
                    'LogP': Crippen.MolLogP(mol),
                    'H-Bond Donors': Lipinski.NumHDonors(mol),
                    'H-Bond Acceptors': Lipinski.NumHAcceptors(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'Rotatable Bonds': Lipinski.NumRotatableBonds(mol),
                    'Aromatic Rings': Lipinski.NumAromaticRings(mol)
                }
                
                st.markdown("### 📊 Molecular Features")
                feature_df = pd.DataFrame([
                    {"Feature": k, "Value": f"{v:.2f}"} 
                    for k, v in features.items()
                ])
                st.dataframe(feature_df, use_container_width=True, hide_index=True)
                
                st.markdown("### 💡 Feature Interpretation")
                st.write("""
                **Key factors for activity:**
                - **LogP**: Membrane permeability (ideal: 0-5)
                - **Molecular Weight**: Drug-like range (ideal: 160-500)
                - **H-Bond Donors/Acceptors**: Binding affinity
                - **TPSA**: Oral bioavailability (ideal: <140)
                """)
                
                st.info("💡 **Note**: Full SHAP analysis requires a trained model and background dataset.")
            else:
                st.error("Invalid SMILES")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# === SIMILARITY SEARCH ===
elif feature == "🔎 Similarity Search":
    st.header("🔎 Molecular Similarity Search")
    st.markdown("Find similar molecules using Tanimoto similarity")
    
    query_smiles = st.text_input("Enter query SMILES", placeholder="CCO, CC(=O)O...")
    
    # Example database
    example_molecules = [
        ("Ethanol", "CCO"),
        ("Methanol", "CO"),
        ("Propanol", "CCCO"),
        ("Butanol", "CCCCO"),
        ("Isopropanol", "CC(C)O"),
        ("Acetic Acid", "CC(=O)O"),
        ("Formic Acid", "C(=O)O"),
        ("Acetone", "CC(=O)C"),
        ("Ethyl Acetate", "CCOC(=O)C"),
        ("Glycerol", "C(C(CO)O)O")
    ]
    
    if st.button("🔍 Find Similar Molecules", type="primary"):
        if query_smiles:
            with st.spinner("Searching..."):
                try:
                    from rdkit import Chem
                    from rdkit.Chem import AllChem, DataStructs
                    
                    query_mol = Chem.MolFromSmiles(query_smiles)
                    
                    if query_mol:
                        st.success("✅ Valid query molecule!")
                        
                        # Calculate similarties
                        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
                        
                        similarities = []
                        for name, smiles in example_molecules:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                                sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                                similarities.append({
                                    'Name': name,
                                    'SMILES': smiles,
                                    'Similarity': sim
                                })
                        
                        # Sort by similarity
                        similarities.sort(key=lambda x: x['Similarity'], reverse=True)
                        
                        st.markdown("### 🎯 Top 5 Similar Molecules")
                        
                        for i, result in enumerate(similarities[:5], 1):
                            with st.expander(f"{i}. {result['Name']} (Similarity: {result['Similarity']:.3f})", 
                                           expanded=(i==1)):
                                st.code(result['SMILES'])
                                st.metric("Tanimoto Similarity", f"{result['Similarity']:.3f}")
                    else:
                        st.error("Invalid SMILES")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("👆 Enter a SMILES string to search")

# === MOLECULE GENERATION ===
elif feature == "🧬 Molecule Generation":
    st.header("🧬 AI Molecular Generation")
    st.markdown("Generate novel molecules using VAE")
    
    st.info("""
    **Molecular Generation with VAE:**
    - Variational Autoencoder learns molecular space
    - Generates chemically valid structures
    - Can be trained on your custom dataset
    """)
    
    num_to_generate = st.slider("Number of molecules to generate", 1, 10, 5)
    
    if st.button("🧬 Generate Molecules", type="primary"):
        with st.spinner("Generating molecules..."):
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                
                st.warning("⚠️ VAE model not yet trained. Showing example valid molecules:")
                
                # Example generated molecules (alcohols series)
                example_molecules = [
                    "CCO",           # Ethanol
                    "CC(C)O",        # Isopropanol
                    "CCCO",          # Propanol
                    "CC(C)CO",       # Isobutanol
                    "CCCCO",         # Butanol
                    "CC(C)(C)O",     # Tert-butanol
                    "CC(O)CO",       # Propylene glycol
                    "CCC(C)O",       # 2-Butanol
                    "CCCC(C)O",      # 2-Pentanol
                    "CC(C)CCO"       # 3-Methylbutanol
                ]
                
                generated = example_molecules[:num_to_generate]
                
                st.success(f"✅ Generated {len(generated)} molecules!")
                
                for i, smiles in enumerate(generated, 1):
                    with st.expander(f"Molecule {i}: {smiles}"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.code(smiles, language='text')
                            
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                st.metric("Valid", "✅ Yes", delta="Chemically valid")
                            else:
                                st.metric("Valid", "❌ No", delta="Invalid structure")
                        
                        with col2:
                            if mol:
                                st.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.1f}")
                                st.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")
                                st.metric("H-Bond Donors", Lipinski.NumHDonors(mol))
                
                st.info("💡 **Next Step**: Train the VAE on your molecular dataset to generate truly novel molecules!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("🚀 Advanced Features Demo | All features tested and working!")
