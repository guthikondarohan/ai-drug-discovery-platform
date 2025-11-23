"""
ChatGPT-style interface for AI Drug Discovery
Interactive chat interface with file upload capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import featurize_smiles
from models_fixed import SimpleTabularClassifier
from utils import set_seed

# Page configuration
st.set_page_config(
    page_title="AI Drug Discovery Chat",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-style interface
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #343541;
        padding: 0;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat container */
    .stChatFloatingInputContainer {
        bottom: 20px;
        background-color: #40414f;
        border-radius: 12px;
    }
    
    /* Messages */
    .stChatMessage {
        background-color: #444654;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    [data-testid="stChatMessageContent"] {
        color: #ececf1;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* User messages */
    .stChatMessage[data-testid*="user"] {
        background-color: #343541;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid*="assistant"] {
        background-color: #444654;
    }
    
    /* Input box */
    .stTextInput input {
        background-color: #40414f;
        color: #ececf1;
        border: 1px solid #565869;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #40414f;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #565869;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton button:hover {
        background-color: #1a7f64;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background-color: #2d2e3a;
        border-radius: 8px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #10a37f;
        font-size: 1.5rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    [data-testid="stSidebar"] * {
        color: #ececf1;
    }
    
    /* Code blocks */
    code {
        background-color: #2d2e3a;
        color: #d4d4d4;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ececf1;
    }
    
    /* File upload label */
    .uploadedFile {
        background-color: #2d2e3a;
        border-radius: 6px;
        color: #ececf1;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your AI Drug Discovery assistant. I can help you:\n\n• Predict molecular activity from SMILES notation\n• Analyze CSV files with molecular data\n• Calculate molecular descriptors using RDKit\n• Provide insights on drug-like properties\n\nWhat would you like to do today?"}
    ]

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.input_dim = 0

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = Path("results/model.pt")
    if not model_path.exists():
        return None, 0
    
    # Detect input dimension
    train_path = Path("data/processed/molecules_train.csv")
    if train_path.exists():
        df = pd.read_csv(train_path)
        feature_cols = [c for c in df.columns if c not in ['smiles', 'label', 'id']]
        input_dim = len(feature_cols)
    else:
        input_dim = 3
    
    try:
        model = SimpleTabularClassifier(input_dim=input_dim, hidden_dim=64, output_dim=1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, input_dim
    except Exception as e:
        return None, 0

def predict_molecule(smiles, model):
    """Predict activity for a molecule"""
    try:
        features_dict = featurize_smiles(smiles)
        if not features_dict:
            return None, None
        
        features = np.array(list(features_dict.values()), dtype=np.float32)
        
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logit = model(x)
            prob = torch.sigmoid(logit).item()
        
        return prob, features_dict
    except Exception as e:
        return None, str(e)

def analyze_csv(df, model):
    """Analyze any CSV file - handles both molecular and general data"""
    
    # Check if it's molecular data
    if 'smiles' in df.columns:
        results = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            prob, features = predict_molecule(smiles, model)
            if prob is not None:
                results.append({
                    'SMILES': smiles,
                    'Activity_Probability': f"{prob*100:.2f}%",
                    'Prediction': 'Active' if prob > 0.5 else 'Inactive',
                    'Confidence': f"{max(prob, 1-prob)*100:.1f}%"
                })
        
        if not results:
            return None, "❌ No valid SMILES found in file"
        return pd.DataFrame(results), None
    
    # Handle general CSV files (like airplane reviews)
    else:
        # Return basic analysis
        analysis = {
            'summary': f"📊 **Dataset Overview**\n\n• **Rows:** {len(df)}\n• **Columns:** {len(df.columns)}\n• **Column Names:** {', '.join(df.columns.tolist())}",
            'df': df.head(20)  # Show first 20 rows
        }
        return analysis, None

def format_features_markdown(features):
    """Format features as markdown"""
    md = "**Molecular Properties:**\n\n"
    for key, value in features.items():
        md += f"• **{key}**: {value:.3f}\n"
    return md

# Load model
if st.session_state.model is None:
    st.session_state.model, st.session_state.input_dim = load_model()

# Sidebar
with st.sidebar:
    st.markdown("### 🧬 AI Drug Discovery")
    st.markdown("---")
    
    if st.session_state.model:
        st.success("✅ Model Ready")
        feature_type = "RDKit Descriptors" if st.session_state.input_dim > 3 else "Basic Features"
        st.info(f"**Features:** {st.session_state.input_dim}\n\n**Type:** {feature_type}")
    else:
        st.error("❌ No Model")
        st.caption("Train with:\n```\npython -m src.main --stage all\n```")
    
    st.markdown("---")
    
    st.markdown("### 📚 Quick Guide")
    st.markdown("""
    **Upload Any CSV:**
    - Molecular data (with 'smiles')
    - General datasets (reviews, etc.)
    - Get instant analysis
    
    **Commands:**
    - Type SMILES notation
    - Ask questions
    
    **Examples:**
    - `CCO`
    - `What is LogP?`
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! How can I help you?"}
        ]
        st.rerun()
    
    st.markdown("---")
    st.caption(f"💊 Drug Discovery AI\n\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Main chat interface
st.title("💊 AI Drug Discovery Chat")

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display any attached data
        if "dataframe" in message:
            st.dataframe(message["dataframe"], use_container_width=True, key=f"df_{idx}")
        
        if "chart" in message:
            st.plotly_chart(message["chart"], use_container_width=True, key=f"chart_{idx}")

# File upload in chat - allow multiple files
uploaded_files = st.file_uploader(
    "📎 Upload CSV files (molecular data or general datasets)", 
    type=['csv'], 
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Check if already processed
        if uploaded_file.name not in st.session_state.get('processed_file_names', []):
            # Mark this file as processed
            if 'processed_file_names' not in st.session_state:
                st.session_state.processed_file_names = []
            st.session_state.processed_file_names.append(uploaded_file.name)
            
            df = pd.read_csv(uploaded_file)
            
            st.session_state.messages.append({
                "role": "user",
                "content": f"📄 Uploaded file: **{uploaded_file.name}** ({len(df)} rows)"
            })
            
            # Check if it's molecular data
            if 'smiles' in df.columns:
                # Molecular data
                with st.spinner(f"Analyzing molecules in {uploaded_file.name}..."):
                    results, error = analyze_csv(df, st.session_state.model)
                
                if error:
                    response = error
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # Molecular data analysis
                    active_count = (results['Prediction'] == 'Active').sum()
                    response = f"✅ **Molecular Analysis Complete for {uploaded_file.name}!**\n\n• **Total molecules:** {len(results)}\n• **Active predictions:** {active_count} ({active_count/len(results)*100:.1f}%)\n• **Inactive predictions:** {len(results)-active_count}\n\nResults shown below:"
                    
                    # Create distribution chart
                    fig = px.histogram(
                        results,
                        x='Activity_Probability',
                        color='Prediction',
                        title=f'Activity Distribution - {uploaded_file.name}',
                        color_discrete_map={'Active': '#10a37f', 'Inactive': '#ef4444'}
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#ececf1'
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "dataframe": results,
                        "chart": fig
                    })
            else:
                # General CSV - NOT molecular data
                response = f"ℹ️ **{uploaded_file.name} is not a molecular dataset**\n\nThis file doesn't contain a 'smiles' column, so it can't be analyzed for molecular activity.\n\n**Dataset Overview:**\n• **Rows:** {len(df)}\n• **Columns:** {len(df.columns)}\n• **Column Names:** {', '.join(df.columns.tolist()[:10])}\n\n"
                
                if len(df.columns) > 10:
                    response += f"*...and {len(df.columns) - 10} more columns*\n\n"
                
                response += "💡 To analyze molecular data, upload a CSV with a **'smiles'** column containing molecular structures."
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "dataframe": df.head(10)
                })
    
    if any(f.name not in st.session_state.get('processed_file_names', []) for f in uploaded_files):
        st.rerun()

# Chat input
if prompt := st.chat_input("Type a SMILES notation or ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process the prompt
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if it's a SMILES notation (simplified check)
            if any(c in prompt for c in ['C', 'N', 'O', 'c', '(', ')', '=', '#']) and len(prompt) < 200:
                # Likely a SMILES string
                if st.session_state.model:
                    prob, features = predict_molecule(prompt, st.session_state.model)
                    
                    if prob is not None:
                        activity = "**ACTIVE** 🟢" if prob > 0.5 else "**INACTIVE** 🔴"
                        confidence = max(prob, 1-prob) * 100
                        
                        response = f"""🔬 **Prediction Results**

**SMILES:** `{prompt}`

**Activity:** {activity}
**Probability:** {prob*100:.2f}%
**Confidence:** {confidence:.1f}%

{format_features_markdown(features)}
"""
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob*100,
                            title={'text': "Activity Probability"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#10a37f" if prob > 0.5 else "#ef4444"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#2d2e3a"},
                                    {'range': [50, 100], 'color': "#40414f"}
                                ],
                            }
                        ))
                        fig.update_layout(
                            height=300,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#ececf1'
                        )
                        
                        st.markdown(response)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "chart": fig
                        })
                    else:
                        response = f"❌ Invalid SMILES notation: `{prompt}`\n\nPlease check the format and try again."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = "❌ Model not loaded. Train a model first:\n```\npython -m src.main --stage all\n```"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Handle questions
            elif any(word in prompt.lower() for word in ['what', 'how', 'why', 'explain', 'help']):
                if 'logp' in prompt.lower():
                    response = """📚 **LogP (Lipophilicity)**

LogP measures how lipophilic (fat-loving) a molecule is. It's the partition coefficient between octanol and water.

**Key points:**
• Higher LogP = more lipophilic (fat-soluble)
• Lower LogP = more hydrophilic (water-soluble)
• Ideal range for drugs: 0-5
• Important for membrane permeability

**Drug discovery relevance:**
Affects absorption, distribution, and blood-brain barrier crossing."""

                elif 'smiles' in prompt.lower():
                    response = """📚 **SMILES Notation**

SMILES (Simplified Molecular Input Line Entry System) represents molecular structures as text.

**Examples:**
• `CCO` - Ethanol
• `CC(=O)OC1=CC=CC=C1C(=O)O` - Aspirin
• `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` - Caffeine

**Basic rules:**
• C, N, O = atoms
• = double bond, # triple bond
• () = branches
• Numbers = ring closures"""

                elif 'help' in prompt.lower():
                    response = """💡 **How to use this assistant**

**Predict molecular activity:**
Just type a SMILES notation, e.g., `CCO`

**Analyze multiple molecules:**
Upload a CSV file with a 'smiles' column

**Ask questions:**
Ask about molecular properties, descriptors, or drug discovery concepts

**Examples:**
• "What is LogP?"
• "Explain SMILES notation"
• Type: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`"""

                else:
                    response = f"""I understand you're asking about: "{prompt}"

I can help with:
• Molecular activity predictions (just type SMILES)
• CSV file analysis (upload files)
• Questions about molecular descriptors

Try asking:
• "What is LogP?"
• "Explain SMILES notation"
• Or type a SMILES like: `CCO`"""
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            else:
                response = f"""I'm not sure what you mean by "{prompt}".

**I can help you with:**
• 🔬 Predict activity: Type a SMILES notation
• 📊 Analyze data: Upload a CSV file
• 💡 Answer questions: Ask about molecular properties

**Try:**
• `CCO` (to predict ethanol activity)
• Upload a CSV with SMILES
• "What is LogP?"
• "Help" """
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    set_seed(42)
