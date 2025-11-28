"""
Visualization tools for multimodal scientific dataset.

Creates network graphs and interactive visualizations for molecule-protein relationships.
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict
import numpy as np


class MoleculeProteinNetwork:
    """
    Create and visualize molecule-protein interaction networks.
    """
    
    def __init__(self):
        self.G = nx.Graph()
    
    def add_interaction(self, molecule: str, protein: str, interaction_type: str = "binds"):
        """Add molecule-protein interaction to network."""
        self.G.add_node(molecule, node_type='molecule')
        self.G.add_node(protein, node_type='protein')
        self.G.add_edge(molecule, protein, interaction=interaction_type)
    
    def create_network_from_dataframe(self, df: pd.DataFrame):
        """
        Create network from research dataset.
        
        Args:
            df: DataFrame with Molecule and Target_Protein columns
        """
        for _, row in df.iterrows():
            self.add_interaction(
                row['Molecule'],
                row['Target_Protein'],
                interaction_type='inhibits/activates'
            )
    
    def visualize_interactive(self) -> go.Figure:
        """
        Create interactive Plotly network visualization.
        
        Returns:
            Plotly figure object
        """
        # Get positions using spring layout
        pos = nx.spring_layout(self.G, k=2, iterations=50)
        
        # Separate nodes by type
        molecule_nodes = [n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'molecule']
        protein_nodes = [n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'protein']
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#a0aec0'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create molecule nodes
        mol_x = [pos[node][0] for node in molecule_nodes]
        mol_y = [pos[node][1] for node in molecule_nodes]
        
        molecule_trace = go.Scatter(
            x=mol_x, y=mol_y,
            mode='markers+text',
            hoverinfo='text',
            text=molecule_nodes,
            textposition="top center",
            marker=dict(
                size=30,
                color='#667eea',
                line=dict(width=2, color='white')
            ),
            hovertext=[f"Molecule: {n}" for n in molecule_nodes],
            name='Molecules'
        )
        
        # Create protein nodes
        prot_x = [pos[node][0] for node in protein_nodes]
        prot_y = [pos[node][1] for node in protein_nodes]
        
        protein_trace = go.Scatter(
            x=prot_x, y=prot_y,
            mode='markers+text',
            hoverinfo='text',
            text=protein_nodes,
            textposition="bottom center",
            marker=dict(
                size=25,
                color='#f093fb',
                symbol='square',
                line=dict(width=2, color='white')
            ),
            hovertext=[f"Protein: {n}" for n in protein_nodes],
            name='Proteins'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, molecule_trace, protein_trace])
        
        fig.update_layout(
            title={
                'text': "Molecule-Protein Interaction Network",
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        
        return fig
    
    def get_statistics(self) -> Dict:
        """Get network statistics."""
        return {
            'num_molecules': len([n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'molecule']),
            'num_proteins': len([n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'protein']),
            'num_interactions': self.G.number_of_edges(),
            'avg_degree': np.mean([d for n, d in self.G.degree()]),
            'density': nx.density(self.G)
        }


def create_dataset_overview(df: pd.DataFrame) -> go.Figure:
    """
    Create overview visualization of research dataset.
    
    Args:
        df: Research dataset DataFrame
    
    Returns:
        Plotly figure with dataset overview
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Molecules in Dataset',
            'Target Proteins',
            'Organisms',
            'Protein Lengths'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'pie'}, {'type': 'histogram'}]
        ]
    )
    
    # Molecules bar chart
    fig.add_trace(
        go.Bar(
            x=df['Molecule'],
            y=[1] * len(df),
            name='Molecules',
            marker_color='#667eea'
        ),
        row=1, col=1
    )
    
    # Proteins bar chart
    fig.add_trace(
        go.Bar(
            x=df['Target_Protein'],
            y=[1] * len(df),
            name='Proteins',
            marker_color='#f093fb'
        ),
        row=1, col=2
    )
    
    # Organisms pie chart
    organism_counts = df['Organism'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=organism_counts.index,
            values=organism_counts.values,
            name='Organisms'
        ),
        row=2, col=1
    )
    
    # Protein length histogram
    fig.add_trace(
        go.Histogram(
            x=df['Protein_Length'].dropna(),
            name='Protein Length',
            marker_color='#764ba2'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Multimodal Scientific Dataset Overview"
    )
    
    return fig


def create_smiles_table(df: pd.DataFrame) -> go.Figure:
    """
    Create interactive table of SMILES and molecular data.
    
    Args:
        df: Research dataset DataFrame
    
    Returns:
        Plotly table figure
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Molecule', 'PubChem CID', 'SMILES', 'Target Protein', 'UniProt ID'],
            fill_color='#667eea',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[
                df['Molecule'],
                df['PubChem_CID'],
                df['SMILES'],
                df['Target_Protein'],
                df['UniProt_ID']
            ],
            fill_color='#1a1f3a',
            align='left',
            font=dict(color='white', size=11),
            height=30
        )
    )])
    
    fig.update_layout(
        title="Table 3.1: Cross-modal Scientific Dataset",
        height=400
    )
    
    return fig
