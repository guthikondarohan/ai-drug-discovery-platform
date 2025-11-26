"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    """Available model types."""
    MLP = "mlp"
    GNN = "gnn"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class PredictionRequest(BaseModel):
    """Request model for single molecule prediction."""
    
    smiles: str = Field(
        ...,
        description="SMILES notation of the molecule",
        example="CCO"
    )
    model_type: ModelType = Field(
        default=ModelType.MLP,
        description="Type of model to use for prediction"
    )
    include_features: bool = Field(
        default=True,
        description="Include molecular features in response"
    )
    include_fingerprints: bool = Field(
        default=False,
        description="Include molecular fingerprints in response"
    )
    
    @validator('smiles')
    def validate_smiles(cls, v):
        """Validate SMILES notation."""
        if not v or len(v.strip()) == 0:
            raise ValueError("SMILES cannot be empty")
        if len(v) > 1000:
            raise ValueError("SMILES too long (max 1000 characters)")
        return v.strip()


class MolecularFeatures(BaseModel):
    """Molecular features response."""
    
    descriptors: Dict[str, float] = Field(
        description="RDKit molecular descriptors"
    )
    fingerprints: Optional[Dict[str, List[int]]] = Field(
        default=None,
        description="Molecular fingerprints"
    )


class PredictionResponse(BaseModel):
    """Response model for single molecule prediction."""
    
    smiles: str = Field(description="Input SMILES notation")
    prediction: str = Field(description="Prediction class (Active/Inactive)")
    probability: float = Field(
        description="Activity probability",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        description="Model confidence",
        ge=0.0,
        le=1.0
    )
    model_used: str = Field(description="Model type used for prediction")
    features: Optional[MolecularFeatures] = Field(
        default=None,
        description="Molecular features"
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    
    smiles_list: List[str] = Field(
        ...,
        description="List of SMILES notations",
        min_items=1,
        max_items=1000
    )
    model_type: ModelType = Field(
        default=ModelType.MLP,
        description="Type of model to use"
    )
    include_features: bool = Field(
        default=False,
        description="Include features (slower for large batches)"
    )
    
    @validator('smiles_list')
    def validate_smiles_list(cls, v):
        """Validate list of SMILES."""
        if len(v) > 1000:
            raise ValueError("Maximum 1000 molecules per batch")
        return [s.strip() for s in v if s.strip()]


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    
    predictions: List[PredictionResponse]
    total_count: int
    active_count: int
    inactive_count: int
    processing_time_seconds: float


class ModelInfo(BaseModel):
    """Model metadata response."""
    
    name: str
    version: str
    architecture: str
    input_features: int
    trained_on: str
    performance_metrics: Dict[str, float]
    available: bool


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    models_loaded: List[str]
    rdkit_available: bool
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    detail: Optional[str] = None
    smiles: Optional[str] = None


class SimilarityRequest(BaseModel):
    """Request for molecular similarity search."""
    
    query_smiles: str = Field(
        ...,
        description="Query molecule SMILES"
    )
    smiles_list: List[str] = Field(
        ...,
        description="List of candidate molecules",
        max_items=10000
    )
    topk: int = Field(
        default=10,
        description="Number of top results",
        ge=1,
        le=100
    )
    threshold: float = Field(
        default=0.5,
        description="Minimum similarity threshold",
        ge=0.0,
        le=1.0
    )
    method: str = Field(
        default="tanimoto",
        description="Similarity metric"
    )


class SimilarityResult(BaseModel):
    """Single similarity result."""
    
    smiles: str
    similarity: float


class SimilarityResponse(BaseModel):
    """Similarity search response."""
    
    query_smiles: str
    results: List[SimilarityResult]
    method: str
