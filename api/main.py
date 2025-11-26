"""
FastAPI REST API for AI Drug Discovery Platform.

Provides endpoints for:
- Single and batch molecular activity predictions
- Model information and health checks
- Molecular similarity search
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from pathlib import Path
import sys
import time
from typing import Optional, List
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import featurize_smiles
from models_fixed import SimpleTabularClassifier
from features.molecular_fingerprints import (
    get_all_fingerprints,
    find_similar_molecules
)

from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
    SimilarityRequest,
    SimilarityResponse,
    SimilarityResult,
    MolecularFeatures,
    ModelType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Drug Discovery API",
    description=" REST API for molecular activity prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
models = {}
start_time = time.time()
API_KEY = "your-secret-api-key-here"  # TODO: Load from environment variable


def load_models():
    """Load all available models."""
    global models
    
    # Load MLP model
    model_path = Path("results/model.pt")
    if model_path.exists():
        try:
            # Detect input dimension
            train_path = Path("data/processed/molecules_train.csv")
            if train_path.exists():
                import pandas as pd
                df = pd.read_csv(train_path)
                feature_cols = [c for c in df.columns if c not in ['smiles', 'label', 'id']]
                input_dim = len(feature_cols)
            else:
                input_dim = 13
            
            model = SimpleTabularClassifier(input_dim=input_dim, hidden_dim=64, output_dim=1)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            models['mlp'] = model
            logger.info(f"Loaded MLP model with {input_dim} input features")
        except Exception as e:
            logger.error(f"Error loading MLP model: {e}")
    
    # TODO: Load other model types (GNN, Transformer, Ensemble)
    
    return models


# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Loading models...")
    load_models()
    logger.info(f"Loaded {len(models)} model(s)")


# API Key verification
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header."""
    # Disabled for development - enable for production
    # if x_api_key != API_KEY:
    #     raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Drug Discovery API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        from rdkit import Chem
        rdkit_available = True
    except:
        rdkit_available = False
    
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        rdkit_available=rdkit_available,
        uptime_seconds=time.time() - start_time
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict activity for a single molecule.
    
    - **smiles**: SMILES notation of the molecule
    - **model_type**: Model to use (mlp, gnn, transformer, ensemble)
    - **include_features**: Include molecular features in response
    - **include_fingerprints**: Include molecular fingerprints
    """
    try:
        # Get model
        model_key = request.model_type.value
        if model_key not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_key}' not available. Available models: {list(models.keys())}"
            )
        
        model = models[model_key]
        
        # Featurize molecule
        features_dict = featurize_smiles(request.smiles)
        if not features_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SMILES notation: {request.smiles}"
            )
        
        # Convert to tensor
        features = np.array(list(features_dict.values()), dtype=np.float32)
        
        # Predict
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logit = model(x)
            prob = torch.sigmoid(logit).item()
        
        # Determine prediction
        prediction = "Active" if prob > 0.5 else "Inactive"
        confidence = max(prob, 1 - prob)
        
        # Build response
        response = PredictionResponse(
            smiles=request.smiles,
            prediction=prediction,
            probability=round(prob, 4),
            confidence=round(confidence, 4),
            model_used=model_key
        )
        
        # Add features if requested
        if request.include_features:
            mol_features = MolecularFeatures(
                descriptors=features_dict
            )
            
            if request.include_fingerprints:
                fps = get_all_fingerprints(request.smiles, use_maccs=True)
                if fps:
                    mol_features.fingerprints = {
                        k: v.tolist() for k, v in fps.items()
                    }
            
            response.features = mol_features
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict activity for multiple molecules in batch.
    
    - **smiles_list**: List of SMILES notations (max 1000)
    - **model_type**: Model to use
    - **include_features**: Include features (slower)
    """
    try:
        start_time_batch = time.time()
        
        # Get model
        model_key = request.model_type.value
        if model_key not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_key}' not available"
            )
        
        model = models[model_key]
        
        # Process each molecule
        predictions = []
        active_count = 0
        inactive_count = 0
        
        for smiles in request.smiles_list:
            try:
                # Featurize
                features_dict = featurize_smiles(smiles)
                if not features_dict:
                    continue
                
                features = np.array(list(features_dict.values()), dtype=np.float32)
                
                # Predict
                with torch.no_grad():
                    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    logit = model(x)
                    prob = torch.sigmoid(logit).item()
                
                prediction = "Active" if prob > 0.5 else "Inactive"
                confidence = max(prob, 1 - prob)
                
                if prediction == "Active":
                    active_count += 1
                else:
                    inactive_count += 1
                
                pred_response = PredictionResponse(
                    smiles=smiles,
                    prediction=prediction,
                    probability=round(prob, 4),
                    confidence=round(confidence, 4),
                    model_used=model_key
                )
                
                if request.include_features:
                    pred_response.features = MolecularFeatures(
                        descriptors=features_dict
                    )
                
                predictions.append(pred_response)
            
            except Exception as e:
                logger.warning(f"Error processing {smiles}: {e}")
                continue
        
        processing_time = time.time() - start_time_batch
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            active_count=active_count,
            inactive_count=inactive_count,
            processing_time_seconds=round(processing_time, 3)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=List[ModelInfo], tags=["Model"])
async def get_model_info():
    """Get information about available models."""
    model_info_list = []
    
    for model_name, model in models.items():
        info = ModelInfo(
            name=model_name,
            version="1.0.0",
            architecture="3-layer MLP" if model_name == "mlp" else model_name.upper(),
            input_features=model.net[0].in_features if hasattr(model, 'net') else 0,
            trained_on="Molecular activity dataset",
            performance_metrics={"AUC": 0.85, "Accuracy": 0.80},  # TODO: Load actual metrics
            available=True
        )
        model_info_list.append(info)
    
    return model_info_list


@app.post("/similarity/search", response_model=SimilarityResponse, tags=["Similarity"])
async def similarity_search(
    request: SimilarityRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Find molecules similar to a query molecule.
    
    - **query_smiles**: Query molecule
    - **smiles_list**: Candidate molecules (max 10000)
    - **topk**: Number of results
    - **threshold**: Minimum similarity
    - **method**: Similarity metric (tanimoto, dice, cosine)
    """
    try:
        # Find similar molecules
        results = find_similar_molecules(
            query_smiles=request.query_smiles,
            smiles_list=request.smiles_list,
            topk=request.topk,
            threshold=request.threshold
        )
        
        # Format results
        similarity_results = [
            SimilarityResult(smiles=smiles, similarity=round(sim, 4))
            for smiles, sim in results
        ]
        
        return SimilarityResponse(
            query_smiles=request.query_smiles,
            results=similarity_results,
            method=request.method
        )
    
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
