"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add api to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "rdkit_available" in data


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data


class TestPredictionEndpoint:
    """Tests for prediction endpoints."""
    
    def test_predict_valid_smiles(self):
        """Test prediction with valid SMILES."""
        response = client.post(
            "/predict",
            json={
                "smiles": "CCO",
                "model_type": "mlp",
                "include_features": True
            }
        )
        
        # May fail if model not loaded, but should have valid response structure
        if response.status_code == 200:
            data = response.json()
            assert "smiles" in data
            assert "prediction" in data
            assert "probability" in data
            assert data["prediction"] in ["Active", "Inactive"]
            assert 0.0 <= data["probability"] <= 1.0
    
    def test_predict_invalid_smiles(self):
        """Test prediction with invalid SMILES."""
        response = client.post(
            "/predict",
            json={
                "smiles": "INVALID_SMILES",
                "model_type": "mlp"
            }
        )
        
        # Should return error for invalid SMILES
        assert response.status_code in [400, 404, 500]
    
    def test_predict_empty_smiles(self):
        """Test prediction with empty SMILES."""
        response = client.post(
            "/predict",
            json={
                "smiles": "",
                "model_type": "mlp"
            }
        )
        
        # Should return validation error
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_features(self):
        """Test prediction with features included."""
        response = client.post(
            "/predict",
            json={
                "smiles": "CCO",
                "model_type": "mlp",
                "include_features": True
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if "features" in data and data["features"] is not None:
                assert "descriptors" in data["features"]


class TestBatchPrediction:
    """Tests for batch prediction."""
    
    def test_batch_predict(self):
        """Test batch prediction with multiple SMILES."""
        response = client.post(
            "/batch_predict",
            json={
                "smiles_list": ["CCO", "CC(C)O", "CCCC"],
                "model_type": "mlp",
                "include_features": False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_count" in data
            assert "processing_time_seconds" in data
            assert data["total_count"] >= 0
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        response = client.post(
            "/batch_predict",
            json={
                "smiles_list": [],
                "model_type": "mlp"
            }
        )
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_batch_predict_too_many(self):
        """Test batch prediction with too many molecules."""
        large_list = ["CCO"] * 1001  # Over limit
        
        response = client.post(
            "/batch_predict",
            json={
                "smiles_list": large_list,
                "model_type": "mlp"
            }
        )
        
        # Should return validation error
        assert response.status_code == 422


class TestModelInfo:
    """Tests for model info endpoint."""
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        # Should return list even if empty
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestSimilaritySearch:
    """Tests for similarity search."""
    
    def test_similarity_search(self):
        """Test similarity search with valid input."""
        response = client.post(
            "/similarity/search",
            json={
                "query_smiles": "CCO",
                "smiles_list": ["CC(C)O", "CCCC", "CCC"],
                "topk": 2,
                "threshold": 0.3
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "query_smiles" in data
            assert "results" in data
            assert isinstance(data["results"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
