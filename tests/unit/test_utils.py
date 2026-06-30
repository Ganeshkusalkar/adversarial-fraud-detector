import os
import json
import shutil
import tempfile
import logging
import pytest
from src.utils.logger import setup_logger
from src.utils.model_registry import ModelRegistry

def test_setup_logger_basic():
    logger_name = "test_logger_unique_1"
    logger = setup_logger(logger_name)
    assert logger.name == logger_name
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1

def test_setup_logger_with_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger_name = "test_logger_unique_2"
    logger = setup_logger(logger_name, log_file=str(log_file))
    
    assert logger.name == logger_name
    assert os.path.exists(log_file)
    
    # Clean up file handler to release the file lock
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)

def test_model_registry_operations():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_dir = os.path.join(tmpdir, "registry")
        prod_dir = os.path.join(tmpdir, "production")
        
        # Create versions in registry
        v1_dir = os.path.join(registry_dir, "v1")
        os.makedirs(v1_dir, exist_ok=True)
        
        # Write dummy model files
        with open(os.path.join(v1_dir, "fraud_model.onnx"), "w") as f:
            f.write("mock-onnx-v1")
        with open(os.path.join(v1_dir, "scaler.pkl"), "w") as f:
            f.write("mock-scaler-v1")
        with open(os.path.join(v1_dir, "prob_calibrator.pkl"), "w") as f:
            f.write("mock-calibrator-v1")
        with open(os.path.join(v1_dir, "metadata.json"), "w") as f:
            json.dump({"version": "v1", "auc": 0.85}, f)
            
        registry = ModelRegistry(registry_dir=registry_dir, prod_dir=prod_dir)
        
        # Verify initial state
        assert os.path.exists(registry_dir)
        assert os.path.exists(prod_dir)
        
        # Promote v1
        registry.promote_to_production("v1")
        
        # Check files in production folder
        assert os.path.exists(os.path.join(prod_dir, "fraud_model.onnx"))
        assert os.path.exists(os.path.join(prod_dir, "scaler.pkl"))
        assert os.path.exists(os.path.join(prod_dir, "prob_calibrator.pkl"))
        
        # Check active version metadata
        active_meta_path = os.path.join(prod_dir, "active_version.json")
        assert os.path.exists(active_meta_path)
        with open(active_meta_path, "r") as f:
            metadata = json.load(f)
            assert metadata["version"] == "v1"
            assert metadata["auc"] == 0.85
            
        # Rollback (which just promotes again)
        # First write a v2
        v2_dir = os.path.join(registry_dir, "v2")
        os.makedirs(v2_dir, exist_ok=True)
        with open(os.path.join(v2_dir, "fraud_model.onnx"), "w") as f:
            f.write("mock-onnx-v2")
        with open(os.path.join(v2_dir, "metadata.json"), "w") as f:
            json.dump({"version": "v2", "auc": 0.87}, f)
            
        registry.promote_to_production("v2")
        
        # Verify promoted to v2
        with open(active_meta_path, "r") as f:
            metadata = json.load(f)
            assert metadata["version"] == "v2"
            
        # Rollback to v1
        registry.rollback("v1", reason="V2 performance issues")
        
        # Verify rolled back to v1
        with open(active_meta_path, "r") as f:
            metadata = json.load(f)
            assert metadata["version"] == "v1"
            
        # Verify promote invalid version raises error
        with pytest.raises(ValueError):
            registry.promote_to_production("non_existent_version")
