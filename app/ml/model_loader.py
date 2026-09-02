"""
ML model loader — loads the attrition pipeline from disk at startup.
"""
import joblib
from pathlib import Path
from app.utils.config import MODEL_PATH, MODEL_V1_PATH
from app.utils.logger import ml_logger


def load_model(prefer_v1: bool = True):
    """Load the attrition prediction pipeline."""
    path = MODEL_V1_PATH if prefer_v1 and MODEL_V1_PATH.exists() else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run notebooks 07 and 09 first."
        )
    ml_logger.info(f"Loading model from {path}")
    pipeline = joblib.load(path)
    ml_logger.info(f"Model loaded: {type(pipeline).__name__}")
    return pipeline


# Singleton — loaded once at startup
_model = None


def get_model():
    """Return the cached model, loading it on first call."""
    global _model
    if _model is None:
        _model = load_model()
    return _model
