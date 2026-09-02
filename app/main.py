"""
Enterprise HR AI — FastAPI Application Entry Point
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import attrition, dashboard, skills
from app.ml.model_loader import get_model
from app.utils.config import API_TITLE, API_VERSION, API_DESCRIPTION
from app.utils.logger import app_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — pre-load model at startup."""
    app_logger.info("=" * 60)
    app_logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    app_logger.info("=" * 60)
    
    # Pre-load model to avoid cold-start latency on first request
    try:
        model = get_model()
        app_logger.info(f"Model pre-loaded successfully: {type(model).__name__}")
    except FileNotFoundError as e:
        app_logger.error(f"Model not found — predictions will fail: {e}")
    except Exception as e:
        app_logger.error(f"Model loading error: {e}")
    
    app_logger.info("Application startup complete. Ready to serve.")
    yield
    
    app_logger.info("Application shutting down.")


# ── Create application ──
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow Streamlit frontend) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──
app.include_router(attrition.router)
app.include_router(dashboard.router)
app.include_router(skills.router)


# ── Health check ──
@app.get("/health", tags=["Health"])
async def health_check():
    """Returns API health status."""
    try:
        model = get_model()
        model_status = "loaded"
        model_name = type(model).__name__
    except Exception as e:
        model_status = f"error: {e}"
        model_name = "unknown"
    
    return {
        "status": "healthy",
        "api_version": API_VERSION,
        "model_status": model_status,
        "model_name": model_name,
    }


@app.get("/", tags=["Root"])
async def root():
    """API root — returns available endpoints."""
    return {
        "message": "Enterprise HR AI — Workforce Intelligence API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "attrition_prediction": "POST /predict/attrition",
            "dashboard_summary": "GET /dashboard/summary",
            "attrition_by_dept": "GET /dashboard/attrition-by-department",
            "skill_gaps": "GET /dashboard/skill-gaps",
            "recommendations": "GET /dashboard/recommendations",
            "employee_profile": "GET /employees/{employee_id}",
        },
    }
