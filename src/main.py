"""
Main FastAPI application entry point.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.database import Base, engine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Edge AI AutoML Platform...")
    logger.info(f"Binding to port: {os.getenv('PORT', '8000')}")

    # Create database tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    # Create required directories
    logger.info("Creating required directories...")
    settings.dataset_path.mkdir(parents=True, exist_ok=True)
    settings.preset_dataset_path.mkdir(parents=True, exist_ok=True)
    settings.custom_dataset_path.mkdir(parents=True, exist_ok=True)
    settings.models_path.mkdir(parents=True, exist_ok=True)
    settings.pretrained_models_path.mkdir(parents=True, exist_ok=True)
    settings.custom_models_path.mkdir(parents=True, exist_ok=True)
    settings.optimized_models_path.mkdir(parents=True, exist_ok=True)

    logger.info("Edge AI AutoML Platform started successfully!")

    yield

    # Shutdown
    logger.info("Shutting down Edge AI AutoML Platform...")


# Create FastAPI application
app = FastAPI(
    title="Edge AI AutoML Platform",
    description="Intelligent AutoML Model Optimization Platform for Edge Deployment",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint - MUST respond immediately
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint - responds immediately."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Edge AI AutoML Platform",
        "version": "1.0.0",
        "description": "Intelligent AutoML Model Optimization Platform for Edge Deployment",
        "docs": "/docs",
        "status": "operational",
    }


# Import API routers AFTER app is created (lazy loading heavy dependencies)
from src.api import experiments, upload, optimize, results, performance, websocket

# Register API routers
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(optimize.router, prefix="/api/optimize", tags=["Optimization"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])
app.include_router(performance.router, prefix="/api/performance", tags=["Performance"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])


@app.get("/api/info", tags=["Info"])
async def api_info():
    """Get API information and available endpoints."""
    return {
        "platform": "Edge AI AutoML Platform",
        "version": "1.0.0",
        "endpoints": {
            "experiments": "/api/experiments",
            "upload": "/api/upload",
            "optimize": "/api/optimize",
            "results": "/api/results",
            "performance": "/api/performance",
            "websocket": "/ws/experiments/{experiment_id}",
        },
        "supported_frameworks": ["pytorch", "tensorflow"],
        "supported_datasets": ["mnist", "cifar10", "imagenet_subset"],
        "supported_devices": [
            "raspberry_pi_3b",
            "raspberry_pi_4",
            "raspberry_pi_5",
            "jetson_nano",
            "jetson_xavier_nx",
            "coral_dev_board",
        ],
        "optimization_techniques": {
            "pytorch": [
                "ptq_int8",
                "ptq_int4",
                "pruning_magnitude_unstructured",
                "pruning_magnitude_structured",
                "quantization_aware_training",
                "hybrid_prune_quantize",
            ],
            "tensorflow": [
                "ptq_int8",
                "ptq_int4",
                "pruning_magnitude_unstructured",
                "pruning_magnitude_structured",
                "quantization_aware_training",
                "hybrid_prune_quantize",
            ],
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run(
        "src.main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        log_level="info"
    )
