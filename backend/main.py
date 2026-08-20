"""
AURA DIAMONDS — OpenCLIP & Grown Brilliance Multi-Modal Visual Search Engine
FastAPI Application Entrypoint
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.config import FRONTEND_DIR
from backend.core.engine import OpenCLIPSearchEngine
from backend.api.routes import router as api_router, set_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up OpenCLIP Visual Search Engine
    logger.info("Initializing OpenCLIP Visual Search Engine on startup...")
    engine = OpenCLIPSearchEngine()
    set_engine(engine)
    logger.info("OpenCLIP Search Engine loaded and ready for queries!")
    yield
    logger.info("Shutting down AURA Visual Search Engine.")


app = FastAPI(
    title="AURA Diamonds — OpenCLIP Visual Search API",
    description="State-of-the-art OpenCLIP ViT-B-32 jewelry vision search engine with Interactive Target Object Hotspots & Adjustable Frame Cropping.",
    version="3.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Endpoints
app.include_router(api_router)

# Mount Frontend UI Static Assets
frontend_path = str(FRONTEND_DIR)
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_ui():
        index_file = os.path.join(frontend_path, "index.html")
        return FileResponse(index_file)
