"""
Central Configuration for AURA Diamonds Visual Search Engine
"""

import os
from pathlib import Path

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

CATALOG_PATH = str(DATA_DIR / "catalog.json")
CLIP_EMBEDDINGS_PATH = str(DATA_DIR / "clip_embeddings.npz")
SCRAPED_PRODUCTS_PATH = str(DATA_DIR / "scraped_products.json")

# Model Configuration
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_DEVICE = "cpu"

# Cache Configuration
FEATURE_CACHE_CAPACITY = 256
URL_IMAGE_CACHE_CAPACITY = 200

# Server Defaults
DEFAULT_HOST = os.environ.get("HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("PORT", "8080"))
