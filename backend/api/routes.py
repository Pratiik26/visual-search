"""
API Route Handlers for Visual Search, Object Detection, Catalog Browsing & System Health
"""

import os
import io
import time
import base64
import ssl
import logging
import asyncio
import urllib.request
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, File, UploadFile, Form, Query, HTTPException, Depends
from PIL import Image

from backend.config import FRONTEND_DIR, URL_IMAGE_CACHE_CAPACITY
from backend.core.engine import OpenCLIPSearchEngine
from backend.core.detector import NyrisRegionDetector
from backend.api.schemas import (
    ImageUrlSearchRequest,
    AttributeTaxonomyResponse,
    ImageMetadataResponse,
    ImageUrlMetadataRequest
)

logger = logging.getLogger(__name__)

router = APIRouter()

_URL_IMAGE_CACHE: Dict[str, bytes] = {}
_engine_instance: Optional[OpenCLIPSearchEngine] = None


def get_engine() -> OpenCLIPSearchEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = OpenCLIPSearchEngine()
    return _engine_instance


def set_engine(engine: OpenCLIPSearchEngine):
    global _engine_instance
    _engine_instance = engine


def load_image_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def load_image_url(url: str) -> Image.Image:
    url_str = url.strip()
    if url_str in _URL_IMAGE_CACHE:
        return load_image_bytes(_URL_IMAGE_CACHE[url_str])

    # 1. Base64 data URL
    if url_str.startswith("data:image"):
        try:
            _, base64_data = url_str.split(",", 1)
            img_bytes = base64.b64decode(base64_data)
            if len(_URL_IMAGE_CACHE) < URL_IMAGE_CACHE_CAPACITY:
                _URL_IMAGE_CACHE[url_str] = img_bytes
            return load_image_bytes(img_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data URL: {e}")

    # 2. Local relative static path
    if url_str.startswith("/static/") or url_str.startswith("static/"):
        rel_path = url_str.replace("/static/", "").replace("static/", "")
        local_file = os.path.join(str(FRONTEND_DIR), rel_path)
        if os.path.exists(local_file):
            with open(local_file, "rb") as f:
                data = f.read()
                if len(_URL_IMAGE_CACHE) < URL_IMAGE_CACHE_CAPACITY:
                    _URL_IMAGE_CACHE[url_str] = data
                return load_image_bytes(data)

    # 3. Local filesystem path
    if os.path.exists(url_str):
        with open(url_str, "rb") as f:
            data = f.read()
            if len(_URL_IMAGE_CACHE) < URL_IMAGE_CACHE_CAPACITY:
                _URL_IMAGE_CACHE[url_str] = data
            return load_image_bytes(data)

    # 4. Remote HTTP/HTTPS URL
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            url_str,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
            }
        )
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = resp.read()
            if len(_URL_IMAGE_CACHE) < URL_IMAGE_CACHE_CAPACITY:
                _URL_IMAGE_CACHE[url_str] = data
            return load_image_bytes(data)
    except Exception as e:
        logger.error(f"Failed to fetch image URL {url_str}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/api/v1/health", summary="System Health & Stats", tags=["System"])
async def health_check(engine: OpenCLIPSearchEngine = Depends(get_engine)):
    return {
        "status": "healthy",
        "model": "OpenCLIP ViT-B-32",
        "total_products": len(engine.products),
        "indexed_products": len(engine.products),
        "indexed_views": len(engine.indexed_items),
        "device": str(engine.device)
    }


@router.get("/api/v1/metadata/attributes", response_model=AttributeTaxonomyResponse, summary="Taxonomy Attributes", tags=["Catalog"])
async def get_taxonomy(engine: OpenCLIPSearchEngine = Depends(get_engine)):
    tax = engine.catalog.get("taxonomy", {})
    return {
        "shapes": tax.get("shapes", []),
        "band_colors": tax.get("band_colors", []),
        "band_types": tax.get("band_types", []),
        "band_architectures": tax.get("band_architectures", []),
        "prong_styles": tax.get("prong_styles", []),
        "styles": tax.get("styles", []),
        "total_products": len(engine.products),
        "total_search_images": len(engine.indexed_items)
    }


@router.get("/api/v1/products", summary="Browse Catalog Products", tags=["Catalog"])
async def list_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    shape: Optional[str] = Query(None),
    band_color: Optional[str] = Query(None),
    style: Optional[str] = Query(None),
    engine: OpenCLIPSearchEngine = Depends(get_engine)
):
    all_prods = list(engine.products.values())

    if shape:
        s_l = shape.lower()
        all_prods = [p for p in all_prods if s_l in p.get("shape", "").lower() or s_l in p.get("title", "").lower()]
    if band_color:
        c_l = band_color.lower()
        all_prods = [p for p in all_prods if c_l in p.get("band_color", "").lower()]
    if style:
        st_l = style.lower()
        all_prods = [p for p in all_prods if st_l in p.get("style", "").lower()]

    total = len(all_prods)
    start = (page - 1) * limit
    end = start + limit
    page_items = all_prods[start:end]

    return {
        "success": True,
        "page": page,
        "limit": limit,
        "total_items": total,
        "total_pages": max(1, (total + limit - 1) // limit),
        "products": page_items
    }


@router.post("/find/v2/regions", summary="AI Ring Region Detection", tags=["AI Region Detection"])
async def detect_ring_regions(file: UploadFile = File(...), engine: OpenCLIPSearchEngine = Depends(get_engine)):
    data = await file.read()
    img = load_image_bytes(data)
    res = await asyncio.to_thread(NyrisRegionDetector.detect_regions, img, engine)
    return {
        "modelName": "aura-openclip-vit-detector",
        "modelVersion": "3.5",
        "confidence": res["confidence"],
        "primary_region": res["region"],
        "regions": res.get("all_regions", [res["region"]])
    }


@router.post("/aiSearchProducts", summary="Grown Brilliance Visual Search", tags=["Visual Search"])
@router.post("/api/v1/search/image", summary="Search by Uploaded Ring Image", tags=["Visual Search"])
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(12),
    shape: Optional[str] = Form(None),
    band_color: Optional[str] = Form(None),
    band_architecture: Optional[str] = Form(None),
    prong_style: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
    crop_x1: Optional[float] = Form(None),
    crop_y1: Optional[float] = Form(None),
    crop_x2: Optional[float] = Form(None),
    crop_y2: Optional[float] = Form(None),
    engine: OpenCLIPSearchEngine = Depends(get_engine)
):
    data = await file.read()
    img = load_image_bytes(data)

    crop_box = None
    if crop_x1 is not None and crop_y1 is not None and crop_x2 is not None and crop_y2 is not None:
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

    t0 = time.time()
    res = await asyncio.to_thread(
        engine.search,
        query_image=img,
        top_k=top_k,
        shape=shape,
        metal=band_color,
        architecture=band_architecture,
        prong=prong_style,
        style=style,
        crop_box=crop_box
    )
    logger.info(f"OpenCLIP search completed in {(time.time() - t0) * 1000:.1f}ms")
    return res


@router.post("/api/v1/search/url", summary="Search by Image URL", tags=["Visual Search"])
async def search_by_url(
    payload: ImageUrlSearchRequest,
    engine: OpenCLIPSearchEngine = Depends(get_engine)
):
    img = load_image_url(payload.image_url)
    crop_box = None
    if getattr(payload, "crop_x1", None) is not None:
        crop_box = (payload.crop_x1, payload.crop_y1, payload.crop_x2, payload.crop_y2)

    return await asyncio.to_thread(
        engine.search,
        query_image=img,
        top_k=payload.top_k or 12,
        shape=payload.shape,
        metal=payload.band_color,
        architecture=payload.band_architecture,
        prong=payload.prong_style,
        style=payload.style,
        crop_box=crop_box
    )


@router.get("/api/v1/search/by-sku/{sku}", summary="Visual Similarity Search by SKU", tags=["Visual Search"])
async def search_by_sku(
    sku: str,
    top_k: int = Query(10, ge=1, le=50),
    engine: OpenCLIPSearchEngine = Depends(get_engine)
):
    return await asyncio.to_thread(engine.search_by_sku, sku=sku, top_k=top_k)


# ============================================================================
# IMAGE METADATA EXTRACTION ENDPOINTS
# ============================================================================

@router.post(
    "/api/v1/extract-metadata",
    response_model=ImageMetadataResponse,
    summary="Extract Ring Metadata from Uploaded Image",
    tags=["Image Metadata Extraction"]
)
@router.post(
    "/api/v1/metadata/extract",
    response_model=ImageMetadataResponse,
    summary="Extract Ring Metadata from Uploaded Image (Alias)",
    tags=["Image Metadata Extraction"]
)
async def extract_image_metadata_file(
    file: UploadFile = File(..., description="Ring image file (JPEG, PNG, WebP)"),
    crop_x1: Optional[float] = Form(None, description="Optional crop left coordinate"),
    crop_y1: Optional[float] = Form(None, description="Optional crop top coordinate"),
    crop_x2: Optional[float] = Form(None, description="Optional crop right coordinate"),
    crop_y2: Optional[float] = Form(None, description="Optional crop bottom coordinate"),
    engine: OpenCLIPSearchEngine = Depends(get_engine)
):
    """
    Upload any ring image (via POST multipart/form-data) to extract complete visual metadata:
    - Center diamond shape cut (Round, Oval, Princess, Cushion, Emerald, Pear, etc.)
    - Band metal tone & color (14K Yellow Gold, White Gold, Rose Gold, Platinum)
    - Band architecture / shank (Straight Shank, Split Shank, Twisted Shank, etc.)
    - Prong setting style (Classic 4-Prong, 6-Prong, Bezel, Halo, etc.)
    - Ring design style (Solitaire, Halo, Three-Stone, Modern Classic, Vintage)
    - Confidence scores & bounding box / ROI crop coordinates
    """
    data = await file.read()
    img = load_image_bytes(data)

    crop_box = None
    if crop_x1 is not None and crop_y1 is not None and crop_x2 is not None and crop_y2 is not None:
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

    return await asyncio.to_thread(engine.extract_metadata, query_image=img, crop_box=crop_box)


@router.post(
    "/api/v1/metadata/extract-url",
    response_model=ImageMetadataResponse,
    summary="Extract Ring Metadata from Remote Image URL or Base64",
    tags=["Image Metadata Extraction"]
)
async def extract_image_metadata_url(
    payload: ImageUrlMetadataRequest,
    engine: OpenCLIPSearchEngine = Depends(get_engine)
):
    """
    Extract ring metadata by providing an image URL or base64 data string in JSON body.
    """
    img = load_image_url(payload.image_url)

    crop_box = None
    if payload.crop_x1 is not None and payload.crop_y1 is not None and payload.crop_x2 is not None and payload.crop_y2 is not None:
        crop_box = (payload.crop_x1, payload.crop_y1, payload.crop_x2, payload.crop_y2)

    return await asyncio.to_thread(engine.extract_metadata, query_image=img, crop_box=crop_box)

