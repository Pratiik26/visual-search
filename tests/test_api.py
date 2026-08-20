"""
Automated Test Suite for Diamond Ring Visual Search API
"""

import os
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

client = TestClient(app)


def test_health_check():
    """Verify health check endpoint returns 200 and valid system info."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_products" in data
    assert data["indexed_products"] > 0


def test_metadata_attributes():
    """Verify metadata taxonomy endpoint returns all required attributes."""
    response = client.get("/api/v1/metadata/attributes")
    assert response.status_code == 200
    data = response.json()
    
    # Verify core taxonomies exist
    assert "shapes" in data and len(data["shapes"]) >= 10
    assert "band_colors" in data and len(data["band_colors"]) >= 4
    assert "band_types" in data and len(data["band_types"]) > 0
    assert "band_architectures" in data and len(data["band_architectures"]) > 0
    assert "prong_styles" in data and len(data["prong_styles"]) > 0
    assert "styles" in data and len(data["styles"]) > 0
    assert data["total_products"] > 0


def test_list_products_and_pagination():
    """Verify product browsing with pagination and shape filter."""
    response = client.get("/api/v1/products?page=1&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["products"]) == 5
    assert data["total_items"] > 0
    
    # Test shape filter
    response_round = client.get("/api/v1/products?shape=Round&limit=5")
    assert response_round.status_code == 200
    data_round = response_round.json()
    for prod in data_round["products"]:
        assert "Round" in prod["shape"] or "Round" in prod["title"]


def test_visual_search_by_url():
    """Verify visual search via remote image URL returns accurate metadata."""
    sample_url = "https://overnightmountings.s3.amazonaws.com/85121.jpg"
    response = client.post("/api/v1/search/url", json={
        "image_url": sample_url,
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "query_analysis" in data
    assert len(data["results"]) > 0
    
    # Check first result metadata fields
    first_result = data["results"][0]
    required_fields = [
        "rank", "similarity_score", "product_id", "style_number", "title",
        "shape", "color", "band_type", "band_color", "band_architecture",
        "prong_color", "prong_style", "style", "matched_image_url"
    ]
    for field in required_fields:
        assert field in first_result, f"Missing required field: {field}"
        assert first_result[field] is not None, f"Field {field} is null"


def test_visual_search_by_file_upload():
    """Verify visual search via direct image file upload."""
    # Create a synthetic test image in memory
    img = Image.new("RGB", (224, 224), color=(240, 240, 240))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    response = client.post(
        "/api/v1/search/image",
        files={"file": ("test_ring.jpg", img_byte_arr, "image/jpeg")},
        data={"top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["results"]) > 0
    assert data["results"][0]["similarity_score"] > 0


def test_search_by_sku():
    """Verify SKU search returns visually similar alternative rings."""
    # First get a valid product ID
    products_resp = client.get("/api/v1/products?limit=1")
    pid = products_resp.json()["products"][0]["style_number"]
    
    response = client.get(f"/api/v1/search/by-sku/{pid}?top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["results"]) > 0


def test_extract_metadata_by_file_upload():
    """Verify image metadata extraction via direct file upload POST request."""
    img = Image.new("RGB", (224, 224), color=(220, 200, 180))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/api/v1/extract-metadata",
        files={"file": ("test_ring.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "metadata" in data

    meta = data["metadata"]
    assert "diamond_shape" in meta and meta["diamond_shape"]
    assert "band_metal_tone" in meta and meta["band_metal_tone"]
    assert "band_architecture" in meta and meta["band_architecture"]
    assert "prong_setting" in meta and meta["prong_setting"]
    assert "ring_style" in meta and meta["ring_style"]
    assert "confidence_scores" in meta
    assert "region_detection" in data
    assert "inference_time_ms" in data


def test_extract_metadata_by_url():
    """Verify image metadata extraction via image URL."""
    sample_url = "https://overnightmountings.s3.amazonaws.com/85121.jpg"
    response = client.post(
        "/api/v1/metadata/extract-url",
        json={"image_url": sample_url}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "metadata" in data
    assert data["metadata"]["diamond_shape"]
    assert data["metadata"]["band_metal_tone"]

