"""
OpenCLIP Visual Indexer for Diamond Ring Catalog
Generates high-dimensional ViT-B-32 visual embeddings for catalog images.
"""

import os
import io
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch
import open_clip

from backend.config import (
    CATALOG_PATH,
    CLIP_EMBEDDINGS_PATH,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_clip_index():
    logger.info(f"Initializing OpenCLIP {CLIP_MODEL_NAME} visual backbone...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    model = model.to(device)
    model.eval()

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    products = catalog.get("products", {})
    logger.info(f"Loaded {len(products)} products from catalog.")

    images_to_index = []
    seen = set()
    for pid, p in products.items():
        primary = p.get("primary_image")
        alt = next((img for img in p.get("all_images", []) if ".alt." in img or ".alt1." in img), None)

        for img_url in [primary, alt]:
            if img_url and img_url not in seen:
                seen.add(img_url)

                metal = "White Gold"
                if ".alt1." in img_url:
                    metal = "Rose Gold"
                elif ".alt." in img_url:
                    metal = "Yellow Gold"
                elif "platinum" in img_url.lower():
                    metal = "Platinum"

                images_to_index.append({
                    "product_id": pid,
                    "image_url": img_url,
                    "title": p.get("title", ""),
                    "shape": p.get("shape", "Round"),
                    "band_color": metal,
                    "band_type": p.get("band_type", "Plain Solitaire Band"),
                    "band_architecture": p.get("band_architecture", "Classic Straight Shank"),
                    "prong_style": p.get("prong_style", "Classic 4-Prong Setting"),
                    "style": p.get("style", "Solitaire")
                })

    logger.info(f"Downloading {len(images_to_index)} representative images in parallel...")

    def fetch(item):
        try:
            req = urllib.request.Request(item["image_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                return item, img
        except Exception:
            return item, None

    downloaded = []
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(fetch, itm) for itm in images_to_index]
        for fut in as_completed(futures):
            itm, img = fut.result()
            if img is not None:
                downloaded.append((itm, img))

    logger.info(f"Successfully downloaded {len(downloaded)} images. Computing OpenCLIP embeddings...")

    batch_size = 64
    all_embeddings = []
    valid_items = []

    for i in range(0, len(downloaded), batch_size):
        batch = downloaded[i:i + batch_size]
        items = [x[0] for x in batch]
        tensors = torch.stack([preprocess(x[1]) for x in batch]).to(device)

        with torch.no_grad():
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy())
            valid_items.extend(items)

    emb_matrix = np.vstack(all_embeddings).astype(np.float32)
    logger.info(f"OpenCLIP index complete. Matrix shape: {emb_matrix.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(CLIP_EMBEDDINGS_PATH)), exist_ok=True)
    np.savez_compressed(
        CLIP_EMBEDDINGS_PATH,
        embeddings=emb_matrix,
        items=np.array(valid_items, dtype=object)
    )
    logger.info(f"Saved OpenCLIP embeddings to {CLIP_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_clip_index()
