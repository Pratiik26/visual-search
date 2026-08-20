import numpy as np
from PIL import Image, ImageFilter
import open_clip
import torch
import time
import json
import urllib.request
import ssl

def get_precise_ring_box(image: Image.Image):
    orig_w, orig_h = image.size
    sw, sh = 320, 320
    small_img = image.resize((sw, sh), Image.Resampling.BILINEAR).convert("RGB")
    arr = np.array(small_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Edge map (captures sharp diamond facet boundaries and prongs)
    gray = Image.fromarray(lum.astype(np.uint8))
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)

    # Specular reflection (diamonds have intense bright reflection vs skin)
    specular = np.maximum(0.0, lum - 135.0) / 120.0

    # Suppress outer 3% image border
    border = 10
    edges[:border, :] = 0; edges[-border:, :] = 0; edges[:, :border] = 0; edges[:, -border:] = 0
    specular[:border, :] = 0; specular[-border:, :] = 0; specular[:, :border] = 0; specular[:, -border:] = 0

    # High-intensity facet sparkle map
    max_e = np.max(edges) or 1.0
    facet_map = ((edges / max_e) ** 1.3) * 0.65 + (specular ** 1.1) * 0.35

    # Gaussian smoothing to find diamond center
    smooth = np.array(
        Image.fromarray((facet_map * 100).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4)),
        dtype=np.float32
    )

    # Find the diamond facet sparkle centroid
    py, px = np.unravel_index(np.argmax(smooth), smooth.shape)
    peak_cx = px / float(sw)
    peak_cy = py / float(sh)
    print(f"Diamond Facet Peak: x={peak_cx:.3f}, y={peak_cy:.3f}")

    # Connected active contour of the diamond + setting
    sal_max = np.max(smooth)
    thresh = max(8.0, sal_max * 0.32)
    active_mask = smooth >= thresh

    y_pts, x_pts = np.where(active_mask)
    max_radius = min(sw, sh) * 0.30
    dists = np.sqrt((x_pts - px) ** 2 + (y_pts - py) ** 2)
    near = dists < max_radius

    if np.sum(near) > 15:
        y_j = y_pts[near]
        x_j = x_pts[near]
        min_x, max_x = np.min(x_j), np.max(x_j)
        min_y, max_y = np.min(y_j), np.max(y_j)

        w_j = max(int(sw * 0.18), max_x - min_x)
        h_j = max(int(sh * 0.18), max_y - min_y)
        pad_x = int(w_j * 0.14)
        pad_y = int(h_j * 0.14)

        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2

        left = max(0.0, (cx - w_j // 2 - pad_x) / float(sw))
        right = min(1.0, (cx + w_j // 2 + pad_x) / float(sw))
        top = max(0.0, (cy - h_j // 2 - pad_y) / float(sh))
        bottom = min(1.0, (cy + h_j // 2 + pad_y) / float(sh))
    else:
        # Fallback centered around peak
        box_sz = 0.32
        left = max(0.0, peak_cx - box_sz / 2)
        right = min(1.0, peak_cx + box_sz / 2)
        top = max(0.0, peak_cy - box_sz / 2)
        bottom = min(1.0, peak_cy + box_sz / 2)

    print(f"Precise Box: left={left:.3f}, top={top:.3f}, right={right:.3f}, bottom={bottom:.3f}, width={right-left:.3f}, height={bottom-top:.3f}")
    return left, top, right, bottom

# Test on sample catalog images
with open("backend/data/catalog.json", "r", encoding="utf-8") as f:
    cat = json.load(f)

products = list(cat.get("products", {}).values())
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for i in [0, 5, 10]:
    url = products[i].get("primary_image")
    print(f"\n--- Testing on: {url} ---")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        img = Image.open(resp).convert("RGB")
    get_precise_ring_box(img)
