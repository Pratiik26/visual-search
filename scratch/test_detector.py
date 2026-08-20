import json
import urllib.request
import ssl
import numpy as np
from PIL import Image, ImageFilter

# Load a sample product image from catalog
with open("backend/data/catalog.json", "r", encoding="utf-8") as f:
    cat = json.load(f)

products = list(cat.get("products", {}).values())
print(f"Loaded {len(products)} products.")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for idx in [0, 5, 12, 25, 50]:
    p = products[idx]
    test_url = p.get("primary_image")
    print(f"\n--- Testing Product {p.get('product_id')} ({p.get('shape')}) ---")
    req = urllib.request.Request(test_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        img = Image.open(resp).convert("RGB")

print(f"Image size: {img.size}")

# Test facet density detector
orig_w, orig_h = img.size
max_dim = 400
scale = max_dim / float(max(orig_w, orig_h))
sw, sh = int(orig_w * scale), int(orig_h * scale)
proc_img = img.resize((sw, sh), Image.Resampling.BILINEAR).convert("RGB")

arr = np.array(proc_img, dtype=np.float32)
r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
lum = 0.299 * r + 0.587 * g + 0.114 * b

gray = Image.fromarray(lum.astype(np.uint8))
edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)

# Local contrast/variance
blur = np.array(gray.filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
local_contrast = np.abs(lum - blur)

# Exclude outer 4% border
b_h = max(2, int(sh * 0.04))
b_w = max(2, int(sw * 0.04))
edges[:b_h, :] = 0
edges[-b_h:, :] = 0
edges[:, :b_w] = 0
edges[:, -b_w:] = 0

# Jewelry saliency
p90_e = np.percentile(edges, 90) or 1.0
p90_c = np.percentile(local_contrast, 90) or 1.0

norm_e = np.clip(edges / p90_e, 0.0, 3.0)
norm_c = np.clip(local_contrast / p90_c, 0.0, 3.0)

sal = (norm_e ** 1.3) * 0.60 + (norm_c ** 1.1) * 0.40
smooth_sal = np.array(Image.fromarray((sal * 60).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32) / 255.0

# Find peak
py, px = np.unravel_index(np.argmax(smooth_sal), smooth_sal.shape)
print(f"Peak at: x={px}/{sw} ({px/sw:.2f}), y={py}/{sh} ({py/sh:.2f}), max_val={np.max(smooth_sal):.2f}")

# Threshold around peak to find ring bounds
sal_thresh = max(0.08, np.max(smooth_sal) * 0.35)
mask = smooth_sal >= sal_thresh

y_pts, x_pts = np.where(mask)
# Keep only cluster near peak
max_r = min(sw, sh) * 0.32
dists = np.sqrt((y_pts - py)**2 + (x_pts - px)**2)
near = dists < max_r
y_ring = y_pts[near]
x_ring = x_pts[near]

min_x, max_x = np.min(x_ring), np.max(x_ring)
min_y, max_y = np.min(y_ring), np.max(y_ring)

bw = max(int(sw * 0.16), max_x - min_x)
bh = max(int(sh * 0.16), max_y - min_y)
pad_x = int(bw * 0.18)
pad_y = int(bh * 0.18)

cx = (min_x + max_x) // 2
cy = (min_y + max_y) // 2

left = max(0, cx - bw//2 - pad_x) / sw
right = min(sw, cx + bw//2 + pad_x) / sw
top = max(0, cy - bh//2 - pad_y) / sh
bottom = min(sh, cy + bh//2 + pad_y) / sh

print(f"Detected Ring Focus Box: rel_left={left:.3f}, rel_top={top:.3f}, rel_right={right:.3f}, rel_bottom={bottom:.3f}")
print(f"Width={right-left:.3f}, Height={bottom-top:.3f}")
