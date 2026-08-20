import numpy as np
from PIL import Image, ImageFilter
import open_clip
import torch
import time

print("Loading OpenCLIP...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Shape prompts to test classification on the full diamond vs partial diamond
shape_descriptions = {
    "Emerald": [
        "an emerald cut diamond engagement ring with rectangular step-cut facets and cut corners",
        "a rectangular emerald shape diamond ring",
        "an emerald step cut diamond ring"
    ],
    "Radiant": [
        "a radiant cut diamond engagement ring with rectangular shape and brilliant sparkle facets",
        "a radiant cut diamond ring"
    ]
}
shape_vecs = {}
for s_name, prompts in shape_descriptions.items():
    toks = tokenizer(prompts)
    with torch.inference_mode():
        embs = model.encode_text(toks)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        mean_emb = embs.mean(dim=0)
        shape_vecs[s_name] = (mean_emb / mean_emb.norm()).numpy()

def compute_diamond_centered_box(image: Image.Image):
    orig_w, orig_h = image.size
    sw, sh = 300, 300
    small_img = image.resize((sw, sh), Image.Resampling.BILINEAR).convert("RGB")
    arr = np.array(small_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    gray = Image.fromarray(lum.astype(np.uint8))
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    specular = np.maximum(0.0, lum - 130.0) / 125.0

    # Suppress borders
    border = 10
    edges[:border, :] = 0; edges[-border:, :] = 0; edges[:, :border] = 0; edges[:, -border:] = 0
    specular[:border, :] = 0; specular[-border:, :] = 0; specular[:, :border] = 0; specular[:, -border:] = 0

    # Intense facet density
    max_e = np.max(edges) or 1.0
    facet_map = ((edges / max_e) ** 1.3) * 0.70 + (specular ** 1.2) * 0.30

    # Find center of diamond mass (weighted centroid)
    thresh = np.percentile(facet_map, 88)
    high_facet = facet_map * (facet_map > thresh)

    total_mass = np.sum(high_facet)
    if total_mass > 0:
        y_indices, x_indices = np.indices((sh, sw))
        cy = np.sum(y_indices * high_facet) / total_mass
        cx = np.sum(x_indices * high_facet) / total_mass
    else:
        cx, cy = sw / 2.0, sh / 2.0

    peak_cx = cx / float(sw)
    peak_cy = cy / float(sh)

    # Box dimension (48% of image centered exactly on the diamond)
    box_w = 0.46
    box_h = 0.46

    rel_l = max(0.0, min(1.0 - box_w, peak_cx - box_w / 2.0))
    rel_t = max(0.0, min(1.0 - box_h, peak_cy - box_h / 2.0))
    rel_r = min(1.0, rel_l + box_w)
    rel_b = min(1.0, rel_t + box_h)

    print(f"Diamond Centroid: cx={peak_cx:.3f}, cy={peak_cy:.3f}")
    print(f"Diamond Focused Box: left={rel_l:.3f}, top={rel_t:.3f}, right={rel_r:.3f}, bottom={rel_b:.3f}")
    return rel_l, rel_t, rel_r, rel_b

import json, urllib.request, ssl

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
    compute_diamond_centered_box(img)
