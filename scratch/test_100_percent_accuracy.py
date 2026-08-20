import torch
import open_clip
import numpy as np
from PIL import Image, ImageFilter
import json
import urllib.request
import ssl

print("Loading OpenCLIP for 100% accuracy pipeline test...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Shape descriptions
shape_descriptions = {
    "Emerald": [
        "an emerald cut diamond engagement ring with rectangular step-cut facets and cut corners",
        "a rectangular emerald cut diamond ring with hall-of-mirrors step facets",
        "an emerald step cut diamond ring"
    ],
    "Round": [
        "a round brilliant cut diamond engagement ring with circular facets",
        "a round shape diamond solitaire ring",
        "a circular round brilliant diamond ring"
    ],
    "Oval": [
        "an oval cut diamond engagement ring with elongated elliptical curved facets",
        "an elongated oval shape diamond ring",
        "an oval brilliant diamond ring"
    ],
    "Cushion": [
        "a cushion cut diamond engagement ring with pillow-shaped rounded rectangular corners",
        "a square cushion pillow shape diamond ring",
        "a cushion modified brilliant diamond ring"
    ],
    "Princess": [
        "a princess cut diamond engagement ring with sharp square 90 degree corners",
        "a square princess cut diamond ring",
        "a princess cut diamond ring"
    ],
    "Radiant": [
        "a radiant cut diamond engagement ring with rectangular shape and brilliant crushed-ice sparkle facets",
        "a radiant cut diamond ring"
    ],
    "Pear": [
        "a pear shape teardrop cut diamond engagement ring with one rounded end and one pointed tip",
        "a teardrop pear cut diamond ring"
    ],
    "Marquise": [
        "a marquise cut diamond engagement ring with elongated football eye shape and two sharp pointed tips",
        "a marquise shape diamond ring"
    ],
    "Asscher": [
        "an asscher cut diamond engagement ring with square step-cut facets and windmills",
        "a square asscher step cut diamond ring"
    ],
    "Heart": [
        "a heart shape cut diamond engagement ring with romantic cleft and pointed tip",
        "a heart cut diamond ring"
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

def extract_feat(img: Image.Image) -> np.ndarray:
    t = preprocess(img).unsqueeze(0)
    with torch.inference_mode():
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.numpy()[0]

def test_pipeline_on_image(image: Image.Image, expected_shape: str = None):
    w, h = image.size
    sw, sh = 320, 320
    small_img = image.resize((sw, sh), Image.Resampling.BILINEAR).convert("RGB")
    arr = np.array(small_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    gray = Image.fromarray(lum.astype(np.uint8))
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    specular = np.maximum(0.0, lum - 130.0) / 125.0

    border = 8
    edges[:border, :] = 0; edges[-border:, :] = 0; edges[:, :border] = 0; edges[:, -border:] = 0
    specular[:border, :] = 0; specular[-border:, :] = 0; specular[:, :border] = 0; specular[:, -border:] = 0

    max_e = np.max(edges) or 1.0
    facet_map = ((edges / max_e) ** 1.3) * 0.65 + (specular ** 1.2) * 0.35

    thresh = np.percentile(facet_map, 88)
    high_facet = facet_map * (facet_map > thresh)
    total_mass = np.sum(high_facet)

    if total_mass > 0:
        y_indices, x_indices = np.indices((sh, sw))
        gem_cy = (np.sum(y_indices * high_facet) / total_mass) / float(sh)
        gem_cx = (np.sum(x_indices * high_facet) / total_mass) / float(sw)
    else:
        gem_cx, gem_cy = 0.5, 0.5

    # 1. Global image vector
    q_global = extract_feat(image)

    # 2. Pure diamond stone crop centered directly on (gem_cx, gem_cy)
    sz_stone = 0.42
    sl = max(0, int((gem_cx - sz_stone / 2.0) * w))
    st = max(0, int((gem_cy - sz_stone / 2.0) * h))
    sr = min(w, max(sl + 10, int((gem_cx + sz_stone / 2.0) * w)))
    sb = min(h, max(st + 10, int((gem_cy + sz_stone / 2.0) * h)))
    stone_crop = image.crop((sl, st, sr, sb))
    f_stone = extract_feat(stone_crop)

    # Multi-modal shape classification
    scores = {}
    for s_name, t_vec in shape_vecs.items():
        sim_g = float(np.dot(q_global, t_vec))
        sim_s = float(np.dot(f_stone, t_vec))
        scores[s_name] = sim_s * 0.65 + sim_g * 0.35

    best_shape = sorted(scores.items(), key=lambda x: -x[1])[0][0]
    print(f"Predicted Shape: {best_shape} (Expected: {expected_shape})")
    top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
    print(f"Top 3: {top3}")
    return best_shape

# Test on sample images from catalog
with open("backend/data/catalog.json", "r", encoding="utf-8") as f:
    cat = json.load(f)

products = list(cat.get("products", {}).values())
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

shapes_tested = set()
for p in products:
    s = p.get("shape")
    if s and s not in shapes_tested and len(shapes_tested) < 6:
        shapes_tested.add(s)
        url = p.get("primary_image")
        print(f"\n--- Testing on {s} ring ({url}) ---")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                img = Image.open(resp).convert("RGB")
            test_pipeline_on_image(img, s)
        except Exception as e:
            print(f"Error: {e}")
