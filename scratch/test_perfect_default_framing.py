import torch
import open_clip
import numpy as np
from PIL import Image, ImageFilter
import json
import urllib.request
import ssl

print("Loading OpenCLIP...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

pos_prompts = [
    "a close-up photo of a diamond engagement ring with center gemstone and setting",
    "a diamond ring with center stone and band",
    "a photo of an engagement ring with sparkling diamond",
    "a gemstone solitaire engagement ring"
]
neg_prompts = [
    "plain empty tabletop marble surface without diamond gemstone",
    "manicured fingernails and bare fingers without jewelry",
    "a human face and skin without ring",
    "empty background without diamond"
]

with torch.inference_mode():
    p_embs = model.encode_text(tokenizer(pos_prompts))
    p_embs = p_embs / p_embs.norm(dim=-1, keepdim=True)
    ring_pos_vec = (p_embs.mean(dim=0) / p_embs.mean(dim=0).norm()).numpy()

    n_embs = model.encode_text(tokenizer(neg_prompts))
    n_embs = n_embs / n_embs.norm(dim=-1, keepdim=True)
    ring_neg_vec = (n_embs.mean(dim=0) / n_embs.mean(dim=0).norm()).numpy()

def detect_perfect_box(image: Image.Image):
    orig_w, orig_h = image.size
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

    # Find diamond gemstone center of mass
    thresh = np.percentile(facet_map, 88)
    high_facet = facet_map * (facet_map > thresh)
    total_mass = np.sum(high_facet)

    if total_mass > 0:
        y_indices, x_indices = np.indices((sh, sw))
        gem_cy = (np.sum(y_indices * high_facet) / total_mass) / float(sh)
        gem_cx = (np.sum(x_indices * high_facet) / total_mass) / float(sw)
    else:
        gem_cx, gem_cy = 0.5, 0.5

    # Check if the photo is a close-up ring photo (gemstone near center) vs off-center hand photo
    dist_from_center = np.sqrt((gem_cx - 0.5) ** 2 + (gem_cy - 0.5) ** 2)
    
    if dist_from_center < 0.22:
        # Centered ring / close-up: Frame the full ring with comfortable generous margins
        best_box = (0.04, 0.04, 0.96, 0.96)
    else:
        # Off-center hand photo: Frame the ring with generous 60% x 60% margin around gemstone
        sz = 0.58
        l = max(0.0, min(1.0 - sz, gem_cx - sz / 2.0))
        t = max(0.0, min(1.0 - sz, gem_cy - sz / 2.0))
        r = min(1.0, l + sz)
        b = min(1.0, t + sz)
        best_box = (round(l, 3), round(t, 3), round(r, 3), round(b, 3))

    print(f"Gemstone cx={gem_cx:.3f}, cy={gem_cy:.3f}, dist={dist_from_center:.3f} -> Default Box: {best_box}")
    return best_box

# Test on sample images
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
    detect_perfect_box(img)
