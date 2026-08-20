import torch
import open_clip
import numpy as np
from PIL import Image, ImageFilter
import json
import urllib.request
import ssl

print("Loading OpenCLIP for gemstone centering testing...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

pos_prompts = [
    "a close up photo of a diamond engagement ring with center gemstone and setting",
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

def detect_gemstone_centered_box(image: Image.Image):
    orig_w, orig_h = image.size
    sw, sh = 320, 320
    small_img = image.resize((sw, sh), Image.Resampling.BILINEAR).convert("RGB")
    arr = np.array(small_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    gray = Image.fromarray(lum.astype(np.uint8))
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    specular = np.maximum(0.0, lum - 130.0) / 125.0

    border = 10
    edges[:border, :] = 0; edges[-border:, :] = 0; edges[:, :border] = 0; edges[:, -border:] = 0
    specular[:border, :] = 0; specular[-border:, :] = 0; specular[:, :border] = 0; specular[:, -border:] = 0

    max_e = np.max(edges) or 1.0
    facet_map = ((edges / max_e) ** 1.3) * 0.65 + (specular ** 1.2) * 0.35

    # Find highest facet density cluster (the diamond center gemstone)
    thresh = np.percentile(facet_map, 88)
    high_facet = facet_map * (facet_map > thresh)
    total_mass = np.sum(high_facet)

    if total_mass > 0:
        y_indices, x_indices = np.indices((sh, sw))
        gem_cy = (np.sum(y_indices * high_facet) / total_mass) / float(sh)
        gem_cx = (np.sum(x_indices * high_facet) / total_mass) / float(sw)
    else:
        gem_cx, gem_cy = 0.5, 0.5

    print(f"Gemstone center: cx={gem_cx:.3f}, cy={gem_cy:.3f}")

    candidate_boxes_set = set()

    # 1. Proposals centered squarely on the diamond center gemstone
    for sz_w, sz_h in [(0.42, 0.42), (0.52, 0.52), (0.62, 0.62), (0.75, 0.75), (0.55, 0.42), (0.42, 0.55)]:
        l = max(0.0, min(1.0 - sz_w, gem_cx - sz_w / 2.0))
        t = max(0.0, min(1.0 - sz_h, gem_cy - sz_h / 2.0))
        r = min(1.0, l + sz_w)
        b = min(1.0, t + sz_h)
        candidate_boxes_set.add((round(l, 3), round(t, 3), round(r, 3), round(b, 3)))

    # 2. Multi-scale spatial proposals
    for sz in [0.45, 0.60, 0.75]:
        for cy in [0.30, 0.50, 0.70]:
            for cx in [0.30, 0.50, 0.70]:
                l = max(0.0, min(1.0 - sz, cx - sz / 2.0))
                t = max(0.0, min(1.0 - sz, cy - sz / 2.0))
                r = min(1.0, l + sz)
                b = min(1.0, t + sz)
                candidate_boxes_set.add((round(l, 3), round(t, 3), round(r, 3), round(b, 3)))

    candidate_boxes = list(candidate_boxes_set)

    # OpenCLIP Semantic Scoring
    crops = []
    for l, t, r, b in candidate_boxes:
        c = image.crop((int(l * orig_w), int(t * orig_h), int(r * orig_w), int(b * orig_h)))
        crops.append(preprocess(c))

    tensors = torch.stack(crops)
    with torch.inference_mode():
        feats = model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        pos_t = torch.tensor(ring_pos_vec).float()
        neg_t = torch.tensor(ring_neg_vec).float()

        sim_pos = torch.matmul(feats, pos_t)
        sim_neg = torch.matmul(feats, neg_t)
        scores = (sim_pos - (sim_neg * 0.85)).numpy()

    best_idx = int(np.argmax(scores))
    best_box = candidate_boxes[best_idx]
    print(f"Best Box: {best_box}, Score: {scores[best_idx]:.4f}, SimPos: {sim_pos[best_idx]:.4f}")
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
    detect_gemstone_centered_box(img)
