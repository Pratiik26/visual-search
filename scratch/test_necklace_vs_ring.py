import torch
import open_clip
import numpy as np
from PIL import Image
import json
import urllib.request
import ssl

print("Loading OpenCLIP for Ring vs Necklace detector...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

pos_ring_prompts = [
    "a photo of a diamond engagement ring worn on a finger",
    "a gemstone diamond ring on a hand",
    "a finger wearing a sparkling diamond ring",
    "a jewelry ring on a hand"
]
neg_prompts = [
    "a diamond necklace and neck clavicle",
    "a necklace collar on a woman's chest and neck",
    "bare chest, collarbone, neck and shoulders",
    "a woman's face, cheek and lips",
    "fabric clothing and shirt without ring"
]

with torch.inference_mode():
    p_embs = model.encode_text(tokenizer(pos_ring_prompts))
    p_embs = p_embs / p_embs.norm(dim=-1, keepdim=True)
    ring_pos_vec = (p_embs.mean(dim=0) / p_embs.mean(dim=0).norm()).numpy()

    n_embs = model.encode_text(tokenizer(neg_prompts))
    n_embs = n_embs / n_embs.norm(dim=-1, keepdim=True)
    ring_neg_vec = (n_embs.mean(dim=0) / n_embs.mean(dim=0).norm()).numpy()

print("Ring-specific vectors precomputed.")

def detect_ring_ignore_necklaces(image: Image.Image):
    orig_w, orig_h = image.size

    # Spatial proposals across the entire frame
    candidate_boxes = []
    for sz_w, sz_h in [(0.35, 0.35), (0.45, 0.45), (0.55, 0.55)]:
        for cy in [0.25, 0.40, 0.55, 0.70, 0.85]:
            for cx in [0.25, 0.40, 0.55, 0.70, 0.85]:
                l = max(0.0, min(1.0 - sz_w, cx - sz_w / 2.0))
                t = max(0.0, min(1.0 - sz_h, cy - sz_h / 2.0))
                r = min(1.0, l + sz_w)
                b = min(1.0, t + sz_h)
                candidate_boxes.append((round(l, 3), round(t, 3), round(r, 3), round(b, 3)))

    candidate_boxes = list(set(candidate_boxes))
    print(f"Evaluating {len(candidate_boxes)} proposals...")

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
        scores = (sim_pos - (sim_neg * 0.90)).numpy()

    best_idx = int(np.argmax(scores))
    best_box = candidate_boxes[best_idx]
    print(f"Selected Ring Box: {best_box}, Score: {scores[best_idx]:.4f}, SimPos: {sim_pos[best_idx]:.4f}, SimNeg: {sim_neg[best_idx]:.4f}")
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
    detect_ring_ignore_necklaces(img)
