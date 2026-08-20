import torch
import open_clip
import numpy as np
from PIL import Image
import json
import urllib.request
import ssl

print("Loading OpenCLIP for proposal evaluation...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

pos_prompts = [
    "a photo of a diamond engagement ring worn on a finger",
    "a gemstone diamond ring on a hand",
    "a close up of a diamond ring on a finger"
]
neg_prompts = [
    "manicured fingernails and nail polish tips without ring",
    "bare skin and fingers without any jewelry",
    "a human face and cheek without ring",
    "empty hand without jewelry"
]

pos_tokens = tokenizer(pos_prompts)
neg_tokens = tokenizer(neg_prompts)

with torch.inference_mode():
    p_embs = model.encode_text(pos_tokens)
    p_embs = p_embs / p_embs.norm(dim=-1, keepdim=True)
    ring_pos_vec = (p_embs.mean(dim=0) / p_embs.mean(dim=0).norm()).numpy()

    n_embs = model.encode_text(neg_tokens)
    n_embs = n_embs / n_embs.norm(dim=-1, keepdim=True)
    ring_neg_vec = (n_embs.mean(dim=0) / n_embs.mean(dim=0).norm()).numpy()

print("Prompt features ready.")

def generate_spatial_proposals():
    boxes = []
    # Grid of candidate boxes across image
    for sz in [0.36, 0.48]:
        for cy in [0.25, 0.40, 0.55, 0.70]:
            for cx in [0.25, 0.40, 0.55, 0.70]:
                l = max(0.0, min(1.0 - sz, cx - sz / 2.0))
                t = max(0.0, min(1.0 - sz, cy - sz / 2.0))
                r = min(1.0, l + sz)
                b = min(1.0, t + sz)
                boxes.append((round(l, 3), round(t, 3), round(r, 3), round(b, 3)))
    return list(set(boxes))

boxes = generate_spatial_proposals()
print(f"Generated {len(boxes)} spatial candidate proposals.")

def detect_best_ring_box(image: Image.Image):
    w, h = image.size
    crops = []
    for l, t, r, b in boxes:
        c = image.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
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
    best_box = boxes[best_idx]
    print(f"Best Ring Box: {best_box}, Score: {scores[best_idx]:.4f}, SimPos: {sim_pos[best_idx]:.4f}")
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
    detect_best_ring_box(img)
