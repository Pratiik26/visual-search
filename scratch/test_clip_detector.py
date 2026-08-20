import torch
import open_clip
import numpy as np
from PIL import Image
import time
import json
import urllib.request
import ssl

print("Loading OpenCLIP ViT-B-32...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

ring_prompts = [
    "a close-up photo of a diamond engagement ring on a finger",
    "a diamond engagement ring with diamond center stone and metal band",
    "a luxury diamond ring on a hand",
    "a close-up photo of a diamond solitaire ring"
]
tokens = tokenizer(ring_prompts)
with torch.inference_mode():
    text_embs = model.encode_text(tokens)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    ring_vec = text_embs.mean(dim=0)
    ring_vec = ring_vec / ring_vec.norm()

print("Ring prompt vector initialized.")

def detect_ring_only(image: Image.Image):
    t0 = time.time()
    orig_w, orig_h = image.size
    std_img = image.resize((384, 384), Image.Resampling.BILINEAR).convert("RGB")
    
    candidate_boxes = []
    
    # 1. 4x4 grid of 30% x 30% focused crops
    for gy in [0.20, 0.40, 0.60, 0.80]:
        for gx in [0.20, 0.40, 0.60, 0.80]:
            candidate_boxes.append((
                max(0.0, gx - 0.15),
                max(0.0, gy - 0.15),
                min(1.0, gx + 0.15),
                min(1.0, gy + 0.15)
            ))
            
    # 2. 3x3 grid of 42% x 42% focused crops
    for gy in [0.25, 0.50, 0.75]:
        for gx in [0.25, 0.50, 0.75]:
            candidate_boxes.append((
                max(0.0, gx - 0.21),
                max(0.0, gy - 0.21),
                min(1.0, gx + 0.21),
                min(1.0, gy + 0.21)
            ))

    crops = []
    for l, t, r, b in candidate_boxes:
        c = std_img.crop((int(l * 384), int(t * 384), int(r * 384), int(b * 384)))
        crops.append(preprocess(c))
        
    tensors = torch.stack(crops)
    with torch.inference_mode():
        feats = model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        scores = torch.matmul(feats, ring_vec.float()).numpy()
        
    best_idx = int(np.argmax(scores))
    best_box = candidate_boxes[best_idx]
    best_score = float(scores[best_idx])
    
    dt = (time.time() - t0) * 1000
    print(f"Detection took {dt:.1f}ms. Best box: {best_box}, width={best_box[2]-best_box[0]:.2f}, height={best_box[3]-best_box[1]:.2f}, score={best_score:.3f}")
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
    print(f"\nTesting on: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        img = Image.open(resp).convert("RGB")
    detect_ring_only(img)
