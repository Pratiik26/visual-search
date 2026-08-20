import torch
import open_clip
import numpy as np
from PIL import Image
import time

print("Loading OpenCLIP...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# 1. Contrastive Text Embeddings (Ring vs Bare Hand / Background)
pos_prompts = [
    "a photo of a ring worn on a finger",
    "a gemstone engagement ring on a hand",
    "a jewelry ring with gemstone and band on a finger",
    "a close-up photo of an engagement ring"
]
neg_prompts = [
    "bare skin without any jewelry",
    "empty hand and fingers without any ring",
    "a human palm and knuckles without jewelry",
    "bare fingers without rings"
]

pos_toks = tokenizer(pos_prompts)
neg_toks = tokenizer(neg_prompts)

with torch.inference_mode():
    pos_embs = model.encode_text(pos_toks)
    pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
    pos_vec = pos_embs.mean(dim=0)
    pos_vec = pos_vec / pos_vec.norm()
    
    neg_embs = model.encode_text(neg_toks)
    neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)
    neg_vec = neg_embs.mean(dim=0)
    neg_vec = neg_vec / neg_vec.norm()

print("Contrastive prompt vectors precomputed.")

def detect_ring_contrastive(image: Image.Image):
    orig_w, orig_h = image.size
    std_img = image.resize((384, 384), Image.Resampling.BILINEAR).convert("RGB")
    
    candidate_boxes = []
    
    # Dense spatial sampling across the image at 3 different scales
    # Scale 1: 30% x 30% (tight ring focus)
    for gy in [0.20, 0.35, 0.50, 0.65, 0.80]:
        for gx in [0.20, 0.35, 0.50, 0.65, 0.80]:
            candidate_boxes.append((
                max(0.0, gx - 0.15),
                max(0.0, gy - 0.15),
                min(1.0, gx + 0.15),
                min(1.0, gy + 0.15)
            ))
            
    # Scale 2: 42% x 42% (medium ring focus)
    for gy in [0.25, 0.45, 0.65]:
        for gx in [0.25, 0.45, 0.65, 0.75]:
            candidate_boxes.append((
                max(0.0, gx - 0.21),
                max(0.0, gy - 0.21),
                min(1.0, gx + 0.21),
                min(1.0, gy + 0.21)
            ))
            
    # Center 50% box
    candidate_boxes.append((0.25, 0.25, 0.75, 0.75))

    crops = []
    for l, t, r, b in candidate_boxes:
        c = std_img.crop((int(l * 384), int(t * 384), int(r * 384), int(b * 384)))
        crops.append(preprocess(c))
        
    tensors = torch.stack(crops)
    with torch.inference_mode():
        feats = model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        
        sim_pos = torch.matmul(feats, pos_vec.float()).numpy()
        sim_neg = torch.matmul(feats, neg_vec.float()).numpy()
        
        # Contrastive score: rewards ring presence, heavily penalizes bare skin/knuckles
        scores = sim_pos - (sim_neg * 0.85)

    best_idx = int(np.argmax(scores))
    best_box = candidate_boxes[best_idx]
    best_score = float(scores[best_idx])
    
    print(f"Total candidates: {len(candidate_boxes)}")
    print(f"Best candidate idx={best_idx}, contrastive_score={best_score:.4f}, box={best_box}")
    print(f"sim_pos={sim_pos[best_idx]:.4f}, sim_neg={sim_neg[best_idx]:.4f}")
    return best_box

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
    detect_ring_contrastive(img)
