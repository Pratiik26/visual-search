import torch
import open_clip
import numpy as np
from PIL import Image, ImageFilter
import json
import urllib.request
import ssl

print("Loading OpenCLIP for adaptive saliency testing...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

pos_prompts = [
    "a photo of a diamond engagement ring with gemstone and band",
    "a close up photo of a diamond engagement ring",
    "a sparkling diamond solitaire ring on a finger",
    "a diamond ring with precious gemstone"
]
neg_prompts = [
    "manicured fingernails and painted nail tips without ring",
    "bare skin and fingers without any jewelry",
    "a human face and cheek without ring",
    "empty hand without jewelry"
]

with torch.inference_mode():
    p_embs = model.encode_text(tokenizer(pos_prompts))
    p_embs = p_embs / p_embs.norm(dim=-1, keepdim=True)
    ring_pos_vec = (p_embs.mean(dim=0) / p_embs.mean(dim=0).norm()).numpy()

    n_embs = model.encode_text(tokenizer(neg_prompts))
    n_embs = n_embs / n_embs.norm(dim=-1, keepdim=True)
    ring_neg_vec = (n_embs.mean(dim=0) / n_embs.mean(dim=0).norm()).numpy()

def detect_ring_adaptive(image: Image.Image):
    orig_w, orig_h = image.size
    sw, sh = 320, 320
    small_img = image.resize((sw, sh), Image.Resampling.BILINEAR).convert("RGB")
    arr = np.array(small_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    gray = Image.fromarray(lum.astype(np.uint8))
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    specular = np.maximum(0.0, lum - 120.0) / 135.0
    blur_lum = np.array(gray.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32)
    local_var = np.abs(lum - blur_lum) / 255.0

    border = 8
    edges[:border, :] = 0; edges[-border:, :] = 0; edges[:, :border] = 0; edges[:, -border:] = 0
    specular[:border, :] = 0; specular[-border:, :] = 0; specular[:, :border] = 0; specular[:, -border:] = 0
    local_var[:border, :] = 0; local_var[-border:, :] = 0; local_var[:, :border] = 0; local_var[:, -border:] = 0

    max_e = np.max(edges) or 1.0
    saliency = ((edges / max_e) ** 1.2) * 0.45 + (specular ** 1.2) * 0.35 + (local_var ** 1.2) * 0.20

    # Smooth saliency
    smooth = np.array(
        Image.fromarray((saliency * 100).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=6)),
        dtype=np.float32
    )

    thresh = np.percentile(smooth, 82)
    high_sal = smooth >= thresh
    y_pts, x_pts = np.where(high_sal)

    candidate_boxes = []

    if len(y_pts) > 20:
        min_x, max_x = np.percentile(x_pts, 3), np.percentile(x_pts, 97)
        min_y, max_y = np.percentile(y_pts, 3), np.percentile(y_pts, 97)
        
        bw = max(0.40, (max_x - min_x) * 1.30 / float(sw))
        bh = max(0.40, (max_y - min_y) * 1.30 / float(sh))
        cx = ((min_x + max_x) / 2.0) / float(sw)
        cy = ((min_y + max_y) / 2.0) / float(sh)

        l = max(0.0, min(1.0 - bw, cx - bw / 2.0))
        t = max(0.0, min(1.0 - bh, cy - bh / 2.0))
        r = min(1.0, l + bw)
        b = min(1.0, t + bh)
        candidate_boxes.append((round(l, 3), round(t, 3), round(r, 3), round(b, 3)))

    # Add multi-scale proposals
    for sz in [0.48, 0.60, 0.72, 0.85]:
        for cy in [0.35, 0.50, 0.65]:
            for cx in [0.35, 0.50, 0.65]:
                l = max(0.0, min(1.0 - sz, cx - sz / 2.0))
                t = max(0.0, min(1.0 - sz, cy - sz / 2.0))
                r = min(1.0, l + sz)
                b = min(1.0, t + sz)
                candidate_boxes.append((round(l, 3), round(t, 3), round(r, 3), round(b, 3)))

    # Also add standard centered full-view
    candidate_boxes.append((0.05, 0.05, 0.95, 0.95))
    candidate_boxes = list(set(candidate_boxes))

    # Evaluate with OpenCLIP
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
    print(f"Selected Box: {best_box}, Score: {scores[best_idx]:.4f}, SimPos: {sim_pos[best_idx]:.4f}")
    return best_box

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
    detect_ring_adaptive(img)
