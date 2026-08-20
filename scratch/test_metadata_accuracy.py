import torch
import open_clip
import numpy as np
from PIL import Image
import json
import urllib.request
import ssl

print("Loading OpenCLIP...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Metal prompts
metal_descriptions = {
    "14K Yellow Gold": [
        "a yellow gold engagement ring with yellow gold band",
        "a 14k yellow gold diamond ring",
        "a warm yellow gold precious metal ring"
    ],
    "14K White Gold": [
        "a white gold platinum diamond engagement ring with silvery white metal band",
        "a white gold diamond solitaire ring",
        "a platinum white gold diamond ring"
    ],
    "14K Rose Gold": [
        "a rose gold engagement ring with copper pink metal band",
        "a pink rose gold diamond ring",
        "a 14k rose gold ring"
    ]
}

metal_vecs = {}
for m_name, prompts in metal_descriptions.items():
    toks = tokenizer(prompts)
    with torch.inference_mode():
        embs = model.encode_text(toks)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        mean_emb = embs.mean(dim=0)
        metal_vecs[m_name] = (mean_emb / mean_emb.norm()).numpy()

print("Metal prompt vectors initialized.")

# Style prompts
style_descriptions = {
    "Solitaire": ["a solitaire diamond engagement ring with single center stone"],
    "Halo": ["a halo diamond engagement ring with small diamonds framing center stone"],
    "Three-Stone": ["a three-stone diamond engagement ring with three diamonds"],
    "Modern Classic": ["a modern classic diamond engagement ring with clean elegant band"],
    "Vintage": ["a vintage art-deco antique diamond engagement ring with milgrain"]
}
style_vecs = {}
for s_name, prompts in style_descriptions.items():
    toks = tokenizer(prompts)
    with torch.inference_mode():
        embs = model.encode_text(toks)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        mean_emb = embs.mean(dim=0)
        style_vecs[s_name] = (mean_emb / mean_emb.norm()).numpy()

print("Style prompt vectors initialized.")

def classify_metadata_clip(image: Image.Image):
    t = preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        feat = model.encode_image(t)
        feat = (feat / feat.norm(dim=-1, keepdim=True)).numpy()[0]

    # Metal
    m_scores = {m: float(np.dot(feat, vec)) for m, vec in metal_vecs.items()}
    best_metal = sorted(m_scores.items(), key=lambda x: -x[1])[0][0]

    # Style
    st_scores = {s: float(np.dot(feat, vec)) for s, vec in style_vecs.items()}
    best_style = sorted(st_scores.items(), key=lambda x: -x[1])[0][0]

    print(f"Predicted Metal: {best_metal}, Scores: {m_scores}")
    print(f"Predicted Style: {best_style}, Scores: {st_scores}")
    return best_metal, best_style

# Test on sample catalog images
with open("backend/data/catalog.json", "r", encoding="utf-8") as f:
    cat = json.load(f)

products = list(cat.get("products", {}).values())
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for i in [0, 5, 10]:
    p = products[i]
    url = p.get("primary_image")
    print(f"\n--- Testing on Product {p.get('product_id')} ({p.get('shape')}, {p.get('metal')}) ---")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        img = Image.open(resp).convert("RGB")
    classify_metadata_clip(img)
