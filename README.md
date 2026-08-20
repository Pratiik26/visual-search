# 💎 AURA DIAMONDS — OpenCLIP Visual Search Engine & REST API

A production-grade AI visual search engine and REST API built for diamond engagement rings. Upload any photo, click on interactive proposal hotspots, or pass an image URL of a diamond ring: the system extracts high-dimensional OpenCLIP ViT-B-32 visual feature vectors and colorimetric distributions to find matching rings from our dataset of **1,887 ring models** and **7,500+ views**, returning rich structured metadata for every result.

---

## 🌟 Key Features

- **OpenCLIP ViT-B-32 Multi-Modal Vision Backbone**: Zero-shot image and text feature embeddings calibrated for jewelry geometry, center diamond cut silhouettes, prong architectures, and band contours.
- **Nyris-Style AI Region Proposal**: Instant visual saliency edge and gradient detection identifying diamond ring bounding box, center gemstone head ROI, and metal shank ROI.
- **Metal Hue Colorimetry Profiling**: Analyzes HSV and LAB chromaticity distributions to identify White Gold, Yellow Gold, Rose Gold, Platinum, and Two-Tone alloys.
- **Rich Structured Metadata**:
  - **`shape`**: Round, Oval, Cushion, Emerald, Princess, Radiant, Pear, Marquise, Asscher, Heart, Trillion
  - **`band_color` & `color`**: White Gold, Yellow Gold, Rose Gold, Platinum, Two-Tone
  - **`band_type`**: Plain Solitaire Band, Single Row Pave, Multirow, Channel Set, Side Stone Accented, Bypass, Split Shank, Three-Stone
  - **`band_architecture`**: Classic Straight Shank, Twisted Shank, Split Shank, Bypass Shank, Tapered Shank, Cathedral Shank, Multirow Band
  - **`prong_style` & `prong_color`**: Classic 4-Prong Setting, 6-Prong Crown Setting, Peg Head Setting, Bezel Set, Halo with Micro-Prongs, Hidden Halo Basket, 3-Stone Multi-Prong
  - **`style`**: Solitaire, Halo, Hidden Halo, Vintage, Nature Inspired, Three Stone, Side Stone, Bezel, Toi-et-Moi, Pave, Modern Classic
  - **`stone_breakdown`**: Center diamond dimensions, carats, side stones quantity, and total carat weight.
  - **`images & 360° videos`**: Front view, side view, 3/4 set view, alternate metal color variants, and high-definition 360° MP4 render videos.
- **Sub-100ms Query Latency**: In-memory vector similarity with LRU feature cache and normalized cosine matrix dot products.
- **Luxury Haute Joaillerie Web Interface**: Obsidian & Champagne Gold luxury UI with drag-and-drop upload, adjustable target crop frames, dynamic attribute filters, 360° video modal player, and live JSON inspector.

---

## 📁 Clean Project Structure

```
imagesearch/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # FastAPI endpoints (Search, Regions, Products, Taxonomy, Health)
│   │   └── schemas.py         # Pydantic validation request & response schemas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── cache.py           # In-memory fast LRU feature cache
│   │   ├── detector.py        # Nyris-style AI region & ring bounding box detector
│   │   └── engine.py          # OpenCLIP ViT-B-32 multimodal search engine
│   ├── pipelines/             # Offline data pipelines & indexing tools
│   │   ├── __init__.py
│   │   ├── clip_indexer.py    # Generates OpenCLIP embeddings from catalog.json
│   │   ├── dataset_indexer.py # Normalizes raw feeds into unified catalog.json
│   │   └── scraper.py         # Scrapes OvernightMountings engagement rings catalog
│   ├── data/
│   │   ├── catalog.json       # Cleaned 1,887 ring catalog dataset
│   │   ├── clip_embeddings.npz# OpenCLIP 512-dim normalized feature vectors
│   │   └── scraped_products.json
│   ├── config.py              # Centralized environment, paths & model settings
│   ├── main.py                # FastAPI application setup, middleware & static mounts
│   └── __init__.py
├── frontend/
│   ├── app.js                 # Interactive target frame, hotspot pins & video viewer
│   ├── index.html             # Luxury Haute Joaillerie UI layout
│   └── style.css              # Obsidian & Champagne Gold responsive stylesheet
├── tests/
│   ├── __init__.py
│   └── test_api.py            # Automated pytest test suite
├── .gitignore                 # Production ignore rules
├── requirements.txt           # Pinned project dependencies
├── run.py                     # Unified server launcher with auto-port freeing
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
Ensure Python 3.10+ is installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Application Server
```bash
python run.py
```
- **Web Interface**: [http://127.0.0.1:8080/](http://127.0.0.1:8080/)
- **Interactive Swagger Docs**: [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)
- **ReDoc API Reference**: [http://127.0.0.1:8080/redoc](http://127.0.0.1:8080/redoc)

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/extract-metadata` | **Upload image & extract rich ring metadata (Shape, Metal, Shank, Prong, Style)** |
| `POST` | `/api/v1/metadata/extract-url` | Extract ring metadata via remote image URL or base64 |
| `POST` | `/api/v1/search/image` | Multi-part file visual search with catalog recommendations |
| `POST` | `/aiSearchProducts` | Alias for visual search |
| `POST` | `/api/v1/search/url` | Remote image URL visual search |
| `POST` | `/find/v2/regions` | AI ring region proposal & bounding box detection |
| `GET` | `/api/v1/search/by-sku/{sku}` | Visual similarity search by SKU / product ID |
| `GET` | `/api/v1/products` | Paginated catalog browsing with filtering |
| `GET` | `/api/v1/metadata/attributes` | Taxonomy attributes (shapes, metals, styles, settings) |
| `GET` | `/api/v1/health` | System health, model info, and catalog stats |

> 📖 **Frontend & Website Integration**: See [API_INTEGRATION_GUIDE.md](file:///c:/Users/pratik/OneDrive/Documents/imagesearch/API_INTEGRATION_GUIDE.md) for full JavaScript, React, Node.js, Python, and cURL integration snippets.

---

## 🧪 Running Automated Tests

Run the test suite with pytest:
```bash
python -m pytest -v tests/test_api.py
```

