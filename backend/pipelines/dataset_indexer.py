"""
Unified Dataset Indexer & Metadata Normalizer
Parses and normalizes scraped OvernightMountings API data and catalog feeds into:
- shape, color, band_type, band_color, band_architecture, prong_style, style, stone breakdown, and image galleries.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional

from backend.config import DATA_DIR, CATALOG_PATH, SCRAPED_PRODUCTS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_SHAPES = [
    "Round", "Oval", "Cushion", "Emerald", "Princess",
    "Radiant", "Pear", "Marquise", "Asscher", "Heart", "Trillion"
]

METAL_COLOR_MAP = {
    "9K White": {"category": "White Gold", "purity": "9K", "color": "White Gold"},
    "14K White": {"category": "White Gold", "purity": "14K", "color": "White Gold"},
    "18K White": {"category": "White Gold", "purity": "18K", "color": "White Gold"},
    "9K Yellow": {"category": "Yellow Gold", "purity": "9K", "color": "Yellow Gold"},
    "14K Yellow": {"category": "Yellow Gold", "purity": "14K", "color": "Yellow Gold"},
    "18K Yellow": {"category": "Yellow Gold", "purity": "18K", "color": "Yellow Gold"},
    "9K Rose": {"category": "Rose Gold", "purity": "9K", "color": "Rose Gold"},
    "14K Rose": {"category": "Rose Gold", "purity": "14K", "color": "Rose Gold"},
    "18K Rose": {"category": "Rose Gold", "purity": "18K", "color": "Rose Gold"},
    "Platinum": {"category": "Platinum", "purity": "950 Platinum", "color": "Platinum"},
    "White": {"category": "White Gold", "purity": "14K", "color": "White Gold"},
    "Yellow": {"category": "Yellow Gold", "purity": "14K", "color": "Yellow Gold"},
    "Rose": {"category": "Rose Gold", "purity": "14K", "color": "Rose Gold"},
    "White/Yellow": {"category": "Two-Tone", "purity": "14K Two-Tone", "color": "White & Yellow Gold Two-Tone"},
    "White/Rose": {"category": "Two-Tone", "purity": "14K Two-Tone", "color": "White & Rose Gold Two-Tone"}
}


def clean_html(raw_html: str) -> str:
    """Strips HTML tags and normalizes whitespace."""
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(raw_html))
    return re.sub(r"\s+", " ", text).strip()


def extract_band_architecture(title: str, description: str, collections: List[str] = None) -> str:
    """Classifies the shank/band architecture."""
    coll_str = " ".join(collections or []).lower()
    combined = f"{title} {description} {coll_str}".lower()

    if "twisted" in combined or "twist" in combined or "interwoven" in combined:
        return "Twisted Shank"
    elif "split" in combined:
        return "Split Shank"
    elif "bypass" in combined or "cross" in combined:
        return "Bypass Shank"
    elif "tapper" in combined or "taper" in combined:
        return "Tapered Shank"
    elif "multirow" in combined or "multi-row" in combined or "multiple row" in combined:
        return "Multirow Band"
    elif "cathedral" in combined:
        return "Cathedral Shank"
    elif "knife" in combined:
        return "Knife Edge Shank"
    elif "vintage" in combined or "antic" in combined or "filigree" in combined:
        return "Vintage Engraved Shank"
    else:
        return "Classic Straight Shank"


def extract_band_type(title: str, description: str, collections: List[str] = None, side_stones_qty: int = 0) -> str:
    """Classifies band styling."""
    coll_str = " ".join(collections or []).lower()
    combined = f"{title} {description} {coll_str}".lower()

    if "3/4 pave" in combined:
        return "3/4 Pave Diamond Band"
    elif "1/2 pave" in combined:
        return "1/2 Pave Diamond Band"
    elif "single row" in combined:
        return "Single Row Pave Band"
    elif "multirow" in combined or "multi row" in combined:
        return "Multirow Diamond Band"
    elif "pave" in combined or "pav" in combined:
        return "Pave Diamond Band"
    elif "channel" in combined:
        return "Channel Set Band"
    elif "side stone" in combined or "side diamond" in combined or (0 < side_stones_qty <= 10):
        return "Side Stone Accented Band"
    elif "three stone" in combined or "three-stone" in combined:
        return "Three-Stone Band"
    elif "bezel" in combined:
        return "Bezel Accented Band"
    elif "nature" in combined:
        return "Nature-Inspired Band"
    elif "solitaire" in combined or side_stones_qty == 0:
        return "Plain Solitaire Band"
    else:
        return "Classic Accented Band"


def extract_prong_style(title: str, description: str, peg_head: bool = False, collections: List[str] = None, metal_category: str = "White Gold") -> Dict[str, str]:
    """Classifies prong setting style and prong metal color."""
    coll_str = " ".join(collections or []).lower()
    combined = f"{title} {description} {coll_str}".lower()

    if "bezel" in combined:
        if "half bezel" in combined or "semi bezel" in combined:
            setting = "Semi-Bezel Setting"
        else:
            setting = "Full Bezel Setting"
    elif "hidden halo" in combined or "basket" in combined:
        setting = "Hidden Halo with 4-Claw Prongs"
    elif "halo" in combined:
        setting = "Halo with Micro-Prong Basket"
    elif "three stone" in combined or "three-stone" in combined:
        setting = "3-Stone Multi-Prong Setting"
    elif "toi-et-moi" in combined or "toi et moi" in combined:
        setting = "Toi-et-Moi Dual Setting"
    elif "6-prong" in combined or "six prong" in combined:
        setting = "6-Prong Crown Setting"
    elif peg_head or "peg head" in combined:
        setting = "Peg Head 4-Prong Setting"
    elif "claw" in combined:
        setting = "Petite Claw Prongs"
    else:
        setting = "Classic 4-Prong Setting"

    prong_color = metal_category
    if "white/yellow" in combined or "two tone" in combined or "two-tone" in combined:
        prong_color = "White Gold (Two-Tone Head)"
    elif "white/rose" in combined:
        prong_color = "White Gold (Two-Tone Head)"

    return {
        "prong_style": setting,
        "prong_color": prong_color,
        "prong_summary": f"{setting} in {prong_color}"
    }


def extract_style(title: str, description: str, collections: List[str] = None) -> str:
    """Extracts overall ring style category."""
    coll_str = " ".join(collections or []).lower()
    combined = f"{title} {description} {coll_str}".lower()

    if "solitaire" in combined:
        return "Solitaire"
    elif "hidden halo" in combined:
        return "Hidden Halo"
    elif "halo" in combined:
        return "Halo"
    elif "three stone" in combined or "three-stone" in combined:
        return "Three Stone"
    elif "vintage" in combined or "antic" in combined:
        return "Vintage"
    elif "nature" in combined:
        return "Nature Inspired"
    elif "bezel" in combined:
        return "Bezel"
    elif "side stone" in combined or "side diamond" in combined:
        return "Side Stone"
    elif "pave" in combined or "single row" in combined:
        return "Pave"
    elif "toi-et-moi" in combined:
        return "Toi-et-Moi"
    elif "unique" in combined:
        return "Unique"
    return "Modern Classic"


def parse_scraped_api_products(scraped_file: str) -> List[Dict[str, Any]]:
    """Parses products scraped from OvernightMountings API into normalized ring records."""
    if not os.path.exists(scraped_file):
        logger.warning(f"Scraped file not found: {scraped_file}")
        return []

    with open(scraped_file, "r", encoding="utf-8") as f:
        raw_products = json.load(f)

    logger.info(f"Normalizing {len(raw_products)} products from OvernightMountings API...")
    normalized_items = []

    for p in raw_products:
        style_no = p.get("style_number", "")
        title = p.get("title", "")
        description = p.get("description", "")
        collections = p.get("collections", [])
        peg_head = p.get("peg_head_setting", False)

        images = p.get("images", [])
        videos = p.get("videos", [])

        stone_breakdown = p.get("stone_breakdown", [])
        center_stone = next((s for s in stone_breakdown if s.get("center")), None)
        side_stones = [s for s in stone_breakdown if not s.get("center")]

        shape = "Round"
        if center_stone and center_stone.get("shape"):
            shape = center_stone.get("shape").title()
        elif p.get("image_shape_options"):
            shape = p.get("image_shape_options")[0].title()
        elif "oval" in title.lower():
            shape = "Oval"
        elif "emerald" in title.lower():
            shape = "Emerald"
        elif "princess" in title.lower():
            shape = "Princess"
        elif "cushion" in title.lower():
            shape = "Cushion"
        elif "radiant" in title.lower():
            shape = "Radiant"
        elif "pear" in title.lower():
            shape = "Pear"
        elif "marquise" in title.lower():
            shape = "Marquise"
        elif "asscher" in title.lower():
            shape = "Asscher"
        elif "heart" in title.lower():
            shape = "Heart"

        center_carat = center_stone.get("carat", 1.0) if center_stone else 1.0
        center_dimensions = center_stone.get("dimension", "") if center_stone else ""

        total_side_qty = 0
        total_side_carat = 0.0
        for s in side_stones:
            qty = s.get("quantity") or 0
            carat_val = s.get("carat") or 0.0
            try:
                qty_int = int(qty)
                carat_flt = float(carat_val)
                total_side_qty += qty_int
                total_side_carat += qty_int * carat_flt
            except Exception:
                pass
        total_side_carat = round(total_side_carat, 3)
        diamond_color_quality = p.get("default_quality") or "SI1-SI2, G-H"

        band_architecture = extract_band_architecture(title, description, collections)
        band_type = extract_band_type(title, description, collections, total_side_qty)
        overall_style = extract_style(title, description, collections)

        available_colors = p.get("colors", ["White", "Yellow", "Rose", "Platinum"])
        default_metal = p.get("default_metal", "14 KT")
        default_color = p.get("default_color", "White")

        primary_img = images[0] if images else ""
        side_img = next((img for img in images if ".side." in img), "")
        set_img = next((img for img in images if ".set." in img), "")
        angle_img = next((img for img in images if ".angle." in img), "")

        video_white = next((v for v in videos if "white" in v), "")
        video_yellow = next((v for v in videos if "yellow" in v), "")
        video_rose = next((v for v in videos if "rose" in v), "")

        color_variants = []
        for color_name in available_colors:
            metal_info = METAL_COLOR_MAP.get(color_name, {"category": color_name, "purity": f"{default_metal} {color_name}", "color": color_name})
            prong_info = extract_prong_style(title, description, peg_head, collections, metal_info["category"])

            if color_name == "Yellow":
                var_primary = next((img for img in images if ".alt." in img and not ".side." in img and not ".set." in img), primary_img)
                var_video = video_yellow or video_white
            elif color_name == "Rose":
                var_primary = next((img for img in images if ".alt1." in img and not ".side." in img and not ".set." in img), primary_img)
                var_video = video_rose or video_white
            else:
                var_primary = primary_img
                var_video = video_white

            color_variants.append({
                "style_number": f"{style_no}-{color_name[:2].upper()}",
                "metal_color": f"{default_metal} {color_name}",
                "band_color": metal_info["color"],
                "metal_category": metal_info["category"],
                "prong_style": prong_info["prong_style"],
                "prong_color": prong_info["prong_color"],
                "prong_summary": prong_info["prong_summary"],
                "primary_image": var_primary,
                "video_url": var_video
            })

        prong_default = extract_prong_style(title, description, peg_head, collections, default_color)

        normalized_items.append({
            "id": style_no,
            "handle": p.get("url", style_no),
            "style_number": style_no,
            "base_sku": style_no.split("-")[0] if "-" in style_no else style_no,
            "title": title,
            "description": description or f"{title} featuring {shape} cut diamond with {band_architecture} and {band_type}.",
            "vendor": "Overnight Mountings",
            "source": "api",
            "shape": shape,
            "band_color": default_color,
            "band_type": band_type,
            "band_architecture": band_architecture,
            "prong_style": prong_default["prong_style"],
            "prong_color": prong_default["prong_color"],
            "prong_summary": prong_default["prong_summary"],
            "style": overall_style,
            "center_diamond": {
                "shape": shape,
                "carat": center_carat,
                "dimension": center_dimensions,
                "type": "Diamond"
            },
            "side_diamonds": {
                "qty": total_side_qty,
                "carat": total_side_carat,
                "color": diamond_color_quality
            },
            "primary_image": primary_img,
            "side_image": side_img,
            "set_image": set_img,
            "angle_image": angle_img,
            "all_images": images,
            "videos": videos,
            "default_video": video_white or (videos[0] if videos else ""),
            "color_variants": color_variants,
            "collections": collections,
            "finger_size": p.get("finger_size", 7.0),
            "polish_weight_grams": p.get("polish_weight", 2.5),
            "peg_head_setting": peg_head
        })

    logger.info(f"Successfully processed {len(normalized_items)} API products.")
    return normalized_items


def build_complete_unified_catalog(scraped_file: str = SCRAPED_PRODUCTS_PATH, output_file: str = CATALOG_PATH) -> Dict[str, Any]:
    """Unifies and normalizes products into a single master catalog.json."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    api_products = parse_scraped_api_products(scraped_file)

    master_products: Dict[str, Any] = {}
    master_gallery: List[Dict[str, Any]] = []

    for p in api_products:
        pid = p["id"]
        master_products[pid] = p

        for img in p.get("all_images", [])[:6]:
            if img:
                img_band_color = "White Gold"
                if ".alt1." in img:
                    img_band_color = "Rose Gold"
                elif ".alt." in img:
                    img_band_color = "Yellow Gold"
                elif "platinum" in img.lower():
                    img_band_color = "Platinum"

                master_gallery.append({
                    "product_id": pid,
                    "image_url": img,
                    "title": p["title"],
                    "shape": p["shape"],
                    "band_color": img_band_color,
                    "band_type": p["band_type"],
                    "band_architecture": p["band_architecture"],
                    "prong_style": p["prong_style"],
                    "prong_color": img_band_color,
                    "prong_summary": f"{p['prong_style']} in {img_band_color}",
                    "style": p["style"],
                    "side_diamonds": p["side_diamonds"]
                })

    catalog_data = {
        "total_products": len(master_products),
        "total_search_images": len(master_gallery),
        "taxonomy": {
            "shapes": ALL_SHAPES,
            "band_colors": ["White Gold", "Yellow Gold", "Rose Gold", "Platinum", "Two-Tone"],
            "band_types": sorted(list(set(p["band_type"] for p in master_products.values()))),
            "band_architectures": sorted(list(set(p["band_architecture"] for p in master_products.values()))),
            "prong_styles": sorted(list(set(p["prong_style"] for p in master_products.values()))),
            "styles": sorted(list(set(p["style"] for p in master_products.values())))
        },
        "products": master_products,
        "gallery_index": master_gallery
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2)

    logger.info(f"Unified catalog saved to {output_file} ({len(master_products)} products, {len(master_gallery)} images).")
    return master_products


if __name__ == "__main__":
    build_complete_unified_catalog()
