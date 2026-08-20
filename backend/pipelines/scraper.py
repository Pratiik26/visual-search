"""
Scraper for OvernightMountings Engagement Rings API
Endpoint: https://www.overnightmountings.com/api/collection/engagement%20rings/
"""

import os
import time
import json
import logging
import urllib.request
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.config import SCRAPED_PRODUCTS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.overnightmountings.com/api/collection/engagement%20rings/?page={}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def fetch_single_page(page_num: int, max_retries: int = 5) -> Tuple[int, List[Dict[str, Any]], int]:
    """Fetches a single page of engagement rings from the API with retry & backoff."""
    url = BASE_URL.format(page_num)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.overnightmountings.com/"
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                products = data.get("products", [])
                pagination = data.get("pagination", {})
                total_pages = pagination.get("total_pages", 95)
                return page_num, products, total_pages
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = 1.5 * (attempt + 1)
                logger.warning(f"Rate limited on page {page_num}, waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                logger.error(f"HTTP error {e.code} on page {page_num}: {e}")
                time.sleep(1.0)
        except Exception as e:
            logger.warning(f"Fetch error on page {page_num} (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1.0)

    logger.error(f"Failed to fetch page {page_num} after {max_retries} attempts.")
    return page_num, [], 0


def scrape_all_pages(max_pages: int = None, output_file: str = SCRAPED_PRODUCTS_PATH, workers: int = 3) -> List[Dict[str, Any]]:
    """Scrapes engagement ring products across all pages and saves results."""
    logger.info("Starting scrape of OvernightMountings engagement rings...")

    _, page1_products, total_pages = fetch_single_page(1)
    if not page1_products:
        logger.error("Could not fetch page 1. Aborting scrape.")
        return []

    pages_to_fetch = total_pages if max_pages is None else min(max_pages, total_pages)
    logger.info(f"Total pages available: {total_pages}. Fetching {pages_to_fetch} pages...")

    all_products = list(page1_products)
    remaining_pages = list(range(2, pages_to_fetch + 1))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_single_page, p): p for p in remaining_pages}
        for future in as_completed(futures):
            p_num, prods, _ = future.result()
            if prods:
                all_products.extend(prods)
                logger.info(f"Page {p_num}/{pages_to_fetch} fetched ({len(prods)} products). Total so far: {len(all_products)}")
            time.sleep(0.1)

    seen_styles = set()
    unique_products = []
    for p in all_products:
        style_no = p.get("style_number")
        if style_no and style_no not in seen_styles:
            seen_styles.add(style_no)
            unique_products.append(p)
        elif not style_no:
            unique_products.append(p)

    logger.info(f"Scrape finished. Extracted {len(unique_products)} unique engagement ring products.")

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(unique_products, f, indent=2)
        logger.info(f"Saved scraped products to {output_file}")

    return unique_products


if __name__ == "__main__":
    scrape_all_pages(max_pages=95)
