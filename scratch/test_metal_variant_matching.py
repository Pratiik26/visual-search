import json

with open("backend/data/catalog.json", "r", encoding="utf-8") as f:
    cat = json.load(f)

products = list(cat.get("products", {}).values())

print(f"Loaded {len(products)} products from catalog.")
for p in products[:5]:
    alt = p.get("all_images", [])
    white = next((img for img in alt if ".alt" not in img and ".alt1" not in img), p.get("primary_image"))
    yellow = next((img for img in alt if ".alt." in img), white)
    rose = next((img for img in alt if ".alt1." in img), white)
    print(f"PID: {p.get('product_id')} | Shape: {p.get('shape')}")
    print(f"  White:  {white}")
    print(f"  Yellow: {yellow}")
    print(f"  Rose:   {rose}")
