import os
from terracatalogueclient import Catalogue
from shapely.geometry import Polygon

OUT_DIR = "S2_RGB_only"
os.makedirs(OUT_DIR, exist_ok=True)

catalogue = Catalogue().authenticate()

bounds = (10.95, 49.38, 11.15, 49.52)
geometry = Polygon.from_bounds(*bounds)

products = list(catalogue.get_products(
    "urn:eop:VITO:TERRASCOPE_S2_TOC_V2",
    start="2020-01-01",
    end="2023-01-01",
    geometry=geometry,
    cloudCover=30,
    limit=200
))

KEEP = ["TOC-B02_10M", "TOC-B03_10M", "TOC-B04_10M"]  # Blue, Green, Red

print("Found:", len(products))

for p in products:
    ts = p.beginningDateTime.strftime("%Y%m%dT%H%M%S")
    scene_dir = os.path.join(OUT_DIR, f"{p.id}__{ts}")
    os.makedirs(scene_dir, exist_ok=True)

    # p.data = список файлов внутри продукта
    for f in p.data:
        name = f.title or os.path.basename(f.href)
        if any(k in name for k in KEEP):
            catalogue.download_file(f, scene_dir)

    print("Downloaded RGB for:", p.id, ts)