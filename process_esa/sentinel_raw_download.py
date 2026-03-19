import os
from terracatalogueclient import Catalogue
from shapely.geometry import Polygon

from download_config import (
    RAW_OUT_DIR,
    RAW_BOUNDS,
    RAW_PRODUCT_URN,
    RAW_START,
    RAW_END,
    RAW_CLOUD_COVER,
    RAW_LIMIT,
    RAW_KEEP_TOKENS,
)


if __name__ == "__main__":
    os.makedirs(RAW_OUT_DIR, exist_ok=True)

    catalogue = Catalogue().authenticate()

    geometry = Polygon.from_bounds(*RAW_BOUNDS)

    products = list(catalogue.get_products(
        RAW_PRODUCT_URN,
        start=RAW_START,
        end=RAW_END,
        geometry=geometry,
        cloudCover=RAW_CLOUD_COVER,
        limit=RAW_LIMIT,
    ))

    print("Found:", len(products))

    for p in products:
        ts = p.beginningDateTime.strftime("%Y%m%dT%H%M%S")
        scene_dir = os.path.join(RAW_OUT_DIR, f"{p.id}__{ts}")
        os.makedirs(scene_dir, exist_ok=True)

        # p.data = список файлов внутри продукта
        for f in p.data:
            name = f.title or os.path.basename(f.href)
            if any(k in name for k in RAW_KEEP_TOKENS):
                catalogue.download_file(f, scene_dir)

        print(f"Downloaded {RAW_KEEP_TOKENS} for:", p.id, ts)