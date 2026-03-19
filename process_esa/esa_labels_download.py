from terracatalogueclient import Catalogue
from shapely.geometry import Polygon

from download_config import LABEL_BOUNDS, LABEL_YEARS, WORLD_COVER_SPECS

if __name__ == "__main__":
    catalogue = Catalogue().authenticate()

    geometry = Polygon.from_bounds(*LABEL_BOUNDS)

    for year in LABEL_YEARS:
        spec = WORLD_COVER_SPECS[year]
        urn = spec["urn"]
        out_dir = spec["out_dir"]

        products = list(catalogue.get_products(urn, geometry=geometry))
        print(f"WorldCover {year} products:", len(products))
        catalogue.download_products(products, out_dir)