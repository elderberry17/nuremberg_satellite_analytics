from terracatalogueclient import Catalogue
from shapely.geometry import Polygon

catalogue = Catalogue().authenticate()

bounds = (10.95, 49.38, 11.15, 49.52)
geometry = Polygon.from_bounds(*bounds)

# 2020 labels
p2020 = list(catalogue.get_products("urn:eop:VITO:ESA_WorldCover_10m_2020_V1", geometry=geometry))
print("WorldCover 2020 products:", len(p2020))
catalogue.download_products(p2020, "WorldCover_labels_2020")

# 2021 labels
p2021 = list(catalogue.get_products("urn:eop:VITO:ESA_WorldCover_10m_2021_V2", geometry=geometry))
print("WorldCover 2021 products:", len(p2021))
catalogue.download_products(p2021, "WorldCover_labels_2021")