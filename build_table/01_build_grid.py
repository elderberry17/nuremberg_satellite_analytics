from __future__ import annotations

from pathlib import Path

from common import ensure_parent, get_logger, load_boundary, load_config, make_regular_grid

CONFIG_PATH = Path('../config/project_config.yaml')
logger = get_logger('01_build_grid')


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    boundary = load_boundary(cfg['paths']['boundary_file'], cfg['crs']['metric'])
    grid = make_regular_grid(boundary, float(cfg['grid']['cell_size_m']), bool(cfg['grid'].get('keep_full_cells', True)))
    out_path = cfg['paths']['grid_file']
    ensure_parent(out_path)
    grid.to_file(out_path, driver='GeoJSON')
    logger.info('Saved grid to %s', out_path)
    logger.info('Cell size=%sm | cells=%d | mean_area=%.2f m²', cfg['grid']['cell_size_m'], len(grid), grid['area_m2'].mean())


if __name__ == '__main__':
    main()
