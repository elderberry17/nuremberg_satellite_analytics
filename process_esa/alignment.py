import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject

import alignment_config as cfg


def make_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def get_grid_info(tif_path: Path):
    with rasterio.open(tif_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }


def choose_default_nodata(dtype_str, nodata):
    if nodata is not None:
        return nodata

    dtype = np.dtype(dtype_str)
    if np.issubdtype(dtype, np.floating):
        return np.nan
    return 0


def same_grid(path_a: Path, path_b: Path, atol=1e-9) -> bool:
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        if a.crs != b.crs:
            return False
        if a.width != b.width or a.height != b.height:
            return False
        return all(abs(x - y) < atol for x, y in zip(a.transform, b.transform))


def reproject_raster(
    src_path: Path,
    dst_path: Path,
    dst_crs,
    dst_transform,
    dst_width,
    dst_height,
    resampling,
    dst_dtype=None,
    dst_nodata=None,
):
    with rasterio.open(src_path) as src:
        src_count = src.count
        src_dtype = src.dtypes[0]

        if dst_dtype is None:
            dst_dtype = src_dtype

        if dst_nodata is None:
            dst_nodata = choose_default_nodata(dst_dtype, src.nodata)

        meta = src.meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "count": src_count,
                "dtype": dst_dtype,
                "nodata": dst_nodata,
            }
        )

        make_parent(dst_path)

        with rasterio.open(dst_path, "w", **meta) as dst:
            for band_idx in range(1, src_count + 1):
                dst_arr = np.full((dst_height, dst_width), dst_nodata, dtype=dst_dtype)

                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=dst_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                )
                dst.write(dst_arr, band_idx)


def align_label_maps():
    """
    WC2020 is the canonical grid.
    Result:
      - WC2020 aligned copy (or original if overwrite)
      - WC2021 reprojected to WC2020 grid
    """
    master = get_grid_info(cfg.MASTER_GRID_PATH)

    wc2020_out = cfg.WC2020 if cfg.OVERWRITE else cfg.WC2020.parent / cfg.WC2020_ALIGNED_NAME
    wc2021_out = cfg.WC2021 if cfg.OVERWRITE else cfg.WC2021.parent / cfg.WC2021_ALIGNED_NAME

    # WC2020 is already on master grid
    if not cfg.OVERWRITE:
        if wc2020_out.exists() and cfg.SKIP_IF_EXISTS:
            print(f"[SKIP] exists: {wc2020_out}")
        else:
            make_parent(wc2020_out)
            shutil.copy2(cfg.WC2020, wc2020_out)
            print(f"[OK] copied master label: {wc2020_out}")

    # WC2021 -> WC2020 grid
    if wc2021_out.exists() and cfg.SKIP_IF_EXISTS:
        print(f"[SKIP] exists: {wc2021_out}")
    else:
        reproject_raster(
            src_path=cfg.WC2021,
            dst_path=wc2021_out,
            dst_crs=master["crs"],
            dst_transform=master["transform"],
            dst_width=master["width"],
            dst_height=master["height"],
            resampling=cfg.LABEL_RESAMPLING,
            dst_dtype="uint16",
            dst_nodata=0,
        )
        print(f"[OK] aligned WC2021 -> {wc2021_out}")

    # Sanity check
    wc2020_check = cfg.WC2020 if cfg.OVERWRITE else wc2020_out
    if same_grid(wc2020_check, wc2021_out):
        print("[OK] WC2020 and WC2021 are now on the same grid")
    else:
        print("[WARN] WC2020 and WC2021 are still not identical in grid")


def should_process_band(path: Path) -> bool:
    name = path.name

    if cfg.SKIP_DERIVED_FILES:
        if cfg.RAW_SUFFIX in name:
            return False
        if "_aligned" in name:
            return False

    if cfg.BAND_FILENAME_MUST_CONTAIN is not None:
        if not any(token in name for token in cfg.BAND_FILENAME_MUST_CONTAIN):
            return False

    return True


def build_output_path(src_path: Path) -> Path:
    if cfg.OVERWRITE:
        return src_path

    return src_path.with_name(f"{src_path.stem}{cfg.RAW_SUFFIX}{src_path.suffix}")


def align_scene_bands():
    """
    Reproject every raw band in every scene folder to the canonical label grid.
    """
    master = get_grid_info(cfg.MASTER_GRID_PATH)

    scene_dirs = sorted([p for p in cfg.SCENES_ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(scene_dirs)} scene directories")

    total_processed = 0
    total_skipped = 0

    for scene_dir in scene_dirs:
        # if str(scene_dir).endswith('20200426T101549') or str(scene_dir).endswith('20200913T101629'):

        tif_files = sorted(scene_dir.glob(cfg.BAND_GLOB))
        tif_files = [p for p in tif_files if should_process_band(p)]

        if not tif_files:
            print(f"[SKIP] no matching tif files in {scene_dir.name}")
            continue

        print(f"\nScene: {scene_dir.name}")

        for tif_path in tif_files:
            out_path = build_output_path(tif_path)

            if out_path.exists() and cfg.SKIP_IF_EXISTS:
                print(f"  [SKIP] exists: {out_path.name}")
                total_skipped += 1
                continue

            try:
                reproject_raster(
                    src_path=tif_path,
                    dst_path=out_path,
                    dst_crs=master["crs"],
                    dst_transform=master["transform"],
                    dst_width=master["width"],
                    dst_height=master["height"],
                    resampling=cfg.RAW_RESAMPLING,
                )
                print(f"  [OK] {tif_path.name} -> {out_path.name}")
                total_processed += 1
            except Exception as e:
                print(f"  [ERR] {tif_path.name}: {e}")

    print("\nDone.")
    print(f"Processed: {total_processed}")
    print(f"Skipped:   {total_skipped}")


def main():
    print("=== STEP 1: Align label maps ===")
    align_label_maps()

    print("\n=== STEP 2: Align raw scene bands ===")
    align_scene_bands()


if __name__ == "__main__":
    main()