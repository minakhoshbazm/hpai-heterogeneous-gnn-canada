import sys
from pathlib import Path

import geopandas as gpd


def prepare_csd_boundary_asset(
    input_shp: Path = Path("data/raw/boundaries/lcsd000b21a_e.shp"),
    output_geojson: Path = Path(
        "data/processed/gee_assets/canada_csd_2021_boundaries_for_gee.geojson"
    ),
    expected_rows: int = 5161,
) -> None:
    """Prepare a privacy-safe CSD boundary GeoJSON for Google Earth Engine.

    The function performs the following steps:
    1. Load the public Statistics Canada CSD shapefile.
    2. Re-project to EPSG:4326 (WGS84) if needed.
    3. Keep only the required attributes:
       csd_uid, csd_name, province_uid, province_name and geometry.
    4. Cast csd_uid to string to preserve leading zeros.
    5. Validate row count, uniqueness of csd_uid and absence of
       forbidden columns.
    6. Write the cleaned file as GeoJSON ready for GEE upload.
    7. Print a concise QA summary.
    """
    # ----------------------------------------------------------------------
    # 1. Load the shapefile
    # ----------------------------------------------------------------------
    if not input_shp.is_file():
        sys.exit(f"Input shapefile not found: {input_shp}")

    gdf = gpd.read_file(input_shp)

    # ----------------------------------------------------------------------
    # 2. Re-project to EPSG:4326 (WGS84) if required
    # ----------------------------------------------------------------------
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # ----------------------------------------------------------------------
    # 3. Keep only required columns and rename them
    # ----------------------------------------------------------------------
    # Validate required source columns are present
    required_src = ["CSDUID", "CSDNAME", "PRUID", "PRNAME"]
    missing_src = [c for c in required_src if c not in gdf.columns]
    if missing_src:
        sys.exit(f"Missing required source columns: {missing_src}")
    
    rename_map = {
        "CSDUID": "csd_uid",
        "CSDNAME": "csd_name",
        "PRUID": "province_uid",
        "PRNAME": "province_name",
    }

    # Select columns that exist in the source plus geometry
    existing = [c for c in rename_map if c in gdf.columns]
    gdf = gdf[existing + ["geometry"]].rename(columns=rename_map)

    # ----------------------------------------------------------------------
    # 4. Ensure csd_uid is a string (preserve leading zeros)
    # ----------------------------------------------------------------------
    gdf["csd_uid"] = gdf["csd_uid"].astype(str)

    # ----------------------------------------------------------------------
    # 5. Validation
    # ----------------------------------------------------------------------
    if len(gdf) != expected_rows:
        sys.exit(
            f"Row count mismatch: expected {expected_rows}, got {len(gdf)}"
        )

    if gdf["csd_uid"].isnull().any():
        sys.exit("Missing csd_uid values detected")

    if gdf["csd_uid"].duplicated().any():
        dup = gdf[gdf["csd_uid"].duplicated(keep=False)]["csd_uid"].unique()
        sys.exit(f"Duplicate csd_uid values found: {dup}")

    # Ensure no forbidden columns remain
    forbidden = set(gdf.columns) - {
        "csd_uid",
        "csd_name",
        "province_uid",
        "province_name",
        "geometry",
    }
    if forbidden:
        sys.exit(f"Forbidden columns present after filtering: {forbidden}")

    # Geometry must be present and valid
    if gdf.geometry.isnull().any():
        sys.exit("Some features have missing geometry")

    # ----------------------------------------------------------------------
    # 6. Write output GeoJSON (creates parent directories if needed)
    # ----------------------------------------------------------------------
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_geojson, driver="GeoJSON")

    # ----------------------------------------------------------------------
    # 7. QA summary
    # ----------------------------------------------------------------------
    print("--- QA Summary ---")
    print(f"Rows written           : {len(gdf)}")
    print(f"Unique csd_uid count   : {gdf['csd_uid'].nunique()}")
    print("CRS of output file     : EPSG:4326")
    print("Columns retained       :", ", ".join(gdf.columns))


if __name__ == "__main__":
    prepare_csd_boundary_asset()
