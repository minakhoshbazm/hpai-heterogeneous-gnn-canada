import os
import random
import warnings
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
import numpy as np

# ---------------------- Configuration ----------------------
# Paths (as per user request)
CSD_SHAPE_PATH = "d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/processed/gee_assets/canada_csd_2021_boundaries_for_gee_shp/canada_csd_2021_boundaries_for_gee.shp"
MODEL_TABLE_PATH = "d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/processed/modeling/model_table_v1_environment_train_ready.csv"
EBIRD_RAW_ROOT = "d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/raw/hpai_ebird_birds_9km/2023/"
OUTPUT_PILOT_PATH = "d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/processed/features/csd_monthly_ebird_status_2023_9km_pilot.csv"
OUTPUT_IN_PROGRESS_PATH = "d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/processed/features/csd_monthly_ebird_status_2023_9km.in_progress.csv"
OUTPUT_FULL_PATH = "d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/processed/features/csd_monthly_ebird_status_2023_9km.csv"

# Species codes (folder names under the raw path)
SPECIES = {
    "Mallard": "mallar3",
    "American Black Duck": "ambduc",
    "Northern Pintail": "norpin",
    "Green-winged Teal": "gnwtea",
    "Blue-winged Teal": "buwtea",
    "Canvasback": "canvas",
    "Redhead": "redhea",
    "Snow Goose": "snogoo",
    "Canada Goose": "cangoo",
}

# Duck and Goose groupings
DUCK_SPECIES = ["Mallard", "American Black Duck", "Northern Pintail", "Green-winged Teal", "Blue-winged Teal", "Canvasback", "Redhead"]
GOOSE_SPECIES = ["Snow Goose", "Canada Goose"]

# Random seed for reproducibility
RANDOM_SEED = 42

# ----------------------------------------------------------

def load_csd_boundary():
    """Load the processed CSD shapefile and filter to the 5,054 csd_uids present in the model table.
    Uses pyogrio to load the dataset.
    """
    import pyogrio
    csd_gdf = pyogrio.read_dataframe(CSD_SHAPE_PATH)
    if "csd_uid" not in csd_gdf.columns:
        raise KeyError("csd_uid field not found in CSD shapefile.")
    model_df = pd.read_csv(MODEL_TABLE_PATH, dtype={"csd_uid": str})
    valid_uids = set(model_df["csd_uid"].astype(str).unique())
    filtered = csd_gdf[csd_gdf["csd_uid"].astype(str).isin(valid_uids)].copy()
    if filtered.empty:
        raise ValueError("No matching csd_uid found after filtering.")
    return len(csd_gdf), filtered


def weekly_dates():
    """Generate 52 weekly dates starting 2023-01-04."""
    return pd.date_range("2023-01-04", periods=52, freq="7D")


def list_raster_paths(species_code, week_idx):
    """Construct raster file path for a given species and week index (0‑based).
    Searches recursively within the species directory for .tif files and sorts them.
    """
    species_dir = os.path.join(EBIRD_RAW_ROOT, species_code)
    tif_files = []
    for root, _, files in os.walk(species_dir):
        for f in files:
            if f.lower().endswith('.tif'):
                tif_files.append(os.path.join(root, f))
    tif_files.sort()
    if week_idx >= len(tif_files):
        raise IndexError(f"Week index {week_idx} out of range for species {species_code} (found {len(tif_files)} files).")
    return tif_files[week_idx]


def read_first_raster_crs():
    """Read CRS from the first available raster across all species, searching recursively.
    Returns the CRS of the first found TIFF file.
    """
    for sp_code in SPECIES.values():
        species_dir = os.path.join(EBIRD_RAW_ROOT, sp_code)
        if not os.path.exists(species_dir):
            continue
        tif_files = []
        for root, _, files in os.walk(species_dir):
            for f in files:
                if f.lower().endswith('.tif'):
                    tif_files.append(os.path.join(root, f))
        tif_files.sort()
        if tif_files:
            with rasterio.open(tif_files[0]) as src:
                return src.crs
    raise FileNotFoundError(f"No TIFF files found in any species directory under {EBIRD_RAW_ROOT}")


def reproject_csd(csd_gdf, target_crs):
    """Reproject CSD GeoDataFrame to target CRS (if needed)."""
    if csd_gdf.crs != target_crs:
        # Simplify high-vertex boundaries slightly (tolerance=5m) to prevent memory allocation error during reprojection
        simplified = csd_gdf.copy()
        simplified["geometry"] = simplified.geometry.simplify(5, preserve_topology=True)
        return simplified.to_crs(target_crs)
    return csd_gdf


def zonal_mean_from_bands(species_code, band_numbers, csd_subset):
    """Compute NaN‑aware mean of specified bands for each CSD polygon.
    *species_code* is the folder name (e.g., 'mallar3').
    *band_numbers* is a list of 1‑based band indices.
    *csd_subset* is a GeoDataFrame containing the CSD polygons of interest.
    Returns a pandas Series indexed by CSD rows.
    """
    species_dir = os.path.join(EBIRD_RAW_ROOT, species_code)
    tif_path = None
    for root, _, files in os.walk(species_dir):
        for f in files:
            if f.lower().endswith('.tif'):
                tif_path = os.path.join(root, f)
                break
        if tif_path:
            break
    if not tif_path:
        raise FileNotFoundError(f"No TIFF file found for species {species_code} in {species_dir}")

    # Reduced chunk size from 500 to 100 to minimize memory pressure
    chunk_size = 100
    num_chunks = int(np.ceil(len(csd_subset) / chunk_size))
    all_means = []

    for i in range(num_chunks):
        chunk_csd = csd_subset.iloc[i * chunk_size : (i + 1) * chunk_size]
        weekly_means_chunk = []
        
        for band in band_numbers:
            stats = zonal_stats(
                vectors=chunk_csd,
                raster=tif_path,
                band=band,
                stats=["mean"],
                all_touched=True
            )
            means = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
            weekly_means_chunk.append(means)
            
        if weekly_means_chunk:
            weekly_array_chunk = np.array(weekly_means_chunk)  # shape (weeks, chunk_csd)
            avg_means_chunk = np.nanmean(weekly_array_chunk, axis=0)
            all_means.extend(avg_means_chunk)
        else:
            all_means.extend([np.nan] * len(chunk_csd))

    return pd.Series(all_means, index=csd_subset.index)


def run_pilot():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    weekly_dates_vec = weekly_dates()
    jan_week_bands = [index + 1 for index, date in enumerate(weekly_dates_vec) if date.month == 1]
    jan_dates = [date.strftime("%Y-%m-%d") for date in weekly_dates_vec if date.month == 1]
    
    print("--- Pre-Extraction Dates and Band Mapping ---")
    print(f"January Dates: {', '.join(jan_dates)}")
    print(f"January Band Numbers (1-based): {', '.join(map(str, jan_week_bands))}")
    print("---------------------------------------------\n")
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        
        raw_boundary_count, csd_gdf = load_csd_boundary()
        sampled = csd_gdf.sample(n=10, random_state=RANDOM_SEED).copy()
        target_crs = read_first_raster_crs()
        sampled = reproject_csd(sampled, target_crs)
        
        for sp_code in list(SPECIES.values())[1:]:
            species_dir = os.path.join(EBIRD_RAW_ROOT, sp_code)
            tif_files = []
            for root, _, files in os.walk(species_dir):
                for f in files:
                    if f.lower().endswith('.tif'):
                        tif_files.append(os.path.join(root, f))
            tif_files.sort()
            if not tif_files:
                raise FileNotFoundError(f"No TIFF files found in {species_dir}")
            with rasterio.open(tif_files[0]) as src:
                if src.crs != target_crs:
                    raise ValueError(f"CRS mismatch for species {sp_code}")
        
        species_data = {}
        for sp_name, sp_code in SPECIES.items():
            col = sp_name.lower().replace(' ', '_') + "_relative_abundance"
            mean_series = zonal_mean_from_bands(sp_code, jan_week_bands, sampled)
            species_data[col] = mean_series

        rows = []
        for idx, csd_row in sampled.iterrows():
            csd_uid = csd_row["csd_uid"]
            row = {"csd_uid": csd_uid, "year": 2023, "month": 1, "ebird_status_product_year": 2023}
            
            species_vals = {}
            for sp_name, sp_code in SPECIES.items():
                col = sp_name.lower().replace(' ', '_') + "_relative_abundance"
                val = species_data[col].loc[idx]
                row[col] = val
                species_vals[sp_name] = val
                
            duck_sum = sum(species_vals[sp] for sp in DUCK_SPECIES)
            goose_sum = sum(species_vals[sp] for sp in GOOSE_SPECIES)
            row["ebird_duck_relative_abundance"] = duck_sum
            row["ebird_goose_relative_abundance"] = goose_sum
            row["ebird_selected_waterfowl_relative_abundance"] = duck_sum + goose_sum
            row["ebird_missing_flag"] = int(any(pd.isna(v) for v in species_vals.values()))
            rows.append(row)
            
        pilot_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(OUTPUT_PILOT_PATH), exist_ok=True)
        pilot_df.to_csv(OUTPUT_PILOT_PATH, index=False)
        
        print("--- Pilot eBird Monthly CSD Feature Extraction QA Summary ---")
        print(f"Output written to: {OUTPUT_PILOT_PATH}")
        print(f"Raw boundary count: {raw_boundary_count}")
        print(f"Filtered model-scope CSD count: {len(csd_gdf)}")
        print(f"Sampled CSD count: {len(sampled)}")
        print(f"Output row count: {len(pilot_df)}")
        print(f"CRS: {target_crs}")
        
        missingness_cols = list(pilot_df.filter(like="_relative_abundance").columns) + ["ebird_missing_flag"]
        missing_counts = pilot_df[missingness_cols].isna().sum()
        print("\nMissing values (missingness):")
        print(missing_counts.to_string())
        
        feature_cols = [sp_name.lower().replace(' ', '_') + "_relative_abundance" for sp_name in SPECIES.keys()] + [
            "ebird_duck_relative_abundance",
            "ebird_goose_relative_abundance",
            "ebird_selected_waterfowl_relative_abundance"
        ]
        
        print("\nMin / Max values for species and group features:")
        for col in feature_cols:
            series = pilot_df[col].dropna()
            if not series.empty:
                print(f"  {col}: min={series.min():.4f}, max={series.max():.4f}")
            else:
                print(f"  {col}: all values missing")
                
        print("\nWarnings encountered during execution:")
        if caught_warnings:
            seen = set()
            for warning in caught_warnings:
                warning_key = (warning.category, str(warning.message))
                if warning_key not in seen:
                    seen.add(warning_key)
                    print(f"  - {warning.category.__name__}: {warning.message}")
        else:
            print("  None")
            
        print("\nSample rows (first 5):")
        print(pilot_df.head(5).to_string(index=False))


def inspect_in_progress_months(expected_csd_count):
    """Check OUTPUT_IN_PROGRESS_PATH for existing completed months.
    Returns (completed_months_set, needs_cleaning_bool).
    If a month has partial rows (< expected_csd_count), it needs cleaning before resumption.
    """
    if not os.path.exists(OUTPUT_IN_PROGRESS_PATH) or os.path.getsize(OUTPUT_IN_PROGRESS_PATH) == 0:
        return set(), False

    df = pd.read_csv(OUTPUT_IN_PROGRESS_PATH, usecols=["csd_uid", "year", "month"])
    month_counts = df["month"].value_counts().to_dict()

    completed_months = {m for m, count in month_counts.items() if count == expected_csd_count}
    partial_months = {m for m, count in month_counts.items() if count != expected_csd_count}

    if partial_months:
        print(f"Found partial month entries in in-progress file: {partial_months}. Cleaning partial entries...")
        full_df = pd.read_csv(OUTPUT_IN_PROGRESS_PATH)
        clean_df = full_df[full_df["month"].isin(completed_months)].copy()
        clean_df.to_csv(OUTPUT_IN_PROGRESS_PATH, index=False)
        print(f"Cleaned {OUTPUT_IN_PROGRESS_PATH}, kept fully completed months: {completed_months}")

    return completed_months, bool(partial_months)


def run_full_extraction(target_months=None):
    """Run extraction sequentially month-by-month with incremental disk writes and resume capability."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    weekly_dates_vec = weekly_dates()
    all_sp_cols = [sp.lower().replace(' ', '_') + "_relative_abundance" for sp in SPECIES.keys()]
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        
        raw_boundary_count, csd_gdf = load_csd_boundary()
        expected_csd_count = len(csd_gdf)
        target_crs = read_first_raster_crs()
        csd_gdf_reproj = reproject_csd(csd_gdf, target_crs)
        
        # Verify CRS across species
        for sp_code in list(SPECIES.values())[1:]:
            species_dir = os.path.join(EBIRD_RAW_ROOT, sp_code)
            tif_files = []
            for root, _, files in os.walk(species_dir):
                for f in files:
                    if f.lower().endswith('.tif'):
                        tif_files.append(os.path.join(root, f))
            tif_files.sort()
            if not tif_files:
                raise FileNotFoundError(f"No TIFF files found in {species_dir}")
            with rasterio.open(tif_files[0]) as src:
                if src.crs != target_crs:
                    raise ValueError(f"CRS mismatch for species {sp_code}")
        
        os.makedirs(os.path.dirname(OUTPUT_IN_PROGRESS_PATH), exist_ok=True)
        completed_months, _ = inspect_in_progress_months(expected_csd_count)
        
        months_to_process = target_months if target_months is not None else list(range(1, 13))
        
        print(f"Starting sequential eBird extraction for months: {months_to_process}")
        print(f"Already completed months in in-progress file: {sorted(list(completed_months))}")

        for month in months_to_process:
            if month in completed_months:
                print(f"--> Month {month:02d} already completed. Skipping.", flush=True)
                continue

            month_bands = [index + 1 for index, date in enumerate(weekly_dates_vec) if date.month == month]
            print(f"\n[Month {month:02d}/12] Extracting zonal means (Bands: {month_bands})...", flush=True)
            
            species_data = {}
            for sp_name, sp_code in SPECIES.items():
                col = sp_name.lower().replace(' ', '_') + "_relative_abundance"
                species_data[col] = zonal_mean_from_bands(sp_code, month_bands, csd_gdf_reproj)
                
            month_df = pd.DataFrame({
                "csd_uid": csd_gdf_reproj["csd_uid"].astype(str),
                "year": 2023,
                "month": month,
                "ebird_status_product_year": 2023
            })
            
            for col in species_data.keys():
                month_df[col] = species_data[col].values
                
            duck_cols = [sp.lower().replace(' ', '_') + "_relative_abundance" for sp in DUCK_SPECIES]
            goose_cols = [sp.lower().replace(' ', '_') + "_relative_abundance" for sp in GOOSE_SPECIES]
            
            month_df["ebird_duck_relative_abundance"] = month_df[duck_cols].sum(axis=1, skipna=False)
            month_df["ebird_goose_relative_abundance"] = month_df[goose_cols].sum(axis=1, skipna=False)
            month_df["ebird_selected_waterfowl_relative_abundance"] = (
                month_df["ebird_duck_relative_abundance"] + month_df["ebird_goose_relative_abundance"]
            )
            month_df["ebird_missing_flag"] = month_df[all_sp_cols].isna().any(axis=1).astype(int)
            
            # Sort month_df by csd_uid to ensure consistent row ordering
            month_df = month_df.sort_values(by="csd_uid").reset_index(drop=True)
            
            # Append immediately to disk
            file_exists = os.path.exists(OUTPUT_IN_PROGRESS_PATH) and os.path.getsize(OUTPUT_IN_PROGRESS_PATH) > 0
            month_df.to_csv(OUTPUT_IN_PROGRESS_PATH, mode='a' if file_exists else 'w', header=not file_exists, index=False)
            completed_months.add(month)
            print(f"Successfully processed and written Month {month:02d} ({len(month_df)} rows) to in-progress CSV.", flush=True)

        # QA Check & Summary for in-progress / final output
        if os.path.exists(OUTPUT_IN_PROGRESS_PATH):
            full_df = pd.read_csv(OUTPUT_IN_PROGRESS_PATH, dtype={"csd_uid": str})
            
            print("\n--- eBird 2023 National Monthly Extraction QA Summary ---", flush=True)
            print(f"In-progress CSV path: {OUTPUT_IN_PROGRESS_PATH}", flush=True)
            print(f"Raw boundary count: {raw_boundary_count}", flush=True)
            print(f"Filtered model-scope CSD count: {expected_csd_count}", flush=True)
            print(f"Total row count in in-progress file: {len(full_df)}", flush=True)
            print(f"Unique CSD count: {full_df['csd_uid'].nunique()}", flush=True)
            
            print("\nRow count for each month present:", flush=True)
            print(full_df["month"].value_counts().sort_index().to_string(), flush=True)
            
            duplicate_count = full_df.duplicated(subset=["csd_uid", "year", "month"]).sum()
            print(f"\nDuplicate csd_uid/year/month rows: {duplicate_count}", flush=True)
            
            print("\nebird_missing_flag counts:", flush=True)
            print(full_df["ebird_missing_flag"].value_counts().to_string(), flush=True)
            
            feature_cols = all_sp_cols + [
                "ebird_duck_relative_abundance",
                "ebird_goose_relative_abundance",
                "ebird_selected_waterfowl_relative_abundance"
            ]
            print("\nMin / Max values for species and group features:", flush=True)
            for col in feature_cols:
                series = full_df[col].dropna()
                if not series.empty:
                    print(f"  {col}: min={series.min():.4f}, max={series.max():.4f}", flush=True)
                else:
                    print(f"  {col}: all values missing", flush=True)
                    
            print("\nWarnings encountered during execution:", flush=True)
            if caught_warnings:
                seen = set()
                for warning in caught_warnings:
                    warning_key = (warning.category, str(warning.message))
                    if warning_key not in seen:
                        seen.add(warning_key)
                        print(f"  - {warning.category.__name__}: {warning.message}", flush=True)
            else:
                print("  None", flush=True)

            # Atomic rename if all 12 months are complete and no duplicate rows exist
            if set(full_df["month"].unique()) == set(range(1, 13)) and len(full_df) == expected_csd_count * 12 and duplicate_count == 0:
                # Re-sort to maintain clean month/csd_uid ordering before final rename
                sorted_df = full_df.sort_values(by=["month", "csd_uid"]).reset_index(drop=True)
                sorted_df.to_csv(OUTPUT_IN_PROGRESS_PATH, index=False)
                os.replace(OUTPUT_IN_PROGRESS_PATH, OUTPUT_FULL_PATH)
                print(f"\n[SUCCESS] All 12 months completed and QA passed. Atomically renamed to: {OUTPUT_FULL_PATH}", flush=True)


if __name__ == "__main__":
    import sys
    if "--pilot" in sys.argv:
        run_pilot()
    else:
        # Check if a specific month was requested via --month M
        target_months = None
        if "--month" in sys.argv:
            idx = sys.argv.index("--month")
            if idx + 1 < len(sys.argv):
                target_months = [int(sys.argv[idx + 1])]
        run_full_extraction(target_months=target_months)
