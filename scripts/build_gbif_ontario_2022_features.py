import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def build_features(input_path, csd_path, output_path):
    print("Starting GBIF Ontario 2022 Waterfowl Feature Aggregator...")
    print("=========================================================")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input GBIF pilot CSV not found: {input_path}")
    if not os.path.exists(csd_path):
        raise FileNotFoundError(f"CSD boundaries shapefile not found: {csd_path}")
        
    # 1. Read GBIF pilot CSV
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} raw GBIF occurrence records.")
    
    # 2. Create point geometries in EPSG:4326
    geometry = [Point(xy) for xy in zip(df['decimalLongitude'], df['decimalLatitude'])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # 3-4. Load and filter CSD boundaries to Ontario (province_u = 35)
    print(f"Loading CSD boundaries from: {csd_path}")
    csd_gdf = gpd.read_file(csd_path)
    ontario_csd = csd_gdf[csd_gdf['province_u'] == '35'].copy()
    ontario_csd['csd_uid'] = ontario_csd['csd_uid'].astype(str)
    num_csds = len(ontario_csd)
    print(f"Filtered boundaries to Ontario: {num_csds} unique CSDs.")
    
    # 5. Spatial Join
    print("Performing spatial join (points to CSD polygons)...")
    joined_gdf = gpd.sjoin(points_gdf, ontario_csd, how='inner', predicate='intersects')
    
    joined_count = len(joined_gdf)
    unjoined_count = len(points_gdf) - joined_count
    print(f"Spatial join completed: {joined_count} joined, {unjoined_count} unjoined.")
    
    # 6. Ensure year and month are integers
    joined_gdf['year'] = joined_gdf['year'].astype(int)
    joined_gdf['month'] = joined_gdf['month'].astype(int)
    
    # 7. Aggregate by CSD-month
    joined_gdf['is_duck'] = (joined_gdf['species_group'] == 'Duck').astype(int)
    joined_gdf['is_goose'] = (joined_gdf['species_group'] == 'Goose').astype(int)
    
    agg = joined_gdf.groupby(['csd_uid', 'year', 'month']).agg(
        gbif_waterfowl_count=('gbifID', 'count'),
        gbif_duck_count=('is_duck', 'sum'),
        gbif_goose_count=('is_goose', 'sum')
    ).reset_index()
    
    # 8. Create a complete Ontario CSD-month grid (577 CSDs * 12 months for 2022)
    print("Building complete Ontario CSD-month grid for 2022...")
    csd_uids = ontario_csd['csd_uid'].unique()
    grid = pd.DataFrame([
        {'csd_uid': csd, 'year': 2022, 'month': m}
        for csd in csd_uids
        for m in range(1, 13)
    ])
    expected_rows = num_csds * 12
    print(f"Grid size: {len(grid)} rows (Expected: {expected_rows})")
    
    # Merge aggregated counts with the grid
    merged = pd.merge(grid, agg, on=['csd_uid', 'year', 'month'], how='left')
    
    # Fill missing counts with 0
    count_cols = ['gbif_waterfowl_count', 'gbif_duck_count', 'gbif_goose_count']
    merged[count_cols] = merged[count_cols].fillna(0).astype(int)
    
    # 9. Apply pilot-specific rules
    # NOTE: In this pilot, because we only downloaded target waterfowl records,
    # we set gbif_total_bird_count equal to gbif_waterfowl_count, and waterfowl share
    # to 1.0 where total bird count > 0. In the full execution, we must download
    # total class Aves occurrences per CSD-month to serve as the denominator
    # for true relative activity/reporting intensity normalization.
    merged['gbif_total_bird_count'] = merged['gbif_waterfowl_count']
    merged['gbif_waterfowl_share'] = (merged['gbif_total_bird_count'] > 0).astype(float)
    merged['gbif_reporting_intensity'] = merged['gbif_total_bird_count']
    
    # bird_data_missing_flag = 1 when gbif_total_bird_count = 0, else 0
    merged['bird_data_missing_flag'] = (merged['gbif_total_bird_count'] == 0).astype(int)
    
    # Merge CSD details (name, province_u)
    merged = pd.merge(merged, ontario_csd[['csd_uid', 'csd_name', 'province_u']], on='csd_uid', how='left')
    
    # Reorder columns
    columns_order = [
        'csd_uid', 'csd_name', 'province_u', 'year', 'month',
        'gbif_waterfowl_count', 'gbif_duck_count', 'gbif_goose_count',
        'gbif_total_bird_count', 'gbif_waterfowl_share',
        'gbif_reporting_intensity', 'bird_data_missing_flag'
    ]
    merged = merged[columns_order]
    
    # QA Checks
    errors = []
    if len(merged) != expected_rows:
        errors.append(f"Grid row count changed: expected {expected_rows}, got {len(merged)}")
        
    dup_count = merged.duplicated(subset=['csd_uid', 'year', 'month']).sum()
    if dup_count > 0:
        errors.append(f"Duplicate CSD-months found: {dup_count}")
        
    for col in count_cols + ['gbif_total_bird_count', 'gbif_reporting_intensity']:
        if (merged[col] < 0).any():
            errors.append(f"Negative count values found in {col}")
            
    share_vals = merged['gbif_waterfowl_share'].dropna()
    if (share_vals < 0).any() or (share_vals > 1).any():
        errors.append("gbif_waterfowl_share is not within [0.0, 1.0]")
        
    # Print QA Summary
    print("\n================== QA Summary ==================")
    print(f"CSDs processed: {num_csds}")
    print(f"Total row count: {len(merged)} (Expected: {expected_rows})")
    print(f"Duplicate rows: {dup_count}")
    print(f"Records joined: {joined_count} | Unjoined: {unjoined_count}")
    
    missing_count = merged['bird_data_missing_flag'].sum()
    print(f"Missing bird data (zero observations): {missing_count} CSD-months ({missing_count / len(merged) * 100:.2f}%)")
    
    print("\nWaterfowl count summary by month:")
    monthly_summary = merged.groupby('month')['gbif_waterfowl_count'].sum()
    for m, val in monthly_summary.items():
        print(f"  - Month {m}: {val} occurrences")
        
    print("\nTop 5 CSD-months by gbif_waterfowl_count:")
    top_months = merged.sort_values(by='gbif_waterfowl_count', ascending=False).head(5)
    for _, row in top_months.iterrows():
        print(f"  - CSD {row['csd_uid']} ({row['csd_name']}), Month {row['month']}: {row['gbif_waterfowl_count']} occurrences")
        
    print("================================================")
    
    if len(errors) > 0:
        print("Validation errors occurred:")
        for err in errors:
            print(f"  [ERROR] {err}")
        raise ValueError("Processed CSV validation failed. Output was not saved.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved features to: {output_path}")

if __name__ == '__main__':
    default_input = 'data/raw/gbif/ontario_2022_waterfowl_pilot.csv'
    default_csd = 'data/processed/gee_assets/canada_csd_2021_boundaries_for_gee_shp/canada_csd_2021_boundaries_for_gee.shp'
    default_output = 'data/processed/features/gbif_ontario_2022_waterfowl_features.csv'
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    csd_file = sys.argv[2] if len(sys.argv) > 2 else default_csd
    output_file = sys.argv[3] if len(sys.argv) > 3 else default_output
    
    try:
        build_features(input_file, csd_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
