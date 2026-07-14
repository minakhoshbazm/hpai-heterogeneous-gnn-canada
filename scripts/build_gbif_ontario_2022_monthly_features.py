import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def build_monthly_features(input_path, csd_path, output_path):
    print("Starting GBIF Ontario 2022 Monthly Waterfowl Feature Aggregator...")
    print("=================================================================")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input GBIF monthly pilot CSV not found: {input_path}")
    if not os.path.exists(csd_path):
        raise FileNotFoundError(f"CSD boundaries shapefile not found: {csd_path}")
        
    # 1. Read GBIF monthly pilot CSV
    df = pd.read_csv(input_path)
    raw_count = len(df)
    print(f"Loaded {raw_count} raw GBIF occurrence records.")
    
    # 2. Exclude domestic and hybrid records
    # We filter out scientificName containing domesticus, domestica, hybrid, or hybrid ' x '
    exclude_pattern = r'domesticus|domestica|hybrid|\bx\b|\bx\s|\s\bx\b'
    filtered_df = df[~df['scientificName'].str.lower().str.contains(exclude_pattern, regex=True, na=False)].copy()
    filtered_count = len(filtered_df)
    excluded_count = raw_count - filtered_count
    print(f"Filtered domestic/hybrid records:")
    print(f"  - Excluded: {excluded_count} records")
    print(f"  - Kept: {filtered_count} records")
    
    # 3. Create point geometries in EPSG:4326
    geometry = [Point(xy) for xy in zip(filtered_df['decimalLongitude'], filtered_df['decimalLatitude'])]
    points_gdf = gpd.GeoDataFrame(filtered_df, geometry=geometry, crs="EPSG:4326")
    
    # Load CSD boundaries and filter to Ontario
    csd_gdf = gpd.read_file(csd_path)
    ontario_csd = csd_gdf[csd_gdf['province_u'] == '35'].copy()
    ontario_csd['csd_uid'] = ontario_csd['csd_uid'].astype(str)
    num_csds = len(ontario_csd)
    
    # Spatial Join
    joined_gdf = gpd.sjoin(points_gdf, ontario_csd, how='inner', predicate='intersects')
    joined_count = len(joined_gdf)
    unjoined_count = filtered_count - joined_count
    
    # Ensure year/month are integers
    joined_gdf['year'] = joined_gdf['year'].astype(int)
    joined_gdf['month'] = joined_gdf['month'].astype(int)
    
    # Aggregate
    joined_gdf['is_duck'] = (joined_gdf['species_group'] == 'Duck').astype(int)
    joined_gdf['is_goose'] = (joined_gdf['species_group'] == 'Goose').astype(int)
    
    agg = joined_gdf.groupby(['csd_uid', 'year', 'month']).agg(
        gbif_waterfowl_count=('gbifID', 'count'),
        gbif_duck_count=('is_duck', 'sum'),
        gbif_goose_count=('is_goose', 'sum')
    ).reset_index()
    
    # Build CSD-month grid
    csd_uids = ontario_csd['csd_uid'].unique()
    grid = pd.DataFrame([
        {'csd_uid': csd, 'year': 2022, 'month': m}
        for csd in csd_uids
        for m in range(1, 13)
    ])
    expected_rows = num_csds * 12
    
    merged = pd.merge(grid, agg, on=['csd_uid', 'year', 'month'], how='left')
    
    count_cols = ['gbif_waterfowl_count', 'gbif_duck_count', 'gbif_goose_count']
    merged[count_cols] = merged[count_cols].fillna(0).astype(int)
    
    # Pilot limitations
    # NOTE: In this pilot, because we only downloaded target waterfowl records,
    # we set gbif_total_bird_count equal to gbif_waterfowl_count, and waterfowl share
    # to 1.0 where total bird count > 0. In the full execution, we must download
    # total class Aves occurrences per CSD-month to serve as the denominator
    # for true relative activity/reporting intensity normalization.
    merged['gbif_total_bird_count'] = merged['gbif_waterfowl_count']
    merged['gbif_waterfowl_share'] = (merged['gbif_total_bird_count'] > 0).astype(float)
    merged['gbif_reporting_intensity'] = merged['gbif_total_bird_count']
    merged['bird_data_missing_flag'] = (merged['gbif_total_bird_count'] == 0).astype(int)
    
    merged = pd.merge(merged, ontario_csd[['csd_uid', 'csd_name', 'province_u']], on='csd_uid', how='left')
    
    columns_order = [
        'csd_uid', 'csd_name', 'province_u', 'year', 'month',
        'gbif_waterfowl_count', 'gbif_duck_count', 'gbif_goose_count',
        'gbif_total_bird_count', 'gbif_waterfowl_share',
        'gbif_reporting_intensity', 'bird_data_missing_flag'
    ]
    merged = merged[columns_order]
    
    # Validation checks
    errors = []
    if len(merged) != expected_rows:
        errors.append(f"Grid row count changed: expected {expected_rows}, got {len(merged)}")
    dup_count = merged.duplicated(subset=['csd_uid', 'year', 'month']).sum()
    if dup_count > 0:
        errors.append(f"Duplicate CSD-months found: {dup_count}")
    for col in count_cols:
        if (merged[col] < 0).any():
            errors.append(f"Negative counts found in {col}")
    share_vals = merged['gbif_waterfowl_share'].dropna()
    if (share_vals < 0).any() or (share_vals > 1).any():
        errors.append("gbif_waterfowl_share is not within [0.0, 1.0]")
        
    # Print QA Summary
    print("\n================== QA Summary ==================")
    print(f"CSDs processed: {num_csds}")
    print(f"Total row count: {len(merged)} (Expected: {expected_rows})")
    print(f"Duplicate rows: {dup_count}")
    print(f"Records before domestic filter: {raw_count} | After: {filtered_count}")
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
    print(f"\nSuccessfully saved monthly features to: {output_path}")

if __name__ == '__main__':
    default_input = 'data/raw/gbif/ontario_2022_waterfowl_monthly_pilot.csv'
    default_csd = 'data/processed/gee_assets/canada_csd_2021_boundaries_for_gee_shp/canada_csd_2021_boundaries_for_gee.shp'
    default_output = 'data/processed/features/gbif_ontario_2022_waterfowl_monthly_features.csv'
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    csd_file = sys.argv[2] if len(sys.argv) > 2 else default_csd
    output_file = sys.argv[3] if len(sys.argv) > 3 else default_output
    
    try:
        build_monthly_features(input_file, csd_file, output_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
