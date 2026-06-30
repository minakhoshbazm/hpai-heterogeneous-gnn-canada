"""
CSD Spatial Mapping and Labeling Pipeline for Poultry Outbreaks.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import difflib

# Disable OGR size limit for complex GeoJSON operations
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

# Province constants
PRUID_TO_NAME = {
    '10': 'Newfoundland and Labrador',
    '11': 'Prince Edward Island',
    '12': 'Nova Scotia',
    '13': 'New Brunswick',
    '24': 'Quebec',
    '35': 'Ontario',
    '46': 'Manitoba',
    '47': 'Saskatchewan',
    '48': 'Alberta',
    '59': 'British Columbia',
    '60': 'Yukon',
    '61': 'Northwest Territories',
    '62': 'Nunavut'
}

PR_ABBR_TO_NAME = {
    'AB': 'Alberta', 'BC': 'British Columbia', 'MB': 'Manitoba',
    'NB': 'New Brunswick', 'NL': 'Newfoundland and Labrador', 'NS': 'Nova Scotia',
    'ON': 'Ontario', 'PE': 'Prince Edward Island', 'QC': 'Quebec', 'SK': 'Saskatchewan',
    'YT': 'Yukon', 'NT': 'Northwest Territories', 'NU': 'Nunavut'
}

NAME_TO_PRUID = {v: k for k, v in PRUID_TO_NAME.items()}

def get_province_code(prov_text):
    if pd.isna(prov_text):
        return None
    prov_str = str(prov_text).strip().upper()
    if prov_str in PR_ABBR_TO_NAME:
        return NAME_TO_PRUID.get(PR_ABBR_TO_NAME[prov_str])
    
    # Try direct name lookup
    for name, code in NAME_TO_PRUID.items():
        if name.lower() == prov_str.lower():
            return code
    return None

def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

def map_outbreaks_to_csd(outbreak_path, csd_boundary_path, lookup_path, cd_boundary_path, output_dir):
    """Executes the hierarchical spatial mapping from CFIA poultry outbreaks to CSD nodes."""
    print("Step 1: Loading datasets...")
    df = pd.read_csv(outbreak_path)
    print(f"Loaded {len(df)} poultry outbreak records.")
    
    # 1. Load CSD Boundaries
    print(f"Loading CSD boundary shapefile from: {csd_boundary_path}")
    csd_gdf = gpd.read_file(csd_boundary_path)
    print(f"Loaded {len(csd_gdf)} CSD boundaries. CRS: {csd_gdf.crs}")
    
    # Ensure geometries are valid
    print("Validating geometries...")
    csd_gdf['geometry'] = csd_gdf.geometry.buffer(0)
    
    # Standardize boundary column names
    csd_gdf['CSDUID'] = csd_gdf['CSDUID'].astype(str)
    csd_gdf['PRUID'] = csd_gdf['PRUID'].astype(str).str.zfill(2)
    
    # 2. Load CSD Lookup Table
    print(f"Loading CSD lookup table from: {lookup_path}")
    lookup_df = pd.read_csv(lookup_path, dtype={'csd_uid': str, 'cd_uid': str, 'pruid': str})
    lookup_df['csd_name_norm'] = lookup_df['csd_name'].apply(normalize_text)
    
    # 3. Load CD Boundaries for CD Name mapping
    print(f"Loading CD boundaries for CD name resolver from: {cd_boundary_path}")
    cd_gdf = gpd.read_file(cd_boundary_path)
    cd_names_map = cd_gdf.set_index('cd_uid')['cd_name'].to_dict()
    
    # 4. Prepare Outbreak Data DataFrame
    mapped_events = df.copy()
    mapped_events['csd_uid'] = None
    mapped_events['csd_name'] = None
    mapped_events['cd_uid'] = None
    mapped_events['cd_name'] = None
    mapped_events['province_code'] = None
    mapped_events['province_name'] = None
    mapped_events['mapping_method'] = "unresolved"
    mapped_events['mapping_confidence'] = "low_or_failed"
    mapped_events['mapping_notes'] = ""
    
    issues_list = []
    
    # Pre-parse dates to check for any parsing failures
    mapped_events['parsed_date'] = pd.to_datetime(mapped_events['event_date'], errors='coerce')
    date_failures = mapped_events[mapped_events['parsed_date'].isna()].copy()
    for _, row in date_failures.iterrows():
        issues_list.append({
            'premises_id': row.get('premises_id'),
            'city': row.get('city'),
            'province': row.get('province'),
            'latitude': row.get('latitude'),
            'longitude': row.get('longitude'),
            'event_date': row.get('event_date'),
            'issue_reason': "date parsing failed",
            'mapping_notes': "Missing or corrupt event date"
        })
        
    # Standardize province text
    mapped_events['declared_province_code'] = mapped_events['province'].apply(get_province_code)
    
    # Step 2: Priority 1 - Coordinate-based mapping
    print("Step 2: Processing coordinate-based spatial joins (Priority 1)...")
    has_coords = mapped_events['latitude'].notna() & mapped_events['longitude'].notna()
    coord_df = mapped_events[has_coords].copy()
    
    if len(coord_df) > 0:
        geometry = [Point(xy) for xy in zip(coord_df['longitude'], coord_df['latitude'])]
        points_gdf = gpd.GeoDataFrame(coord_df, geometry=geometry, crs="EPSG:4326")
        
        # Reproject points to EPSG:3347
        points_gdf = points_gdf.to_crs("EPSG:3347")
        
        # Spatial join with CSD boundaries
        joined = gpd.sjoin(points_gdf, csd_gdf[['CSDUID', 'CSDNAME', 'PRUID', 'geometry']], how="left", predicate="intersects")
        
        # Map fields back
        for idx, row in joined.iterrows():
            csd_uid = row['CSDUID']
            if pd.isna(csd_uid):
                # Point did not intersect any polygon
                mapped_events.at[idx, 'mapping_notes'] = "Coordinates fell outside all CSD polygons."
                issues_list.append({
                    'premises_id': row['premises_id'],
                    'city': row['city'],
                    'province': row['province'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'event_date': row['event_date'],
                    'issue_reason': "coordinate outside boundaries",
                    'mapping_notes': "Coordinates did not intersect any CSD polygon."
                })
                continue
                
            csd_uid = str(csd_uid)
            csd_name = row['CSDNAME']
            pruid = str(row['PRUID']).zfill(2)
            declared_pruid = row['declared_province_code']
            
            mapped_events.at[idx, 'csd_uid'] = csd_uid
            mapped_events.at[idx, 'csd_name'] = csd_name
            mapped_events.at[idx, 'cd_uid'] = csd_uid[:4]
            mapped_events.at[idx, 'cd_name'] = cd_names_map.get(csd_uid[:4], "Unknown")
            mapped_events.at[idx, 'province_code'] = pruid
            mapped_events.at[idx, 'province_name'] = PRUID_TO_NAME.get(pruid, "Unknown")
            mapped_events.at[idx, 'mapping_method'] = "coordinate_intersection"
            mapped_events.at[idx, 'mapping_confidence'] = "high"
            
            # Check for cross-province mismatches
            if declared_pruid is not None and pruid != declared_pruid:
                notes = f"Coordinate maps to province {pruid} ({PRUID_TO_NAME.get(pruid)}) but declared province is {row['province']} ({declared_pruid})."
                mapped_events.at[idx, 'mapping_notes'] = notes
                issues_list.append({
                    'premises_id': row['premises_id'],
                    'city': row['city'],
                    'province': row['province'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'event_date': row['event_date'],
                    'issue_reason': "cross province coordinate mismatch",
                    'mapping_notes': notes
                })
                
    # Step 3: Priority 2 - Name-based mapping
    print("Step 3: Processing name-based lookups (Priority 2)...")
    unmapped_mask = mapped_events['csd_uid'].isna()
    unmapped_df = mapped_events[unmapped_mask].copy()
    print(f"Number of records needing name-based lookups: {len(unmapped_df)}")
    
    for idx, row in unmapped_df.iterrows():
        city_text = row['city']
        prov_code = row['declared_province_code']
        
        if pd.isna(city_text) or prov_code is None:
            # Missing location identifiers
            mapped_events.at[idx, 'mapping_notes'] = "Missing city text or province identifier."
            issues_list.append({
                'premises_id': row['premises_id'],
                'city': row['city'],
                'province': row['province'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'event_date': row['event_date'],
                'issue_reason': "missing location text",
                'mapping_notes': "Both coordinates and city name text are missing or corrupt."
            })
            continue
            
        city_norm = normalize_text(city_text)
        
        # Subset lookup table to current province to avoid duplicates across provinces
        prov_lookup = lookup_df[lookup_df['pruid'] == prov_code].copy()
        
        if len(prov_lookup) == 0:
            mapped_events.at[idx, 'mapping_notes'] = f"No subdivisions found in lookup for province code {prov_code}."
            issues_list.append({
                'premises_id': row['premises_id'],
                'city': row['city'],
                'province': row['province'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'event_date': row['event_date'],
                'issue_reason': "no CSD match in province",
                'mapping_notes': f"Declared province code {prov_code} has no entries in CSD lookup."
            })
            continue
            
        # Try exact match first
        exact_match = prov_lookup[prov_lookup['csd_name_norm'] == city_norm]
        
        if len(exact_match) == 1:
            csd_uid = exact_match.iloc[0]['csd_uid']
            csd_name = exact_match.iloc[0]['csd_name']
            
            mapped_events.at[idx, 'csd_uid'] = csd_uid
            mapped_events.at[idx, 'csd_name'] = csd_name
            mapped_events.at[idx, 'cd_uid'] = csd_uid[:4]
            mapped_events.at[idx, 'cd_name'] = cd_names_map.get(csd_uid[:4], "Unknown")
            mapped_events.at[idx, 'province_code'] = prov_code
            mapped_events.at[idx, 'province_name'] = PRUID_TO_NAME.get(prov_code)
            mapped_events.at[idx, 'mapping_method'] = "csd_name_match"
            mapped_events.at[idx, 'mapping_confidence'] = "medium"
            mapped_events.at[idx, 'mapping_notes'] = "Exact CSD name match."
        elif len(exact_match) > 1:
            # Ambiguous duplicate matching names in same province (unlikely for CSDs but check)
            notes = f"Multiple CSD exact matches found in province for name: {city_text}"
            mapped_events.at[idx, 'mapping_notes'] = notes
            issues_list.append({
                'premises_id': row['premises_id'],
                'city': row['city'],
                'province': row['province'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'event_date': row['event_date'],
                'issue_reason': "multiple CSD matches",
                'mapping_notes': notes
            })
        else:
            # Cautious fuzzy matching
            csd_names_list = prov_lookup['csd_name_norm'].tolist()
            close_matches = difflib.get_close_matches(city_norm, csd_names_list, n=1, cutoff=0.9)
            
            if close_matches:
                matched_norm = close_matches[0]
                matched_row = prov_lookup[prov_lookup['csd_name_norm'] == matched_norm].iloc[0]
                
                csd_uid = matched_row['csd_uid']
                csd_name = matched_row['csd_name']
                
                mapped_events.at[idx, 'csd_uid'] = csd_uid
                mapped_events.at[idx, 'csd_name'] = csd_name
                mapped_events.at[idx, 'cd_uid'] = csd_uid[:4]
                mapped_events.at[idx, 'cd_name'] = cd_names_map.get(csd_uid[:4], "Unknown")
                mapped_events.at[idx, 'province_code'] = prov_code
                mapped_events.at[idx, 'province_name'] = PRUID_TO_NAME.get(prov_code)
                mapped_events.at[idx, 'mapping_method'] = "csd_name_match"
                mapped_events.at[idx, 'mapping_confidence'] = "medium"
                mapped_events.at[idx, 'mapping_notes'] = f"Fuzzy match to: {csd_name} (similarity score >= 90%)."
            else:
                notes = f"Could not find exact or fuzzy CSD name match for: {city_text}"
                mapped_events.at[idx, 'mapping_notes'] = notes
                issues_list.append({
                    'premises_id': row['premises_id'],
                    'city': row['city'],
                    'province': row['province'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'event_date': row['event_date'],
                    'issue_reason': "no CSD match",
                    'mapping_notes': notes
                })
                
    # Step 4: Duplicate Candidates Check
    print("Step 4: Checking for duplicate candidates...")
    mapped_events = mapped_events.sort_values(by='event_date')
    # Duplicate if same premises ID in same month, or same coordinate proximity (+/- 0.01 degree) in same month
    mapped_events['lat_round'] = mapped_events['latitude'].round(2)
    mapped_events['lon_round'] = mapped_events['longitude'].round(2)
    
    dup_prem = mapped_events.duplicated(subset=['premises_id', 'event_month'], keep='first') & mapped_events['premises_id'].notna()
    dup_geo = mapped_events.duplicated(subset=['lat_round', 'lon_round', 'event_month'], keep='first') & mapped_events['latitude'].notna()
    
    duplicates = mapped_events[dup_prem | dup_geo].copy()
    for _, row in duplicates.iterrows():
        issues_list.append({
            'premises_id': row['premises_id'],
            'city': row['city'],
            'province': row['province'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'event_date': row['event_date'],
            'issue_reason': "duplicate candidate",
            'mapping_notes': "Outbreak shares Premises ID or close spatial proximity in same month."
        })
        
    # Write Issues Table
    issues_df = pd.DataFrame(issues_list)
    out_mapping_dir = os.path.join(output_dir, "cfia_mapping")
    os.makedirs(out_mapping_dir, exist_ok=True)
    
    issues_path = os.path.join(out_mapping_dir, "cfia_csd_mapping_issues.csv")
    issues_df.to_csv(issues_path, index=False)
    print(f"Saved mapping issues table to {issues_path}")
    
    # Save main mapped events output file
    mapped_output_cols = [
        'premises_id', 'event_date', 'event_month', 'province', 'city',
        'latitude', 'longitude', 'csd_uid', 'csd_name', 'cd_uid', 'cd_name',
        'province_code', 'province_name', 'mapping_method', 'mapping_confidence', 'mapping_notes'
    ]
    # Clean output renaming
    final_events = mapped_events.rename(columns={'premises_id': 'event_id', 'province': 'original_province', 'city': 'original_city'})
    final_output_cols = [
        'event_id', 'event_date', 'event_month', 'original_province', 'original_city',
        'latitude', 'longitude', 'csd_uid', 'csd_name', 'cd_uid', 'cd_name',
        'province_code', 'province_name', 'mapping_method', 'mapping_confidence', 'mapping_notes'
    ]
    
    events_path = os.path.join(out_mapping_dir, "cfia_events_mapped_to_csd.csv")
    final_events[final_output_cols].to_csv(events_path, index=False)
    print(f"Saved CSD-mapped outbreaks to {events_path}")
    
    # Step 5: CSD Month Outbreak Counts Grid
    print("Step 5: Generating CSD-month outbreak counts grid (Priority 4)...")
    # All 5,161 CSD UIDs
    all_csd_uids = csd_gdf['CSDUID'].unique()
    
    # Sourced months list (Dec 2021 to Dec 2025)
    start_date = pd.to_datetime("2021-12-01")
    end_date = pd.to_datetime("2025-12-01")
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
    
    # Cartesian product
    grid = pd.MultiIndex.from_product([all_csd_uids, all_months], names=['csd_uid', 'month']).to_frame().reset_index(drop=True)
    
    # Map CSD details
    csd_details = lookup_df[['csd_uid', 'csd_name', 'pruid']].copy().rename(columns={'pruid': 'province_code'})
    csd_details['province_name'] = csd_details['province_code'].map(PRUID_TO_NAME)
    
    # Merge CSD info
    grid = grid.merge(csd_details, on='csd_uid', how='left')
    
    # Filter outbreaks that successfully mapped to a CSD UID
    valid_mapped = mapped_events[mapped_events['csd_uid'].notna()].copy()
    
    # Aggregate counts by CSD and month
    counts_agg = valid_mapped.groupby(['csd_uid', 'event_month']).agg(
        outbreak_count=('premises_id', 'count'),
        high_confidence_count=('mapping_confidence', lambda x: (x == 'high').sum()),
        medium_confidence_count=('mapping_confidence', lambda x: (x == 'medium').sum()),
        low_confidence_count=('mapping_confidence', lambda x: (x == 'low_or_failed').sum())
    ).reset_index().rename(columns={'event_month': 'month'})
    
    # Merge aggregations into the full grid
    grid = grid.merge(counts_agg, on=['csd_uid', 'month'], how='left')
    grid['outbreak_count'] = grid['outbreak_count'].fillna(0).astype(int)
    grid['outbreak_binary'] = (grid['outbreak_count'] > 0).astype(int)
    grid['high_confidence_count'] = grid['high_confidence_count'].fillna(0).astype(int)
    grid['medium_confidence_count'] = grid['medium_confidence_count'].fillna(0).astype(int)
    grid['low_confidence_count'] = grid['low_confidence_count'].fillna(0).astype(int)
    
    counts_path = os.path.join(out_mapping_dir, "csd_month_outbreak_counts.csv")
    grid.to_csv(counts_path, index=False)
    print(f"Saved CSD-month outbreak counts to {counts_path}")
    
    # Step 6: CSD Month Labels (shifted targets)
    print("Step 6: Generating shifted target labels...")
    labels_df = grid.sort_values(by=['csd_uid', 'month']).copy()
    
    labels_df['outbreak_count_current_month'] = labels_df['outbreak_count']
    labels_df['outbreak_binary_current_month'] = labels_df['outbreak_binary']
    
    # Shift counts back by 1 step (-1) to represent next month's values
    labels_df['outbreak_count_next_month'] = labels_df.groupby('csd_uid')['outbreak_count'].shift(-1)
    labels_df['outbreak_next_month'] = labels_df.groupby('csd_uid')['outbreak_binary'].shift(-1)
    
    labels_cols = [
        'csd_uid', 'csd_name', 'province_code', 'province_name', 'month',
        'outbreak_count_current_month', 'outbreak_binary_current_month',
        'outbreak_count_next_month', 'outbreak_next_month'
    ]
    labels_path = os.path.join(out_mapping_dir, "csd_month_labels.csv")
    labels_df[labels_cols].to_csv(labels_path, index=False)
    print(f"Saved CSD shifted labels to {labels_path}")
    
    # Step 7: Mapping Summary CSV
    print("Step 7: Generating mapping summary report...")
    total_poultry = len(mapped_events)
    mapped_high = (mapped_events['mapping_confidence'] == 'high').sum()
    mapped_med = (mapped_events['mapping_confidence'] == 'medium').sum()
    mapped_low = (mapped_events['mapping_confidence'] == 'low_or_failed').sum()
    
    prov_counts = mapped_events['province_name'].value_counts().to_dict()
    top_csds = valid_mapped['csd_name'].value_counts().head(5).to_dict()
    unique_csds_with_outbreaks = valid_mapped['csd_uid'].nunique()
    total_csds = len(all_csd_uids)
    pct_csds_with_outbreaks = (unique_csds_with_outbreaks / total_csds) * 100
    
    summary_data = [
        {"Metric": "Total Cleaned Poultry Events", "Value": str(total_poultry)},
        {"Metric": "Mapped High Confidence", "Value": str(mapped_high)},
        {"Metric": "Mapped Medium Confidence", "Value": str(mapped_med)},
        {"Metric": "Low Confidence", "Value": str(mapped_low)},
        {"Metric": "Failed / Unresolved", "Value": str(total_poultry - mapped_high - mapped_med)},
        {"Metric": "Unique CSDs with Outbreaks", "Value": str(unique_csds_with_outbreaks)},
        {"Metric": "Total CSDs in Canada", "Value": str(total_csds)},
        {"Metric": "Percent of CSDs with Outbreaks", "Value": f"{pct_csds_with_outbreaks:.4f}%"}
    ]
    # Add province counts
    for p_name in sorted(PRUID_TO_NAME.values()):
        count = prov_counts.get(p_name, 0)
        summary_data.append({"Metric": f"Outbreak Count - {p_name}", "Value": str(count)})
        
    # Add top CSDs
    for rank, (name, cnt) in enumerate(top_csds.items(), 1):
        summary_data.append({"Metric": f"Top CSD #{rank} ({name})", "Value": str(cnt)})
        
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(out_mapping_dir, "csd_mapping_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved CSD mapping summary report to {summary_path}")
    
    # Step 8: QA Maps Plotting
    print("Step 8: Generating QA visual maps...")
    qa_maps_dir = os.path.abspath(os.path.join(output_dir, "..", "..", "results", "maps", "cfia_csd_mapping_qa"))
    os.makedirs(qa_maps_dir, exist_ok=True)
    
    # Simplify CSD boundaries for plotting to prevent MemoryError (conformal conic EPSG:3347 coordinates)
    print("Simplifying boundaries for plotting...")
    csd_plot_gdf = csd_gdf.copy()
    csd_plot_gdf['geometry'] = csd_plot_gdf.geometry.simplify(500, preserve_topology=False)
    
    # Plot 1: csd_total_outbreak_counts.png
    print("Plotting outbreak counts map...")
    csd_totals = valid_mapped['csd_uid'].value_counts().to_dict()
    csd_plot_gdf['outbreak_count'] = csd_plot_gdf['CSDUID'].map(csd_totals).fillna(0).astype(int)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#ffffff")
    
    # Draw CSD base layer in light gray
    csd_plot_gdf[csd_plot_gdf['outbreak_count'] == 0].plot(
        ax=ax, facecolor="#f0f0f0", edgecolor="#dddddd", linewidth=0.03
    )
    # Draw CSD outbreak layer in red
    csd_plot_gdf[csd_plot_gdf['outbreak_count'] > 0].plot(
        ax=ax, column='outbreak_count', cmap='Reds', legend=True, 
        edgecolor="#cc3333", linewidth=0.1, legend_kwds={'shrink': 0.5, 'label': 'Outbreaks Count'}
    )
    
    ax.set_title("Canada Census Subdivision (CSD) Total Outbreak Counts\n(CRS: EPSG:3347)", fontsize=14, fontweight='bold', pad=10)
    bounds = csd_plot_gdf.total_bounds
    ax.set_xlim(bounds[0] - 50000, bounds[2] + 50000)
    ax.set_ylim(bounds[1] - 50000, bounds[3] + 50000)
    
    count_map_path = os.path.join(qa_maps_dir, "csd_total_outbreak_counts.png")
    plt.savefig(count_map_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved counts map to {count_map_path}")
    
    # Plot 2: csd_mapping_confidence.png
    print("Plotting mapping confidence map...")
    csd_conf = valid_mapped.set_index('csd_uid')['mapping_confidence'].to_dict()
    csd_plot_gdf['confidence'] = csd_plot_gdf['CSDUID'].map(csd_conf).fillna("none")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#ffffff")
    
    # Base
    csd_plot_gdf[csd_plot_gdf['confidence'] == 'none'].plot(
        ax=ax, facecolor="#f5f5f5", edgecolor="#e5e5e5", linewidth=0.03
    )
    # High confidence (coordinate-based) in blue
    csd_plot_gdf[csd_plot_gdf['confidence'] == 'high'].plot(
        ax=ax, facecolor="#1f77b4", edgecolor="#17becf", linewidth=0.1
    )
    # Medium confidence (name-based) in orange
    csd_plot_gdf[csd_plot_gdf['confidence'] == 'medium'].plot(
        ax=ax, facecolor="#ff7f0e", edgecolor="#ffbb78", linewidth=0.1
    )
    
    # Custom legend proxy artists
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='#17becf', label='High Confidence (Coordinate-based)'),
        Patch(facecolor='#ff7f0e', edgecolor='#ffbb78', label='Medium Confidence (Name-based)'),
        Patch(facecolor='#f5f5f5', edgecolor='#e5e5e5', label='No Outbreaks')
    ]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.01, 0.05), fontsize=9)
    
    ax.set_title("Canada Census Subdivision (CSD) Mapping Confidences\n(CRS: EPSG:3347)", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlim(bounds[0] - 50000, bounds[2] + 50000)
    ax.set_ylim(bounds[1] - 50000, bounds[3] + 50000)
    
    conf_map_path = os.path.join(qa_maps_dir, "csd_mapping_confidence.png")
    plt.savefig(conf_map_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence map to {conf_map_path}")
    
    # Plot 3: private_cfia_points_to_csd_check.png
    print("Plotting private coordinates check map...")
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#ffffff")
    
    # Base outlines
    csd_plot_gdf.plot(
        ax=ax, facecolor='none', edgecolor='#cccccc', linewidth=0.03
    )
    # Plot actual coordinates
    if len(coord_df) > 0:
        coord_pts_gdf = gpd.GeoDataFrame(coord_df, geometry=gpd.points_from_xy(coord_df['longitude'], coord_df['latitude']), crs="EPSG:4326")
        coord_pts_gdf = coord_pts_gdf.to_crs("EPSG:3347")
        coord_pts_gdf.plot(
            ax=ax, color='#d62728', marker='o', markersize=3, alpha=0.6, label='Raw Outbreak Points'
        )
        ax.legend(loc='lower left', bbox_to_anchor=(0.01, 0.05), fontsize=9)
        
    ax.set_title("PRIVATE_DO_NOT_SHARE - CFIA Outbreak Points to CSD Check\n(CRS: EPSG:3347)", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlim(bounds[0] - 50000, bounds[2] + 50000)
    ax.set_ylim(bounds[1] - 50000, bounds[3] + 50000)
    
    pts_map_path = os.path.join(qa_maps_dir, "private_cfia_points_to_csd_check.png")
    plt.savefig(pts_map_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved private points check map to {pts_map_path}")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    outbreak_f = os.path.join(base_dir, "data", "processed", "cfia", "cleaned_poultry_outbreak_events.csv")
    csd_boundary_f = os.path.join(base_dir, "data", "raw/boundaries/lcsd000b21a_e.shp")
    lookup_f = os.path.join(base_dir, "data", "processed", "boundaries", "census_subdivision_lookup.csv")
    cd_boundary_f = os.path.join(base_dir, "data", "processed", "boundaries", "census_divisions_epsg3347.gpkg")
    out_dir = os.path.join(base_dir, "data", "processed")
    
    map_outbreaks_to_csd(outbreak_f, csd_boundary_f, lookup_f, cd_boundary_f, out_dir)
