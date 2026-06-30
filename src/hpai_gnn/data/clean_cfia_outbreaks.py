import os
import sys
import pandas as pd
# Disable OGR size limit for complex GeoJSON operations
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

def clean_and_map_outbreaks(raw_cfia_path, boundaries_gpkg_path, csd_lookup_path, output_dir):
    """Runs the CFIA cleaning, spatial joins, deduplication, and label shifting pipeline."""
    try:
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import Point
    except ImportError:
        print("Error: 'geopandas', 'numpy', and 'shapely' are required to run this script.")
        print("Please install them using: pip install geopandas numpy shapely")
        sys.exit(1)
        
    print(f"Loading raw CFIA data from: {raw_cfia_path}")
    if not os.path.exists(raw_cfia_path):
        print(f"Error: Raw CFIA file not found at {raw_cfia_path}")
        sys.exit(1)
        
    # Load raw CFIA CSV with latin-1 encoding
    df = pd.read_csv(raw_cfia_path, encoding='latin-1')
    print(f"Loaded {len(df)} raw outbreak records.")
    
    # 1. Standardize column names (mapping to snake_case while preserving original data)
    rename_map = {
        'CFIA Premises ID': 'premises_id',
        'Wave': 'wave',
        'Pathogenicity': 'pathogenicity',
        'Repeat Detection': 'repeat_detection',
        'Regulation Type': 'regulation_type',
        'WOAH Classification': 'woah_classification',
        'Species - Production Type': 'species_production_type',
        'City': 'city',
        'Province': 'province',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Date of First Clinical Signs': 'date_first_clinical_signs',
        'CFIA Notification Date': 'cfia_notification_date',
        'CAHSN Lab Name': 'cahsn_lab_name',
        'CAHSN Laboratory #': 'cahsn_lab_num',
        'Sample Submission Date to CAHSN': 'cahsn_submission_date',
        'CAHSN Confirmation Date': 'cahsn_confirmation_date',
        'LSTS System ID': 'lsts_system_id',
        'Sample Submission Date to NCFAD': 'ncfad_submission_date',
        'NCFAD Confirmation Date': 'ncfad_confirmation_date',
        'NCFAD LSTS Job Number': 'ncfad_job_num',
        'Clade': 'clade',
        'Cluster': 'cluster',
        'Lineage': 'lineage',
        'Reference Virus': 'reference_virus',
        'Method of IP Identification': 'ip_identification_method',
        'Number of Barns': 'num_barns',
        'Number of Birds on Site': 'num_birds_on_site',
        'Number of Birds in Infected Barn': 'num_birds_in_infected_barn',
        'Number of Susceptible Birds': 'num_birds_susceptible',
        'Number Dead (Mortality)': 'num_dead',
        'Number of Cases': 'num_cases',
        'Number of Birds Destroyed': 'num_birds_destroyed',
        'Control Zone': 'control_zone',
        'Control Zone Status': 'control_zone_status',
        'PCZ Declared Date': 'pcz_declared_date',
        'Depopulation Date of Infected Barn': 'depopulation_date',
        'Date of Destruction Completion': 'destruction_completion_date',
        'Disposal Method': 'disposal_method',
        'Date Compost Piles Built and Capped': 'compost_piles_built_date',
        'Date of Disposal Completion': 'disposal_completion_date',
        'Primary Decontamination Completion Date': 'primary_decon_completion_date',
        'Date of C&D Approval': 'cd_approval_date',
        'PCZ Revoked Date': 'pcz_revoked_date',
        'Date of Restriction Removal on Infected Premises': 'restriction_removal_date'
    }
    
    # Store standard names, keeping all original columns as well
    clean_df = df.copy()
    for raw_col, std_col in rename_map.items():
        if raw_col in clean_df.columns:
            clean_df[std_col] = clean_df[raw_col]
            
    # 2. Parse dates and prioritize to select event_date
    print("Parsing dates and applying prioritization...")
    date_cols = ['ncfad_confirmation_date', 'cahsn_confirmation_date', 'cfia_notification_date', 'date_first_clinical_signs']
    for col in date_cols:
        clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce')
        
    def select_event_date(row):
        if not pd.isna(row['ncfad_confirmation_date']):
            return row['ncfad_confirmation_date']
        if not pd.isna(row['cahsn_confirmation_date']):
            return row['cahsn_confirmation_date']
        if not pd.isna(row['cfia_notification_date']):
            return row['cfia_notification_date']
        if not pd.isna(row['date_first_clinical_signs']):
            return row['date_first_clinical_signs']
        return pd.NaT

    clean_df['event_date'] = clean_df.apply(select_event_date, axis=1)
    
    # Filter rows with missing event dates to issue tracking
    no_date_mask = clean_df['event_date'].isna()
    issues_no_date = clean_df[no_date_mask].copy()
    issues_no_date['issue_reason'] = "missing all date columns"
    
    clean_df = clean_df[~no_date_mask].copy()
    clean_df['event_year'] = clean_df['event_date'].dt.year
    clean_df['event_month'] = clean_df['event_date'].dt.strftime('%Y-%m')
    
    # 3. Filter poultry/domestic outbreaks separately from wild bird detections
    # Captive raptors are wild bird events in this context
    wild_mask = clean_df['species_production_type'].str.contains('wild', case=False, na=False)
    wild_events = clean_df[wild_mask].copy()
    poultry_events = clean_df[~wild_mask].copy()
    
    print(f"Filtered {len(poultry_events)} poultry events and {len(wild_events)} wild bird events.")
    
    # 4. Standardize Province names
    PR_NAMES = {
        'AB': 'Alberta', 'BC': 'British Columbia', 'MB': 'Manitoba',
        'NB': 'New Brunswick', 'NL': 'Newfoundland and Labrador', 'NS': 'Nova Scotia',
        'ON': 'Ontario', 'PE': 'Prince Edward Island', 'QC': 'Quebec', 'SK': 'Saskatchewan',
        'YT': 'Yukon', 'NT': 'Northwest Territories', 'NU': 'Nunavut'
    }
    PR_NAME_TO_CODE = {
        'Newfoundland and Labrador': '10', 'Prince Edward Island': '11', 'Nova Scotia': '12',
        'New Brunswick': '13', 'Quebec': '24', 'Ontario': '35', 'Manitoba': '46',
        'Saskatchewan': '47', 'Alberta': '48', 'British Columbia': '59', 'Yukon': '60',
        'Northwest Territories': '61', 'Nunavut': '62'
    }
    
    PR_CODE_TO_BILINGUAL = {
        '10': 'Newfoundland and Labrador / Terre-Neuve-et-Labrador',
        '11': 'Prince Edward Island / Île-du-Prince-Édouard',
        '12': 'Nova Scotia / Nouvelle-Écosse',
        '13': 'New Brunswick / Nouveau-Brunswick',
        '24': 'Quebec / Québec',
        '35': 'Ontario / Ontario',
        '46': 'Manitoba / Manitoba',
        '47': 'Saskatchewan / Saskatchewan',
        '48': 'Alberta / Alberta',
        '59': 'British Columbia / Colombie-Britannique',
        '60': 'Yukon / Yukon',
        '61': 'Northwest Territories / Territoires du Nord-Ouest',
        '62': 'Nunavut / Nunavut'
    }
    
    PR_NUM_TO_NAME = {k: v for v, k in PR_NAME_TO_CODE.items()}
    poultry_events['province'] = poultry_events['province'].str.strip().str.upper().map(lambda x: PR_NAMES.get(x, x))
    
    # 5. Spatial Mapping Pipeline
    print("Performing spatial mapping...")
    poultry_events['cd_uid'] = None
    poultry_events['cd_name'] = None
    poultry_events['province_code'] = None
    poultry_events['province_name_en'] = None
    poultry_events['province_name_bilingual'] = None
    poultry_events['province_name'] = None
    poultry_events['mapping_confidence'] = "failed"
    
    # Tier 1: Spatial Join (High Confidence)
    has_coords_mask = poultry_events['latitude'].notna() & poultry_events['longitude'].notna()
    coord_events = poultry_events[has_coords_mask].copy()
    
    if len(coord_events) > 0 and os.path.exists(boundaries_gpkg_path):
        print(f"Loading boundary layer for spatial join from: {boundaries_gpkg_path}")
        cd_gdf = gpd.read_file(boundaries_gpkg_path)
        
        # Verify boundary CRS is EPSG:3347
        assert cd_gdf.crs.to_epsg() == 3347, f"Expected boundary CRS to be 3347, got {cd_gdf.crs}"
        
        # Create Point geometries in WGS84 (EPSG:4326)
        geometry = [Point(xy) for xy in zip(coord_events['longitude'], coord_events['latitude'])]
        
        # Drop columns that clash with boundary attributes to prevent sjoin suffix renaming
        clash_cols = ['cd_uid', 'cd_name', 'province_code', 'province_name_en', 'province_name_bilingual', 'province_id', 'province_name']
        coord_events_clean = coord_events.drop(columns=[c for c in clash_cols if c in coord_events.columns])
        
        points_gdf = gpd.GeoDataFrame(coord_events_clean, geometry=geometry, crs="EPSG:4326")
        
        # Reproject points to EPSG:3347
        points_gdf = points_gdf.to_crs("EPSG:3347")
        
        # Validate projection matches before join
        assert points_gdf.crs == cd_gdf.crs, f"Mismatched projections: points={points_gdf.crs}, boundaries={cd_gdf.crs}"
        
        # Spatial join (points inside polygons)
        joined = gpd.sjoin(points_gdf, cd_gdf, how="left", predicate="intersects")
        
        # Assign matched CDUIDs
        poultry_events.loc[has_coords_mask, 'cd_uid'] = joined['cd_uid']
        poultry_events.loc[has_coords_mask, 'cd_name'] = joined['cd_name']
        poultry_events.loc[has_coords_mask, 'province_code'] = joined['province_code']
        poultry_events.loc[has_coords_mask, 'province_name_en'] = joined['province_name_en']
        poultry_events.loc[has_coords_mask, 'province_name_bilingual'] = joined['province_name_bilingual']
        poultry_events.loc[has_coords_mask, 'province_name'] = joined['province_name_en']
        poultry_events.loc[has_coords_mask, 'mapping_confidence'] = joined['cd_uid'].apply(
            lambda x: "high" if not pd.isna(x) else "failed"
        )
    
    # Tier 2: Municipality / CSD Lookup fallback (Medium Confidence)
    no_spatial_match = poultry_events['mapping_confidence'] == "failed"
    print(f"Failed spatial joins: {no_spatial_match.sum()}")
    
    if no_spatial_match.sum() > 0 and os.path.exists(csd_lookup_path):
        print(f"Using subdivision lookup fallback from: {csd_lookup_path}")
        csd_lookup = pd.read_csv(csd_lookup_path)
        csd_lookup['csd_name_norm'] = csd_lookup['csd_name'].str.lower().str.strip()
        
        # Map province code to standard name for lookup comparison
        csd_lookup['province_name_en'] = csd_lookup['pruid'].astype(str).zfill(2).map(PR_NUM_TO_NAME)
        
        # Get parent boundaries for CD Name lookup
        cd_gdf = gpd.read_file(boundaries_gpkg_path)
        cd_names_map = cd_gdf.set_index('cd_uid')['cd_name'].to_dict()
        
        for idx, row in poultry_events[no_spatial_match].iterrows():
            city = str(row['city']).strip().lower()
            prov = str(row['province']).strip()
            
            # Find in lookup matching both subdivision name and province
            match = csd_lookup[(csd_lookup['csd_name_norm'] == city) & (csd_lookup['province_name_en'] == prov)]
            if len(match) > 0:
                cduid = str(match.iloc[0]['cd_uid']).zfill(4)
                pcode = str(match.iloc[0]['pruid']).zfill(2)
                poultry_events.at[idx, 'cd_uid'] = cduid
                poultry_events.at[idx, 'cd_name'] = cd_names_map.get(cduid, "Unknown")
                poultry_events.at[idx, 'province_code'] = pcode
                poultry_events.at[idx, 'province_name_en'] = prov
                poultry_events.at[idx, 'province_name_bilingual'] = PR_CODE_TO_BILINGUAL.get(pcode, prov)
                poultry_events.at[idx, 'province_name'] = prov
                poultry_events.at[idx, 'mapping_confidence'] = "medium"
                
    # Tier 3: Province Fallback / Failed (Low Confidence)
    failed_mask = poultry_events['mapping_confidence'] == "failed"
    for idx, row in poultry_events[failed_mask].iterrows():
        # Low confidence fallback if province matches
        prov = str(row['province']).strip()
        if prov in PR_NAME_TO_CODE:
            pcode = PR_NAME_TO_CODE[prov]
            poultry_events.at[idx, 'province_code'] = pcode
            poultry_events.at[idx, 'province_name_en'] = prov
            poultry_events.at[idx, 'province_name_bilingual'] = PR_CODE_TO_BILINGUAL.get(pcode, prov)
            poultry_events.at[idx, 'province_name'] = prov
            poultry_events.at[idx, 'mapping_confidence'] = "low"
            
    # Log failed mappings to issues table
    issues_failed = poultry_events[poultry_events['mapping_confidence'] == "failed"].copy()
    issues_failed['issue_reason'] = "spatial join and text lookup failed"
    
    # 6. Deduplication Strategy
    print("Running deduplication...")
    # Group by premises ID and month, or within close geographic proximity in the same month
    poultry_events = poultry_events.sort_values(by='event_date')
    
    # Duplicate logic: Same premises ID in same month, or same rounded lat/lon (+/- 0.01 degree) in same month
    poultry_events['lat_round'] = poultry_events['latitude'].round(2)
    poultry_events['lon_round'] = poultry_events['longitude'].round(2)
    
    dup_mask_premises = poultry_events.duplicated(subset=['premises_id', 'event_month'], keep='first') & poultry_events['premises_id'].notna()
    dup_mask_coords = poultry_events.duplicated(subset=['lat_round', 'lon_round', 'event_month'], keep='first') & poultry_events['latitude'].notna()
    
    dup_mask = dup_mask_premises | dup_mask_coords
    
    issues_duplicates = poultry_events[dup_mask].copy()
    issues_duplicates['issue_reason'] = "potential duplicate record"
    
    cleaned_poultry = poultry_events[~dup_mask].copy()
    print(f"Deduplicated events. Retained {len(cleaned_poultry)} unique outbreaks.")
    
    # Combine issues
    issues_df = pd.concat([issues_no_date, issues_failed, issues_duplicates], ignore_index=True)
    
    # 7. Generate Outputs
    print("Writing output tables...")
    os.makedirs(os.path.join(output_dir, "cfia"), exist_ok=True)
    
    # File 1: cleaned_cfia_events.csv (all cleaned records)
    clean_df.to_csv(os.path.join(output_dir, "cfia", "cleaned_cfia_events.csv"), index=False)
    
    # File 2: cleaned_poultry_outbreak_events.csv
    cleaned_poultry.to_csv(os.path.join(output_dir, "cfia", "cleaned_poultry_outbreak_events.csv"), index=False)
    
    # File 3: cleaned_wild_bird_events.csv
    wild_events.to_csv(os.path.join(output_dir, "cfia", "cleaned_wild_bird_events.csv"), index=False)
    
    # File 4: event_to_cd_crosswalk.csv
    crosswalk_cols = [
        'premises_id', 'cd_uid', 'cd_name', 
        'province_code', 'province_name_en', 'province_name_bilingual', 'province_name', 
        'event_month', 'mapping_confidence'
    ]
    cleaned_poultry[crosswalk_cols].to_csv(os.path.join(output_dir, "cfia", "event_to_cd_crosswalk.csv"), index=False)
    
    # File 5: cfia_cleaning_issues.csv
    issues_df.to_csv(os.path.join(output_dir, "cfia", "cfia_cleaning_issues.csv"), index=False)
    
    # 8. Create complete label grid of all 293 CDs and all months
    # Get all active CDs
    cd_gdf = gpd.read_file(boundaries_gpkg_path)
    all_cd_uids = cd_gdf['cd_uid'].unique()
    
    # Generate complete list of months from data
    unique_months = sorted(cleaned_poultry['event_month'].unique())
    if not unique_months:
        unique_months = ["2022-01"]
        
    start_date = pd.to_datetime(unique_months[0] + "-01")
    end_date = pd.to_datetime(unique_months[-1] + "-01")
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
    
    # Create Cartesian Product
    grid = pd.MultiIndex.from_product([all_cd_uids, all_months], names=['cd_uid', 'month']).to_frame().reset_index(drop=True)
    
    # Merge CD details
    cd_details = cd_gdf[['cd_uid', 'cd_name', 'province_code', 'province_name_en', 'province_name_bilingual', 'province_name']].copy()
    grid = grid.merge(cd_details, on='cd_uid', how='left')
    
    # Aggregate outbreaks per CD and month
    # Calculate counts by confidence type
    agg_df = cleaned_poultry.groupby(['cd_uid', 'event_month']).agg(
        outbreak_count=('premises_id', 'count'),
        num_high=('mapping_confidence', lambda x: (x == 'high').sum()),
        num_medium=('mapping_confidence', lambda x: (x == 'medium').sum()),
        num_low=('mapping_confidence', lambda x: (x == 'low').sum())
    ).reset_index().rename(columns={'event_month': 'month'})
    
    # Merge aggregations into the full grid (filling NaNs with 0)
    grid = grid.merge(agg_df, on=['cd_uid', 'month'], how='left')
    grid['outbreak_count'] = grid['outbreak_count'].fillna(0).astype(int)
    grid['outbreak_binary'] = (grid['outbreak_count'] > 0).astype(int)
    grid['number_high_confidence_events'] = grid['num_high'].fillna(0).astype(int)
    grid['number_medium_confidence_events'] = grid['num_medium'].fillna(0).astype(int)
    grid['number_low_confidence_events'] = grid['num_low'].fillna(0).astype(int)
    
    # Drop temp cols
    grid = grid.drop(columns=['num_high', 'num_medium', 'num_low'])
    
    # File 6: cd_month_outbreak_counts.csv
    counts_path = os.path.join(output_dir, "cfia", "cd_month_outbreak_counts.csv")
    grid.to_csv(counts_path, index=False)
    print(f"Saved outbreak counts grid to {counts_path}")
    
    # 9. Create labels table with next-month shifted variables
    # Sort grid to ensure correct temporal shifts
    labels_df = grid.sort_values(by=['cd_uid', 'month']).copy()
    
    # Compute next-month labels within each CDUID group to avoid spatial leakage
    labels_df['outbreak_count_current_month'] = labels_df['outbreak_count']
    labels_df['outbreak_binary_current_month'] = labels_df['outbreak_binary']
    
    # Shift counts back by 1 step (-1) to represent next month's values
    labels_df['outbreak_count_next_month'] = labels_df.groupby('cd_uid')['outbreak_count'].shift(-1)
    labels_df['outbreak_next_month'] = labels_df.groupby('cd_uid')['outbreak_binary'].shift(-1)
    
    # For the last month in the sequence, the shifted target is NaN (which is correct as we cannot see the future)
    # We will preserve NaNs to avoid false zeros
    
    # File 7: cd_month_labels.csv
    labels_cols = [
        'cd_uid', 'cd_name', 'province_code', 'province_name_en', 'province_name_bilingual', 'province_name', 'month',
        'outbreak_count_current_month', 'outbreak_binary_current_month',
        'outbreak_count_next_month', 'outbreak_next_month'
    ]
    labels_path = os.path.join(output_dir, "cfia", "cd_month_labels.csv")
    labels_df[labels_cols].to_csv(labels_path, index=False)
    print(f"Saved outbreak shifted labels to {labels_path}")
    
    # Validation stats
    print("\nExtraction Summary:")
    print(f"  - Total poultry records processed: {len(cleaned_poultry)}")
    print(f"  - Mapped high confidence: {(cleaned_poultry['mapping_confidence'] == 'high').sum()}")
    print(f"  - Mapped medium confidence: {(cleaned_poultry['mapping_confidence'] == 'medium').sum()}")
    print(f"  - Mapped low confidence: {(cleaned_poultry['mapping_confidence'] == 'low').sum()}")
    print(f"  - Total issues logged: {len(issues_df)}")
    print("CFIA cleaning pipeline execution completed.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_cfia = os.path.join(base_dir, "data", "raw", "cfia", "CFIA_outbreaks.csv")
    gpkg_path = os.path.join(base_dir, "data", "processed", "boundaries", "census_divisions_epsg3347.gpkg")
    lookup_path = os.path.join(base_dir, "data", "processed", "boundaries", "census_subdivision_lookup.csv")
    out_dir = os.path.join(base_dir, "data", "processed")
    
    clean_and_map_outbreaks(raw_cfia, gpkg_path, lookup_path, out_dir)
