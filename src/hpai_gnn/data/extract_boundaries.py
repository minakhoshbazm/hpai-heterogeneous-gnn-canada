import os
import sys
import struct
import urllib.request
# Disable OGR size limit for complex GeoJSON operations
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

def parse_dbf_attributes(dbf_path):
    """Parses attributes from a shapefile .dbf file directly."""
    with open(dbf_path, 'rb') as f:
        header_data = f.read(32)
        if len(header_data) < 32:
            return None
        header_len = struct.unpack('<H', header_data[8:10])[0]
        record_len = struct.unpack('<H', header_data[10:12])[0]
        num_fields = (header_len - 33) // 32
        
        fields = []
        for i in range(num_fields):
            field_data = f.read(32)
            name = field_data[0:11].decode('ascii', errors='ignore').strip('\x00').strip()
            field_type = chr(field_data[11])
            field_len = field_data[16]
            fields.append((name, field_type, field_len))
            
        f.seek(header_len)
        records = []
        while True:
            record_status = f.read(1)
            if not record_status:
                break
            record_values = {}
            for name, field_type, field_len in fields:
                val_bytes = f.read(field_len)
                val_str = val_bytes.decode('utf-8', errors='ignore').strip()
                record_values[name] = val_str
            records.append(record_values)
        import pandas as pd
        return pd.DataFrame(records)

def process_boundaries(raw_shp_path, output_dir):
    """Runs the boundary aggregation, name-mapping, CRS-transformation, and lookup extraction."""
    try:
        import geopandas as gpd
        import pandas as pd
        import shapely
    except ImportError:
        print("Error: 'geopandas', 'pandas', and 'shapely' are required to run this script.")
        print("Please install them using: pip install geopandas pandas shapely")
        sys.exit(1)
        
    print(f"Reading raw CSD boundaries from: {raw_shp_path}")
    if not os.path.exists(raw_shp_path):
        print(f"Error: Raw shapefile not found at {raw_shp_path}")
        sys.exit(1)
        
    # Read raw CSD shapefile
    csd_gdf = gpd.read_file(raw_shp_path)
    print(f"Loaded {len(csd_gdf)} Census Subdivisions.")
    
    # 1. Generate local lookup files from DBF attributes
    dbf_path = raw_shp_path.replace(".shp", ".dbf")
    print(f"Parsing CSD attributes from DBF: {dbf_path}")
    dbf_df = parse_dbf_attributes(dbf_path)
    
    if dbf_df is not None:
        dbf_df['csd_uid'] = dbf_df['CSDUID'].str.strip()
        dbf_df['csd_name'] = dbf_df['CSDNAME'].str.strip()
        dbf_df['csd_type'] = dbf_df['CSDTYPE'].str.strip()
        dbf_df['pruid'] = dbf_df['PRUID'].str.strip()
        dbf_df['land_area'] = pd.to_numeric(dbf_df['LANDAREA'], errors='coerce').fillna(0.0)
        dbf_df['cd_uid'] = dbf_df['csd_uid'].str[:4]
        
        # Save lookups
        os.makedirs(output_dir, exist_ok=True)
        crosswalk_path = os.path.join(output_dir, "csd_to_cd_crosswalk.csv")
        lookup_path = os.path.join(output_dir, "census_subdivision_lookup.csv")
        
        dbf_df[['csd_uid', 'cd_uid']].to_csv(crosswalk_path, index=False)
        dbf_df[['csd_uid', 'csd_name', 'csd_type', 'cd_uid', 'pruid', 'land_area']].to_csv(lookup_path, index=False)
        print(f"Saved lookups to:\n  - {crosswalk_path}\n  - {lookup_path}")
    
    # 2. Extract CDUID and dissolve geometries
    csd_gdf['CDUID'] = csd_gdf['CSDUID'].str[:4]
    csd_gdf['LANDAREA'] = pd.to_numeric(csd_gdf['LANDAREA'], errors='coerce').fillna(0.0)
    
    print("Dissolving CSD geometries into Census Divisions (CD)...")
    cd_gdf = csd_gdf.dissolve(
        by='CDUID', 
        aggfunc={
            'LANDAREA': 'sum',
            'PRUID': 'first'
        }
    ).reset_index()
    
    # 3. Fetch CDNAMES & PRNAMES mapping
    print("Downloading CDUID-to-CDNAME mapping list...")
    url = "https://raw.githubusercontent.com/minnzc/Canada_Daily_Climate/main/daily_cd_climate_2020.csv"
    try:
        geo_df = pd.read_csv(url, usecols=['PRUID', 'PRNAME', 'CDUID', 'CDNAME'])
        geo_df['CDUID'] = geo_df['CDUID'].astype(str).str.zfill(4)
        geo_df['PRUID'] = geo_df['PRUID'].astype(str).str.zfill(2)
        mapping = geo_df.drop_duplicates(subset=['CDUID']).copy()
    except Exception as e:
        print(f"Warning: Failed to fetch online names mapping ({e}). Falling back to placeholders.")
        mapping = pd.DataFrame(columns=['CDUID', 'CDNAME', 'PRUID', 'PRNAME'])
        
    cd_gdf['CDUID'] = cd_gdf['CDUID'].astype(str).str.zfill(4)
    merged = cd_gdf.merge(mapping, on='CDUID', how='left')
    
    # Map province details
    PR_NAMES_EN = {
        '10': 'Newfoundland and Labrador', '11': 'Prince Edward Island', '12': 'Nova Scotia',
        '13': 'New Brunswick', '24': 'Quebec', '35': 'Ontario', '46': 'Manitoba',
        '47': 'Saskatchewan', '48': 'Alberta', '59': 'British Columbia', '60': 'Yukon',
        '61': 'Northwest Territories', '62': 'Nunavut'
    }
    
    PR_NAMES_BILINGUAL = {
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
    
    merged['cd_uid'] = merged['CDUID']
    merged['cd_name'] = merged['CDNAME'].fillna("Unknown")
    merged['province_code'] = merged['PRUID_x'].astype(str).str.zfill(2)
    merged['province_name_en'] = merged['province_code'].map(PR_NAMES_EN).fillna("Unknown")
    merged['province_name_bilingual'] = merged['province_code'].map(PR_NAMES_BILINGUAL).fillna("Unknown")
    
    # Synonyms for backward compatibility
    merged['province_id'] = merged['province_code']
    merged['province_name'] = merged['province_name_en']
    merged['land_area'] = merged['LANDAREA']
    
    for idx, row in merged.iterrows():
        if row['cd_name'] == "Unknown":
            merged.at[idx, 'cd_name'] = f"Division No. {row['cd_uid'][2:]}"
            
    # Save a QA table showing unique values before and after standardization
    qa_records = []
    for idx, row in merged.iterrows():
        qa_records.append({
            'raw_province_id': row['PRUID_x'],
            'raw_province_name': row['PRNAME'],
            'standardized_province_code': row['province_code'],
            'standardized_province_name_en': row['province_name_en'],
            'standardized_province_name_bilingual': row['province_name_bilingual']
        })
    qa_df = pd.DataFrame(qa_records).drop_duplicates().sort_values('standardized_province_code')
    
    # Save QA Table to results/maps/province_name_standardization_audit.csv
    maps_dir = os.path.abspath(os.path.join(output_dir, "..", "..", "..", "results", "maps"))
    os.makedirs(maps_dir, exist_ok=True)
    qa_path = os.path.join(maps_dir, "province_name_standardization_audit.csv")
    qa_df.to_csv(qa_path, index=False)
    print(f"Saved province name standardization QA audit to: {qa_path}")
            
    final_cols = [
        'cd_uid', 'cd_name', 
        'province_code', 'province_name_en', 'province_name_bilingual', 
        'province_id', 'province_name', 'land_area', 'geometry'
    ]
    clean_gdf = merged[final_cols].copy()
    
    # 4. Project and Save Boundaries with CRS Validations
    gpkg_path = os.path.join(output_dir, "census_divisions_epsg3347.gpkg")
    geojson_path = os.path.join(output_dir, "census_divisions_wgs84.geojson")
    
    # Web/export CRS: EPSG:4326 (WGS84)
    print("Simplifying boundaries for Web/export CRS to prevent out-of-memory errors...")
    web_clean_gdf = clean_gdf.copy()
    web_clean_gdf['geometry'] = web_clean_gdf.geometry.simplify(100, preserve_topology=True)
    print("Reprojecting boundaries to EPSG:4326...")
    gdf_wgs84 = web_clean_gdf.to_crs("EPSG:4326")
    
    # Analysis/vector CRS: EPSG:3347 (NAD83 / Statistics Canada Lambert)
    print("Reprojecting boundaries to EPSG:3347...")
    gdf_epsg3347 = clean_gdf.to_crs("EPSG:3347")
    
    # CRS Validation checks
    print("Running CRS validation checks...")
    assert gdf_wgs84.crs.to_epsg() == 4326, f"Expected EPSG:4326, got {gdf_wgs84.crs}"
    assert gdf_epsg3347.crs.to_epsg() == 3347, f"Expected EPSG:3347, got {gdf_epsg3347.crs}"
    print("CRS validation passed.")
    
    print(f"Saving analysis boundary to: {gpkg_path}")
    gdf_epsg3347.to_file(gpkg_path, driver="GPKG")
    
    print(f"Saving web sharing boundary to: {geojson_path}")
    gdf_wgs84.to_file(geojson_path, driver="GeoJSON")
    
    print("Boundary processing pipeline completed successfully.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_path = os.path.join(base_dir, "data", "raw", "boundaries", "lcsd000b21a_e.shp")
    out_dir = os.path.join(base_dir, "data", "processed", "boundaries")
    
    process_boundaries(raw_path, out_dir)
