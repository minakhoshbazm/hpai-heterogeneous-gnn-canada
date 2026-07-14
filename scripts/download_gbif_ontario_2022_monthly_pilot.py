import os
import sys
import urllib.request
import urllib.parse
import json
import pandas as pd

def download_monthly_pilot_data(output_path):
    target_species = [
        "Branta canadensis",
        "Anser caerulescens",
        "Anas platyrhynchos",
        "Anas acuta",
        "Mareca americana",
        "Aythya affinis",
        "Aythya marila"
    ]
    
    species_group_mapping = {
        "Branta canadensis": "Goose",
        "Anser caerulescens": "Goose",
        "Anas platyrhynchos": "Duck",
        "Anas acuta": "Duck",
        "Mareca americana": "Duck",
        "Aythya affinis": "Duck",
        "Aythya marila": "Duck"
    }
    
    print("Starting Month-Balanced GBIF Ontario 2022 Downloader...")
    print("==========================================================")
    
    all_records = []
    
    # 1. Resolve keys first to minimize queries
    taxon_keys = {}
    for species in target_species:
        encoded_name = urllib.parse.quote(species)
        match_url = f"https://api.gbif.org/v1/species/match?name={encoded_name}"
        try:
            with urllib.request.urlopen(match_url) as response:
                match_data = json.loads(response.read().decode('utf-8'))
            usage_key = match_data.get('usageKey')
            if usage_key:
                taxon_keys[species] = usage_key
                print(f"Resolved '{species}' -> taxonKey {usage_key}")
        except Exception as e:
            print(f"Error matching '{species}': {e}")
            
    print("\nStarting monthly queries (7 species * 12 months = 84 queries)...")
    
    # Query each species and each month
    for month in range(1, 13):
        print(f"Querying Month {month}...")
        for species in target_species:
            usage_key = taxon_keys.get(species)
            if not usage_key:
                continue
                
            group = species_group_mapping[species]
            
            search_params = {
                'taxonKey': usage_key,
                'country': 'CA',
                'stateProvince': 'Ontario',
                'year': 2022,
                'month': month,
                'hasCoordinate': 'true',
                'limit': 50
            }
            encoded_params = urllib.parse.urlencode(search_params)
            search_url = f"https://api.gbif.org/v1/occurrence/search?{encoded_params}"
            
            req = urllib.request.Request(
                search_url,
                headers={'User-Agent': 'HPAI-CSD-Graph-Project/1.0 (mkhos@example.com)'}
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    search_data = json.loads(response.read().decode('utf-8'))
                results = search_data.get('results', [])
                
                species_records = []
                for r in results:
                    # Local verification of stateProvince
                    state = str(r.get('stateProvince', '')).lower()
                    if 'ontario' not in state:
                        continue
                        
                    rec_dict = {
                        'species_group': group,
                        'scientificName': r.get('scientificName'),
                        'taxonKey': usage_key,
                        'decimalLatitude': r.get('decimalLatitude'),
                        'decimalLongitude': r.get('decimalLongitude'),
                        'eventDate': r.get('eventDate'),
                        'year': r.get('year'),
                        'month': r.get('month'),
                        'countryCode': r.get('countryCode'),
                        'stateProvince': r.get('stateProvince'),
                        'gbifID': r.get('gbifID')
                    }
                    species_records.append(rec_dict)
                    
                all_records.extend(species_records)
                
            except Exception as e:
                print(f"  [ERROR] Failed for '{species}' in Month {month}: {e}")
                
    print("\n==========================================================")
    print(f"Total records combined: {len(all_records)}")
    
    if len(all_records) == 0:
        print("No records retrieved. Output file was not created.")
        sys.exit(1)
        
    df = pd.DataFrame(all_records)
    
    # Print metrics
    print("\nRecords by month:")
    month_counts = df['month'].value_counts().sort_index()
    for m, c in month_counts.items():
        print(f"  - Month {m}: {c} records")
        
    print("\nRecords by species:")
    species_counts = df['scientificName'].value_counts()
    for sp, c in species_counts.items():
        print(f"  - {sp}: {c} records")
        
    # Reorder columns
    columns_order = [
        'species_group', 'scientificName', 'taxonKey', 'decimalLatitude',
        'decimalLongitude', 'eventDate', 'year', 'month', 'countryCode',
        'stateProvince', 'gbifID'
    ]
    df = df[columns_order]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved combined records to: {output_path}")

if __name__ == '__main__':
    default_output = 'data/raw/gbif/ontario_2022_waterfowl_monthly_pilot.csv'
    output_file = sys.argv[1] if len(sys.argv) > 1 else default_output
    download_monthly_pilot_data(output_file)
