import os
import sys
import urllib.request
import urllib.parse
import json
import pandas as pd

def download_pilot_data(output_path):
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
    
    print("Starting GBIF Ontario 2022 Waterfowl Pilot Downloader...")
    print("==========================================================")
    
    all_records = []
    record_counts = {}
    
    for species in target_species:
        print(f"\nProcessing species: '{species}'")
        group = species_group_mapping[species]
        
        # 1. Get usageKey
        encoded_name = urllib.parse.quote(species)
        match_url = f"https://api.gbif.org/v1/species/match?name={encoded_name}"
        
        try:
            with urllib.request.urlopen(match_url) as response:
                match_data = json.loads(response.read().decode('utf-8'))
                
            usage_key = match_data.get('usageKey')
            if not usage_key:
                print(f"  [ERROR] No usageKey returned for '{species}'")
                continue
                
            print(f"  - usageKey resolved: {usage_key}")
            
            # 2. Query occurrences for Ontario, CA, 2022 with coordinates
            search_params = {
                'taxonKey': usage_key,
                'country': 'CA',
                'stateProvince': 'Ontario',
                'year': 2022,
                'hasCoordinate': 'true',
                'limit': 300
            }
            encoded_params = urllib.parse.urlencode(search_params)
            search_url = f"https://api.gbif.org/v1/occurrence/search?{encoded_params}"
            
            req = urllib.request.Request(
                search_url,
                headers={'User-Agent': 'HPAI-CSD-Graph-Project/1.0 (mkhos@example.com)'}
            )
            
            with urllib.request.urlopen(req) as response:
                search_data = json.loads(response.read().decode('utf-8'))
                
            results = search_data.get('results', [])
            print(f"  - Downloaded {len(results)} occurrences from API")
            
            species_records = []
            for r in results:
                # Local verification of stateProvince to be safe
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
                
            print(f"  - Kept {len(species_records)} records after local stateProvince check")
            record_counts[species] = len(species_records)
            all_records.extend(species_records)
            
        except Exception as e:
            print(f"  [ERROR] Failed during extraction for '{species}': {e}")
            record_counts[species] = 0
            
    print("\n==========================================================")
    print("Download Summary:")
    for species, count in record_counts.items():
        print(f"  - {species}: {count} records")
    print(f"Total records combined: {len(all_records)}")
    
    if len(all_records) == 0:
        print("No records retrieved. Output file was not created.")
        sys.exit(1)
        
    # Save combined records to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(all_records)
    
    # Reorder columns to match instructions exactly
    columns_order = [
        'species_group', 'scientificName', 'taxonKey', 'decimalLatitude',
        'decimalLongitude', 'eventDate', 'year', 'month', 'countryCode',
        'stateProvince', 'gbifID'
    ]
    df = df[columns_order]
    
    df.to_csv(output_path, index=False)
    print(f"Successfully saved combined records to: {output_path}")

if __name__ == '__main__':
    default_output = 'data/raw/gbif/ontario_2022_waterfowl_pilot.csv'
    output_file = sys.argv[1] if len(sys.argv) > 1 else default_output
    download_pilot_data(output_file)
