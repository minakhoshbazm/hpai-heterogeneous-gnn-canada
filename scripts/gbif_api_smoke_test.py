import urllib.request
import urllib.parse
import json

def run_smoke_test():
    test_species = ["Branta canadensis", "Anas platyrhynchos"]
    
    print("Starting GBIF API Smoke Test...")
    print("==================================================")
    
    for species in test_species:
        print(f"\nProcessing species: '{species}'")
        
        # 1. Species Match Query
        encoded_name = urllib.parse.quote(species)
        match_url = f"https://api.gbif.org/v1/species/match?name={encoded_name}"
        
        try:
            with urllib.request.urlopen(match_url) as response:
                match_data = json.loads(response.read().decode('utf-8'))
                
            usage_key = match_data.get('usageKey')
            match_name = match_data.get('scientificName')
            match_status = match_data.get('status')
            
            print(f"  - Match URL: {match_url}")
            print(f"  - usageKey: {usage_key} ({match_name}, status={match_status})")
            
            if not usage_key:
                print(f"  [WARNING] No usageKey returned for species '{species}'")
                continue
                
            # 2. Occurrence Search Query
            search_params = {
                'taxonKey': usage_key,
                'country': 'CA',
                'year': 2022,
                'limit': 5
            }
            encoded_params = urllib.parse.urlencode(search_params)
            search_url = f"https://api.gbif.org/v1/occurrence/search?{encoded_params}"
            
            req = urllib.request.Request(
                search_url, 
                headers={'User-Agent': 'HPAI-CSD-Graph-Project/1.0 (mkhos@example.com)'}
            )
            
            with urllib.request.urlopen(req) as response:
                search_data = json.loads(response.read().decode('utf-8'))
                
            total_records = search_data.get('count', 0)
            results = search_data.get('results', [])
            
            print(f"  - Search URL: {search_url}")
            print(f"  - Total records found in CA for 2022: {total_records}")
            print(f"  - Showing first {len(results)} records:")
            
            fields_to_show = [
                'scientificName', 'decimalLatitude', 'decimalLongitude', 
                'eventDate', 'countryCode', 'stateProvince'
            ]
            
            for idx, rec in enumerate(results):
                print(f"    * Record {idx + 1}:")
                for field in fields_to_show:
                    val = rec.get(field, 'N/A')
                    print(f"      {field}: {val}")
                    
        except Exception as e:
            print(f"  [ERROR] Failed to query GBIF API for '{species}': {e}")
            
    print("\n==================================================")
    print("GBIF API Smoke Test Completed.")

if __name__ == '__main__':
    run_smoke_test()
