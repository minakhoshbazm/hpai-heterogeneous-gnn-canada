import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

# Directory to store raw eBird API responses
RAW_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'ebird_api'
RAW_DIR.mkdir(parents=True, exist_ok=True)

# List of Canadian region codes (province/territory) for eBird
CANADA_REGIONS = [
    'CA-AB', 'CA-BC', 'CA-MB', 'CA-NB', 'CA-NL', 'CA-NT',
    'CA-NS', 'CA-NU', 'CA-ON', 'CA-PE', 'CA-QC', 'CA-SK', 'CA-YT'
]

# eBird API endpoint template
ENDPOINT_TEMPLATE = "https://api.ebird.org/v2/data/obs/{region}/recent?maxResults=500&detail=full"

# Header with API key (never printed)
API_KEY = os.getenv('EBIRD_API_KEY')
if not API_KEY:
    raise RuntimeError('EBIRD_API_KEY environment variable not set')
HEADERS = {'X-eBirdApiToken': API_KEY}


def fetch_region_observations(region: str) -> list:
    """Fetch recent observations for a given eBird region.
    Returns a list of observation dicts (may be empty on error)."""
    url = ENDPOINT_TEMPLATE.format(region=region)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        # Record error for audit purposes
        error_record = {
            'region': region,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        # Append to an errors log file
        error_log_path = RAW_DIR / 'api_errors.log'
        with open(error_log_path, 'a') as f:
            f.write(json.dumps(error_record) + '\n')
        return []


def save_raw_response(region: str, observations: list, retrieval_ts: str):
    """Save the raw JSON response unchanged for reproducibility."""
    out_path = RAW_DIR / f"{region}_{retrieval_ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(observations, f, ensure_ascii=False, indent=2)


def main():
    print('Starting eBird API retrieval for Canadian regions...')
    all_records = []
    for region in CANADA_REGIONS:
        retrieval_ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        obs = fetch_region_observations(region)
        save_raw_response(region, obs, retrieval_ts)
        for rec in obs:
            # Extract needed fields; some may be missing
            extracted = {
                'retrieval_timestamp': retrieval_ts,
                'observation_date': rec.get('obsDt'),  # eBird uses 'obsDt' (YYYY-MM-DD)
                'species_code': rec.get('speciesCode'),
                'latitude': rec.get('lat'),
                'longitude': rec.get('lng'),
                'count': rec.get('howMany'),
                'region': region,
                'observation_id': rec.get('obsId') or rec.get('subId')  # fallback
            }
            all_records.append(extracted)
    # Deduplicate on observation_id (if present)
    df = None
    try:
        import pandas as pd
        df = pd.DataFrame(all_records)
        before = len(df)
        df = df.drop_duplicates(subset=['observation_id'])
        after = len(df)
        print(f'Deduplicated observations: {before} -> {after}')
    except Exception as e:
        print('Error constructing DataFrame:', e)
        return
    # Save a consolidated CSV for downstream use (raw observations)
    out_csv = RAW_DIR / 'ebird_observations_raw_2026.csv'
    df.to_csv(out_csv, index=False)
    print('Saved raw observations to', out_csv)
    # Create a brief audit markdown
    audit_md = Path(__file__).resolve().parents[2] / 'results' / 'data_qa' / 'ebird_api_feature_audit_2026.md'
    audit_md.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_md, 'w', encoding='utf-8') as f:
        f.write('# eBird API Retrieval Audit (2026)\n\n')
        f.write(f'**Retrieval timestamp (UTC)**: {datetime.now(timezone.utc).isoformat()}\n\n')
        f.write(f'**Total raw records fetched**: {len(all_records)}\n\n')
        f.write(f'**Unique observations after deduplication**: {len(df)}\n\n')
        f.write('**Note**: eBird observations are NOT confirmed HPAI detections. They are used only as a proxy for wild‑bird activity.\n')
    print('Audit written to', audit_md)

if __name__ == '__main__':
    main()
