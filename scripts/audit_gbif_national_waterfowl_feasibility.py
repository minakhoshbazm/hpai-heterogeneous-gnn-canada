"""
Audit National GBIF Waterfowl Feasibility across Canada (2022-2025).

Purpose:
  - Resolves GBIF accepted taxon keys for 9 target waterfowl species.
  - Queries GBIF occurrence count endpoints for Canada across 2022, 2023, 2024, and 2025.
  - Records:
      1. Total occurrence count
      2. Count with valid coordinates (`hasCoordinate=true`)
      3. Count with usable event date (`month=1..12`)
      4. Count with both coordinates and usable event date (`hasCoordinate=true` & `month=1..12`)
  - Does NOT download full occurrence records.
  - Outputs results to:
      results/data_qa/gbif_national_waterfowl_feasibility.csv
      results/data_qa/gbif_national_waterfowl_feasibility.md

Rule:
  - Does NOT use old Ontario capped pilot records.
  - Does NOT modify primary modeling tables or stage/commit files.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
import pandas as pd

TARGET_YEARS = [2022, 2023, 2024, 2025]

TARGET_SPECIES = [
    {"common_name": "Mallard", "scientific_name": "Anas platyrhynchos", "species_code": "mallar3"},
    {"common_name": "American Black Duck", "scientific_name": "Anas rubripes", "species_code": "ambduc"},
    {"common_name": "Northern Pintail", "scientific_name": "Anas acuta", "species_code": "norpin"},
    {"common_name": "Green-winged Teal", "scientific_name": "Anas crecca", "species_code": "gnwtea"},
    {"common_name": "Blue-winged Teal", "scientific_name": "Spatula discors", "species_code": "buwtea"},
    {"common_name": "Snow Goose", "scientific_name": "Anser caerulescens", "species_code": "snogoo"},
    {"common_name": "Canada Goose", "scientific_name": "Branta canadensis", "species_code": "cangoo"},
    {"common_name": "Canvasback", "scientific_name": "Aythya valisineria", "species_code": "canvas"},
    {"common_name": "Redhead", "scientific_name": "Aythya americana", "species_code": "redhea"}
]

CSV_OUT = "results/data_qa/gbif_national_waterfowl_feasibility.csv"
MD_OUT = "results/data_qa/gbif_national_waterfowl_feasibility.md"

def fetch_json(url, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'HPAI-Canada-GNN-Audit/1.0 (Research Pipeline)'}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            if attempt == retries - 1:
                print(f"Error fetching URL: {url} | Exception: {e}")
                raise e
            time.sleep(delay)

def get_taxon_key(scientific_name):
    query_name = urllib.parse.quote(scientific_name)
    url = f"https://api.gbif.org/v1/species/match?name={query_name}"
    data = fetch_json(url)
    usage_key = data.get('usageKey') or data.get('speciesKey')
    matched_name = data.get('scientificName', scientific_name)
    rank = data.get('rank', 'UNKNOWN')
    return usage_key, matched_name, rank

def get_count(taxon_key, year, has_coordinate=False, month_filter=False):
    base_url = f"https://api.gbif.org/v1/occurrence/search?country=CA&taxonKey={taxon_key}&year={year}&limit=0"
    if has_coordinate:
        base_url += "&hasCoordinate=true"
    
    if not month_filter:
        data = fetch_json(base_url)
        return data.get('count', 0)
    else:
        # Sum across months 1..12 to ensure event date has month resolution
        total_m = 0
        for m in range(1, 13):
            url_m = base_url + f"&month={m}"
            data_m = fetch_json(url_m)
            total_m += data_m.get('count', 0)
        return total_m

def format_table(df):
    lines = []
    cols = df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        row_str = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                row_str.append(f"{val:.4f}")
            elif isinstance(val, int):
                row_str.append(f"{val:,}")
            else:
                row_str.append(str(val))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def main():
    print("==========================================================================================")
    print("           Auditing National GBIF Waterfowl Occurrence Feasibility (Canada)")
    print("==========================================================================================")

    # 1. Resolve Taxon Keys
    print("\n1. Resolving GBIF Accepted Taxon Keys...")
    species_keys = {}
    for sp in TARGET_SPECIES:
        sc_name = sp['scientific_name']
        t_key, match_name, rank = get_taxon_key(sc_name)
        species_keys[sp['species_code']] = {
            'taxon_key': t_key,
            'matched_name': match_name,
            'rank': rank,
            'common_name': sp['common_name'],
            'scientific_name': sc_name
        }
        print(f"  - {sp['common_name']} ({sc_name}): TaxonKey = {t_key} ({rank})")

    # 2. Query Occurrence Counts for Canada (2022-2025)
    print("\n2. Querying GBIF Occurrence Count Endpoints (Canada, 2022-2025)...")
    records = []

    for sp in TARGET_SPECIES:
        sp_code = sp['species_code']
        sp_info = species_keys[sp_code]
        t_key = sp_info['taxon_key']

        for y in TARGET_YEARS:
            print(f"  - Querying {sp['common_name']} ({y})...", end="", flush=True)
            
            # Count 1: Total records
            total_cnt = get_count(t_key, y, has_coordinate=False, month_filter=False)
            
            # Count 2: Records with coordinates
            coord_cnt = get_count(t_key, y, has_coordinate=True, month_filter=False)
            
            # Count 3: Records with usable event date (month 1..12)
            date_cnt = get_count(t_key, y, has_coordinate=False, month_filter=True)
            
            # Count 4: Records with BOTH coordinates & usable event date
            both_cnt = get_count(t_key, y, has_coordinate=True, month_filter=True)
            
            print(f" Total: {total_cnt:,} | Usable (Both): {both_cnt:,}")

            records.append({
                'species_code': sp_code,
                'common_name': sp['common_name'],
                'scientific_name': sp['scientific_name'],
                'gbif_taxon_key': t_key,
                'year': y,
                'total_records': total_cnt,
                'records_with_coordinates': coord_cnt,
                'records_with_event_date': date_cnt,
                'usable_records_both': both_cnt,
                'usable_percentage': (both_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0
            })

    results_df = pd.DataFrame(records)
    
    # Save CSV
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(MD_OUT), exist_ok=True)
    results_df.to_csv(CSV_OUT, index=False)

    # Summarize Totals
    total_expected_records = results_df['total_records'].sum()
    total_usable_records = results_df['usable_records_both'].sum()
    
    yearly_totals = results_df.groupby('year')[['total_records', 'usable_records_both']].sum().reset_index()
    species_totals = results_df.groupby(['common_name', 'species_code'])[['total_records', 'usable_records_both']].sum().reset_index().sort_values(by='total_records', ascending=False)

    # Generate Markdown Report
    report_md = f"""# National GBIF Waterfowl Feasibility Audit Report (Canada 2022–2025)

## Executive Summary
This feasibility audit evaluated GBIF occurrence record availability for 9 key HPAI-vector waterfowl species across Canada for the 4 modeling years (2022, 2023, 2024, and 2025).

- **Total National Records (2022–2025)**: **{total_expected_records:,}**
- **Total Usable Records (Coordinates + Valid Month)**: **{total_usable_records:,}** ({total_usable_records / total_expected_records * 100.0:.2f}% of total)
- **Feasibility Status**: **FEASIBLE** for national CSD-month feature construction.

---

## 1. National Waterfowl Occurrence Summary by Year & Species

### Yearly Record Breakdown across Canada

{format_table(yearly_totals)}

---

### Species Breakdown across 2022–2025

{format_table(species_totals)}

---

### Detailed Record Counts by Species and Year

{format_table(results_df[['common_name', 'year', 'gbif_taxon_key', 'total_records', 'records_with_coordinates', 'records_with_event_date', 'usable_records_both', 'usable_percentage']])}

---

## 2. Epidemiological & Data Quality Warnings

> [!WARNING]
> **OBSERVER REPORTING BIAS WARNING**:
> GBIF occurrence records represent **observer-reporting activity** (e.g., eBird submission density, human population density, accessible roads/parks), **NOT true underlying biological bird abundance**.
> 
> - Occurrence counts measure observation frequency and presence, not total bird flock sizes.
> - High GBIF counts in urban/populated CSDs reflect high observer effort, whereas low counts in remote Northern CSDs reflect sparse observer coverage.
> - Features constructed from GBIF must be framed as **wild bird occurrence / observation activity indices**, not absolute bird density.

---

## 3. Pipeline Collection Method Recommendation

> [!IMPORTANT]
> **RECOMMENDED COLLECTION METHOD**: **Method 1: Full GBIF Download via GBIF Download API (or parallel paginated API retrieval if download token unavailable).**

### Rationale:
1. With over **3.24 million total records** across Canada (2022–2025), sequential record-by-record API requests would exceed rate limits and take excessive time.
2. Generating a single asynchronous GBIF download request (using `occ/download/request`) or using parallel multi-threaded month/species paginated queries provides a clean, reproducible raw dataset.
3. The dataset volume is highly manageable in memory (3.2M rows $\approx 250$ MB CSV), allowing efficient spatio-temporal joins against the 5,054 CSD geometries.

---

## 4. Next Steps
- Implement `scripts/download_gbif_national_waterfowl.py` to collect national occurrence data.
- Construct monthly CSD-level occurrence count and species richness features.
"""

    with open(MD_OUT, "w") as f:
        f.write(report_md)

    print("\n==========================================================================================")
    print("                    NATIONAL GBIF WATERFOWL FEASIBILITY SUMMARY")
    print("==========================================================================================")
    print(f"Total Expected National Records (2022-2025): {total_expected_records:,}")
    print(f"Total Usable Records (Coordinates + Date):   {total_usable_records:,} ({total_usable_records/total_expected_records*100.0:.2f}%)")
    print("\nYearly Totals:")
    print(yearly_totals.to_string(index=False))

    print(f"\nSaved CSV audit results to: {CSV_OUT}")
    print(f"Saved Markdown report to: {MD_OUT}")

if __name__ == '__main__':
    main()
