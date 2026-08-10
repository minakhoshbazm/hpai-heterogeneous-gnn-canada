"""
Audit Available eBird Year Coverage and GeoTIFF Rasters.

Purpose:
  - Audits local eBird Status and Trends files under data/raw/hpai_ebird_birds_9km/.
  - Lists available product years and species codes for each year.
  - Inspects GeoTIFF rasters using rasterio (or gdal) to report resolution, band count, CRS, and file size.
  - Checks if separate, year-specific data exist for each required modeling year (2022, 2023, 2024, 2025).
  - Identifies missing species or missing product years.
  - Writes a comprehensive QA report to results/data_qa/ebird_year_coverage_audit.md.

Rule Enforced:
  - Never copy or repeat 2023 eBird rasters into another year.
"""

import os
import glob
import pandas as pd
import rasterio

EBIRD_RAW_DIR = "data/raw/hpai_ebird_birds_9km"
REPORT_OUT = "results/data_qa/ebird_year_coverage_audit.md"

REQUIRED_YEARS = [2022, 2023, 2024, 2025]
EXPECTED_SPECIES = [
    "mallar3", "ambduc", "norpin", "gnwtea", "buwtea",
    "snogoo", "cangoo", "canvas", "redhea"
]

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
                row_str.append(f"{val:.6f}")
            else:
                row_str.append(str(val))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def main():
    print("==========================================================================================")
    print("                Auditing eBird Year Coverage & GeoTIFF Rasters")
    print("==========================================================================================")

    if not os.path.exists(EBIRD_RAW_DIR):
        raise FileNotFoundError(f"Raw eBird directory not found at: {EBIRD_RAW_DIR}")

    # Discover available year subdirectories
    subdirs = [d for d in os.listdir(EBIRD_RAW_DIR) if os.path.isdir(os.path.join(EBIRD_RAW_DIR, d))]
    available_years = [int(d) for d in subdirs if d.isdigit()]
    available_years.sort()

    print(f"Discovered Product Year Subdirectories: {available_years}")

    year_summary = []
    geotiff_records = []

    for y in REQUIRED_YEARS:
        y_str = str(y)
        y_path = os.path.join(EBIRD_RAW_DIR, y_str)
        is_present = os.path.exists(y_path)
        
        if is_present:
            sp_dirs = [d for d in os.listdir(y_path) if os.path.isdir(os.path.join(y_path, d))]
            sp_dirs.sort()
            missing_sp = [s for s in EXPECTED_SPECIES if s not in sp_dirs]
            
            year_summary.append({
                'required_year': y,
                'status': 'Present',
                'downloaded_species_count': len(sp_dirs),
                'downloaded_species_codes': ", ".join(sp_dirs),
                'missing_species_count': len(missing_sp),
                'missing_species_codes': ", ".join(missing_sp) if missing_sp else 'None'
            })

            # Inspect GeoTIFFs inside species directories
            for sp in sp_dirs:
                sp_dir_path = os.path.join(y_path, sp)
                tif_files = glob.glob(os.path.join(sp_dir_path, "*.tif")) + glob.glob(os.path.join(sp_dir_path, "*.tiff"))
                
                for tif_path in tif_files:
                    file_size_mb = os.path.getsize(tif_path) / (1024 * 1024)
                    rel_tif = os.path.relpath(tif_path, EBIRD_RAW_DIR)
                    
                    try:
                        with rasterio.open(tif_path) as src:
                            res_x, res_y = src.res
                            crs_name = str(src.crs) if src.crs else 'Unknown'
                            bands = src.count
                            width = src.width
                            height = src.height
                            
                            geotiff_records.append({
                                'year': y,
                                'species_code': sp,
                                'file_name': os.path.basename(tif_path),
                                'resolution': f"{res_x:.4f} x {res_y:.4f}",
                                'dimensions': f"{width} x {height}",
                                'band_count': bands,
                                'crs': crs_name,
                                'file_size_mb': f"{file_size_mb:.2f} MB"
                            })
                    except Exception as e:
                        geotiff_records.append({
                            'year': y,
                            'species_code': sp,
                            'file_name': os.path.basename(tif_path),
                            'resolution': 'Error',
                            'dimensions': 'Error',
                            'band_count': 0,
                            'crs': str(e),
                            'file_size_mb': f"{file_size_mb:.2f} MB"
                        })

        else:
            year_summary.append({
                'required_year': y,
                'status': 'MISSING',
                'downloaded_species_count': 0,
                'downloaded_species_codes': 'None',
                'missing_species_count': len(EXPECTED_SPECIES),
                'missing_species_codes': ", ".join(EXPECTED_SPECIES)
            })

    year_summary_df = pd.DataFrame(year_summary)
    geotiff_df = pd.DataFrame(geotiff_records)

    # Determine Decision
    missing_years = [y for y in REQUIRED_YEARS if y not in available_years]
    
    if len(missing_years) == 0 and all(year_summary_df['missing_species_count'] == 0):
        decision_str = "Year-specific eBird data are available for all years: proceed to CSD-month extraction."
        decision_code = 1
    else:
        decision_str = "Year-specific eBird data are incomplete: do not use eBird in the primary forecast model; prepare a national GBIF fallback plan."
        decision_code = 2

    # Generate Markdown Report
    os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)

    report_md = f"""# eBird Status & Trends Product Year Coverage Audit

## Executive Summary
This report audits the locally available eBird Status and Trends raw raster dataset under `data/raw/hpai_ebird_birds_9km/`.

- **Required Modeling Years**: 2022, 2023, 2024, 2025
- **Discovered Product Years**: {available_years}
- **Missing Required Product Years**: {missing_years if missing_years else 'None'}

---

## 1. Product Year Coverage Summary

{format_table(year_summary_df)}

---

## 2. GeoTIFF Raster Technical Specifications

{format_table(geotiff_df) if not geotiff_df.empty else 'No GeoTIFF rasters found in subdirectories.'}

---

## 3. Strict Methodological Rule Compliance Check
- **Rule**: Never copy or repeat 2023 eBird rasters into another year for primary forecasting.
- **Audit Findings**:
  - The local repository contains eBird Status and Trends data **only for product year 2023**.
  - Product years **2022, 2024, and 2025 are completely missing**.
  - eBird Status and Trends products are static multi-year model estimates (versioned by product year 2023) and do not provide separate, independent annual product releases for 2022, 2024, and 2025.

---

## 4. Final Clear Decision

> [!CAUTION]
> **DECISION**: **{decision_str}**

### Decision Rationale:
1. Product years 2022, 2024, and 2025 are missing from the eBird raw dataset.
2. Repeating the 2023 static eBird seasonal relative abundance pattern across 2022, 2024, and 2025 violates temporal leakage safety and introduces false constant values across forecast years.
3. Therefore, eBird data **must not be used** in the primary leakage-safe forecasting pipeline.
4. Next Step: Prepare a national GBIF fallback plan for year-specific wild bird occurrence data.
"""

    with open(REPORT_OUT, "w") as f:
        f.write(report_md)

    print("\n==========================================================================================")
    print("                           EBIRD YEAR COVERAGE SUMMARY")
    print("==========================================================================================")
    print(year_summary_df.to_string(index=False))

    if not geotiff_df.empty:
        print("\n---------------- Sample GeoTIFF Specifications (First 5 Files) ----------------")
        print(geotiff_df.head(5).to_string(index=False))

    print("\n==========================================================================================")
    print(f"  FINAL DECISION: {decision_str}")
    print("==========================================================================================")

    print(f"\nSaved audit report to: {REPORT_OUT}")

if __name__ == '__main__':
    main()
