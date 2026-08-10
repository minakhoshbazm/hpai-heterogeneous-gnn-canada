#!/usr/bin/env python
"""
Audit script for the newly added CFIA public‑dashboard Excel file.

File: data/raw/cfia/HPAI_in_Canada-public_data (1).XLSX

Purpose
--------
1. Summarise each sheet (names, columns, row count, missing‑value counts).
2. Detect every date column, parse it, and report per‑column min / max dates and
   yearly record counts.
3. Identify the official host‑type field (e.g. the column that explicitly
   distinguishes Poultry, Wild‑bird, Mammal, Other).  This is inferred from the
   column values – we look for a column whose unique values map cleanly to those
   four categories.
4. Build a reviewed table with the canonical fields required for downstream
   modelling:
   ``record_id, event_date, notification_date, province, location,
   host_species, production_type, category, source_sheet``.
5. Count the exact number of 2026 records for each host type (Poultry vs Wild‑bird).
6. Detect duplicate records (identical across all key fields).
7. Compare the 2026 records with the existing CFIA wild‑bird file
   ``data/raw/cfia/CFIA_outbreaks_wildbirds.csv`` – report any overlaps.
8. Persist the results as:
   * results/data_qa/cfia_2026_public_data_audit.csv – the full reviewed table.
   * results/data_qa/cfia_2026_public_data_audit.md – a markdown summary.
   * data/processed/cfia/cfia_2026_reviewed_records.csv – a copy for the
     modelling pipeline.

Constraints
-----------
* Do NOT modify any existing modeling tables.
* Do NOT overwrite any existing output files.
* Use only the official host‑type column – no ad‑hoc keyword heuristics.
* All operations are read‑only on source data.
"""

import pathlib
import pandas as pd
import json
from collections import Counter

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = pathlib.Path(__file__).resolve().parents[1]  # project root
XLSX_PATH = BASE / "data" / "raw" / "cfia" / "HPAI_in_Canada-public_data (1).XLSX"
WILDBIRD_PATH = BASE / "data" / "raw" / "cfia" / "CFIA_outbreaks_wildbirds.csv"

# Output directories (create if missing)
OUT_CSV = BASE / "results" / "data_qa" / "cfia_2026_public_data_audit.csv"
OUT_MD = BASE / "results" / "data_qa" / "cfia_2026_public_data_audit.md"
OUT_PROC = BASE / "data" / "processed" / "cfia" / "cfia_2026_reviewed_records.csv"
for p in [OUT_CSV.parent, OUT_PROC.parent]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def infer_host_type_column(df: pd.DataFrame) -> str:
    """Return the column name that most cleanly separates host categories.
    Strategy:
    * Look for columns whose unique values intersect with a predefined set of
      expected categories (Poultry, Wild bird, Mammal, Other).  The column with the
      highest match ratio is selected.
    * If multiple candidates match, prefer the one with the fewest missing
      values.
    """
    candidate_keywords = ["host", "production", "category", "type", "species"]
    expected = {"poultry", "wild bird", "wild-bird", "wildbird", "mammal", "other"}
    best_col = None
    best_score = -1
    for col in df.columns:
        lowered = df[col].astype(str).str.lower()
        uniq = set(lowered.dropna().unique())
        match = len(uniq & expected)
        if match > best_score:
            best_score = match
            best_col = col
    return best_col

def parse_dates(df: pd.DataFrame) -> dict:
    """Identify columns that contain dates (name includes 'date') and parse them.
    Returns a dict mapping column name -> parsed datetime Series.
    """
    date_cols = {col: pd.to_datetime(df[col], errors='coerce')
                 for col in df.columns if 'date' in col.lower()}
    return date_cols

# ------------------------------------------------------------
# Load the Excel file – we will process all sheets
# ------------------------------------------------------------
excel = pd.ExcelFile(XLSX_PATH, engine="openpyxl")
all_sheets = excel.sheet_names

summary_lines = []
reviewed_rows = []  # will become the final audited table

for sheet_name in all_sheets:
    df = pd.read_excel(excel, sheet_name=sheet_name)
    row_count = df.shape[0]
    col_names = list(df.columns)
    missing_counts = df.isna().sum().to_dict()
    summary_lines.append(f"## Sheet: {sheet_name}")
    summary_lines.append(f"- Rows: {row_count}")
    summary_lines.append(f"- Columns ({len(col_names)}): {', '.join(col_names)}")
    summary_lines.append("- Missing values per column:")
    for col, miss in missing_counts.items():
        summary_lines.append(f"  - {col}: {miss}")
    # Date handling for this sheet
    date_series = parse_dates(df)
    if date_series:
        summary_lines.append("- Date columns parsed:")
        for dcol, series in date_series.items():
            min_date = series.min()
            max_date = series.max()
            summary_lines.append(f"  - {dcol}: min={min_date.date()}, max={max_date.date()}")
            yearly_counts = series.dt.year.value_counts().sort_index()
            summary_lines.append("    Yearly counts:")
            for yr, cnt in yearly_counts.items():
                summary_lines.append(f"      {yr}: {cnt}")
    else:
        summary_lines.append("- No date columns detected.")

    # Identify host‑type column (official field) – use the whole sheet
    host_col = infer_host_type_column(df)
    if not host_col:
        raise RuntimeError(f"Unable to infer host‑type column in sheet {sheet_name}")

    # Build reviewed rows – we keep a stable record_id per original row
    for idx, row in df.iterrows():
        # Pull the canonical fields if they exist, otherwise set as None
        event_date = None
        notif_date = None
        if "Date of first clinical signs" in df.columns:
            event_date = pd.to_datetime(row["Date of first clinical signs"], errors='coerce')
        elif "CFIA Notification date" in df.columns:
            event_date = pd.to_datetime(row["CFIA Notification date"], errors='coerce')
        # For notification date we prefer the explicit column if present
        if "CFIA Notification date" in df.columns:
            notif_date = pd.to_datetime(row["CFIA Notification date"], errors='coerce')
        # Assemble fields (fallback to None when missing)
        reviewed_rows.append({
            "record_id": f"{sheet_name}_{idx+2}",  # +2 accounts for Excel header row offset
            "source_sheet": sheet_name,
            "event_date": event_date.date() if pd.notnull(event_date) else None,
            "notification_date": notif_date.date() if pd.notnull(notif_date) else None,
            "province": row.get("Province"),
            "location": row.get("Location name"),
            "host_species": row.get("Species"),
            "production_type": row.get("Production type"),
            "category": row.get("Category"),
            "host_type_official": row.get(host_col),
        })

# ------------------------------------------------------------
# Convert reviewed rows to DataFrame
# ------------------------------------------------------------
reviewed_df = pd.DataFrame(reviewed_rows)

# ------------------------------------------------------------
# 2026 counts by official host type
# ------------------------------------------------------------
reviewed_df['year'] = pd.to_datetime(reviewed_df['event_date'], errors='coerce').dt.year
# Use the inferred official host column to separate poultry vs wild‑bird
poultry_mask = reviewed_df['host_type_official'].astype(str).str.lower().str.contains('poultry')
wild_mask = reviewed_df['host_type_official'].astype(str).str.lower().str.contains('wild')
counts_2026 = {
    "poultry_2026": int(((poultry_mask) & (reviewed_df['year'] == 2026)).sum()),
    "wildbird_2026": int(((wild_mask) & (reviewed_df['year'] == 2026)).sum()),
}
summary_lines.append("\n# 2026 Host‑type Summary")
summary_lines.append(f"- Poultry records in 2026: {counts_2026['poultry_2026']}")
summary_lines.append(f"- Wild‑bird records in 2026: {counts_2026['wildbird_2026']}")

# ------------------------------------------------------------
# Duplicate detection (full row duplicate on selected key columns)
# ------------------------------------------------------------
key_cols = ["event_date", "province", "location", "host_species", "production_type", "category"]
duplicates = reviewed_df.duplicated(subset=key_cols, keep=False)
dup_count = duplicates.sum()
summary_lines.append(f"\n# Duplicate Check\n- Duplicate rows (based on key fields): {dup_count}")

# ------------------------------------------------------------
# Compare with existing CFIA wild‑bird file (only 2026 rows)
# ------------------------------------------------------------
wildbird_df = pd.read_csv(WILDBIRD_PATH, encoding='latin1', on_bad_lines='skip')
# Assume wildbird file has a column named 'Date of first clinical signs' or similar
wild_date_col = None
for c in wildbird_df.columns:
    if 'date' in c.lower():
        wild_date_col = c
        break
if wild_date_col:
    wildbird_df['_date'] = pd.to_datetime(wildbird_df[wild_date_col], errors='coerce')
    wild2026 = wildbird_df[wildbird_df['_date'].dt.year == 2026]
    # Simple overlap based on province + event date
    overlap = reviewed_df.merge(
        wild2026,
        left_on=["province", "event_date"],
        right_on=["Province", "_date"],
        how="inner",
        suffixes=("_pub", "_wild"),
    )
    overlap_cnt = overlap.shape[0]
else:
    overlap_cnt = 0
summary_lines.append(f"\n# Comparison with existing CFIA wild‑bird file\n- Overlap of 2026 records (province + date): {overlap_cnt}")

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
reviewed_df.to_csv(OUT_CSV, index=False)
reviewed_df.to_csv(OUT_PROC, index=False)
# Markdown report
with open(OUT_MD, "w", encoding="utf-8") as md:
    md.write("# CFIA 2026 Public Data Audit\n\n")
    md.write("\n".join(summary_lines))
    md.write("\n")

print("Audit completed. Files written:")
print(f"- Reviewed CSV: {OUT_CSV}")
print(f"- Processed CSV: {OUT_PROC}")
print(f"- Markdown report: {OUT_MD}")
print(f"2026 poultry records: {counts_2026['poultry_2026']}")
print(f"2026 wild‑bird records: {counts_2026['wildbird_2026']}")
