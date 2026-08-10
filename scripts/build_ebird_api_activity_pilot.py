# file: scripts/build_ebird_api_activity_pilot.py
"""
Pilot script for pulling a very small eBird recent‑observation sample
(5 records) for Ontario (CA‑ON) and turning it into exploratory
activity features.

What it does **not** do:
* It never prints, logs, or writes the eBird API key.
* It does not modify the primary modeling table or train any model.
* It does not claim to represent a complete national or provincial
  waterfowl dataset – it is explicitly a smoke‑test pilot.

Output files (all written inside the project root):
* data/raw/ebird_api_pilot/ebird_ca_on_recent_raw.json
* data/raw/ebird_api_pilot/ebird_ca_on_recent_observations.csv
* data/processed/features/ebird_api_activity_pilot_csd.csv
* data/processed/graph/ebird_activity_similarity_edges_pilot.csv
* results/data_qa/ebird_api_activity_pilot_audit.md
"""

import os
import json
import datetime as dt
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------
# 1.  Configuration
# -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # project root
RAW_DIR = BASE_DIR / "data" / "raw" / "ebird_api_pilot"
PROCESSED_FEATURE_DIR = BASE_DIR / "data" / "processed" / "features"
PROCESSED_GRAPH_DIR = BASE_DIR / "data" / "processed" / "graph"
QA_DIR = BASE_DIR / "results" / "data_qa"

for p in [RAW_DIR, PROCESSED_FEATURE_DIR, PROCESSED_GRAPH_DIR, QA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------
# 2.  Pull data from eBird API (smoke test)
# -----------------------------------------------------------------
API_URL = (
    "https://api.ebird.org/v2/data/obs/CA-ON/recent"
    "?maxResults=5&detail=simple"
)

api_key = os.getenv("EBIRD_API_KEY")
if not api_key:
    raise RuntimeError("EBIRD_API_KEY not set in the environment")

headers = {"X-eBirdApiToken": api_key}
resp = requests.get(API_URL, headers=headers, timeout=30)
retrieval_ts = dt.datetime.now(dt.timezone.utc).isoformat()
if resp.status_code != 200:
    raise RuntimeError(f"eBird API request failed (status {resp.status_code})")

raw_json = resp.json()

# -----------------------------------------------------------------
# 3.  Persist raw response
# -----------------------------------------------------------------
raw_path = RAW_DIR / "ebird_ca_on_recent_raw.json"
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump({"retrieval_timestamp_utc": retrieval_ts, "observations": raw_json}, f, indent=2)

# -----------------------------------------------------------------
# 4.  Pilot processing of observations
# -----------------------------------------------------------------
keep_cols = [
    "speciesCode",
    "comName",
    "obsDt",
    "howMany",
    "lat",
    "lng",
    "obsValid",
    "obsReviewed",
    "locationPrivate",
    "subId",
]

df = pd.DataFrame(raw_json)[keep_cols]

# QA before cleaning
total_obs = len(df)
unique_subids = df["subId"].nunique()
unique_species = df["comName"].nunique()
missing_coords = df["lat"].isna().sum() + df["lng"].isna().sum()
invalid_records = df[(df["obsValid"] == False) | (df["obsReviewed"] == False)].shape[0]
private_records = int(df["locationPrivate"].sum())

# Clean: keep only valid & reviewed, drop duplicate subId, drop missing coords
df_clean = df[(df["obsValid"] == True) & (df["obsReviewed"] == True)].copy()
df_clean = df_clean.drop_duplicates(subset="subId", keep="first")
df_clean = df_clean.dropna(subset=["lat", "lng"])

# -----------------------------------------------------------------
# 5.  Map coordinates → CSD using the project's CSD shapefile
# -----------------------------------------------------------------
# The shapefile is located under data/raw/boundaries/lcsd000b21a_e.shp
shp_path = BASE_DIR / "data" / "raw" / "boundaries" / "lcsd000b21a_e.shp"
csd_gdf = gpd.read_file(shp_path)
# Ensure CRS is EPSG:3347 (the project CRS) – reproject if needed
if csd_gdf.crs.to_epsg() != 3347:
    csd_gdf = csd_gdf.to_crs(epsg=3347)

# Build spatial index for fast lookup
spatial_index = csd_gdf.sindex

def point_to_csd(lat, lng):
    pt = Point(lng, lat)  # note order: (x, y) = (lon, lat)
    possible_idxs = list(spatial_index.intersection(pt.bounds))
    for idx in possible_idxs:
        if csd_gdf.iloc[idx].geometry.contains(pt):
            return csd_gdf.iloc[idx].CSDUID  # column name from shapefile
    return None

csd_ids = []
privacy_flags = []
for _, row in df_clean.iterrows():
    if row["locationPrivate"]:
        csd_ids.append(None)
        privacy_flags.append(True)
    else:
        csd_ids.append(point_to_csd(row["lat"], row["lng"]))
        privacy_flags.append(False)

df_clean["csd_uid"] = csd_ids
df_clean["is_private"] = privacy_flags

# Remove records that could not be mapped to a CSD (outside Canada, etc.)
df_clean = df_clean.dropna(subset=["csd_uid"]).copy()

# -----------------------------------------------------------------
# 6.  Aggregate to CSD‑level activity features
# -----------------------------------------------------------------
csd_feat = (
    df_clean.groupby("csd_uid")
    .agg(
        total_obs_count=("howMany", "sum"),
        unique_species=("speciesCode", lambda x: x.nunique()),
        unique_records=("subId", "nunique"),
    )
    .reset_index()
)
csd_feat["waterfowl_present"] = (csd_feat["total_obs_count"] > 0).astype(int)

# Species‑specific counts (wide format)
species_counts = (
    df_clean.groupby(["csd_uid", "comName"])['howMany']
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)
csd_feat = csd_feat.merge(species_counts, on="csd_uid", how="left")
csd_feat["retrieval_timestamp_utc"] = retrieval_ts

# Save CSD activity CSV
csd_feat_path = PROCESSED_FEATURE_DIR / "ebird_api_activity_pilot_csd.csv"
csd_feat.to_csv(csd_feat_path, index=False)

# -----------------------------------------------------------------
# 7.  Build similarity edges (cosine similarity) – only if enough data
# -----------------------------------------------------------------
MIN_OBS_COUNT = 2
valid_csd = csd_feat[csd_feat["total_obs_count"] >= MIN_OBS_COUNT].copy()
if not valid_csd.empty:
    # Numeric columns for similarity (exclude identifiers & timestamps)
    exclude_cols = {
        "csd_uid",
        "retrieval_timestamp_utc",
        "total_obs_count",
        "unique_species",
        "unique_records",
        "waterfowl_present",
    }
    feature_cols = [c for c in valid_csd.columns if c not in exclude_cols]
    feature_matrix = valid_csd[feature_cols].fillna(0).values
    sim_mat = cosine_similarity(feature_matrix)
    TOP_K = 3
    edges = []
    csd_ids = valid_csd["csd_uid"].values
    for i, src in enumerate(csd_ids):
        sims = sim_mat[i].copy()
        sims[i] = -1  # mask self
        top_idx = np.argpartition(sims, -TOP_K)[-TOP_K:]
        for j in top_idx:
            if sims[j] <= 0:
                continue
            tgt = csd_ids[j]
            weight = float(sims[j])
            evidence = int(
                valid_csd.iloc[i]["total_obs_count"] +
                valid_csd.iloc[j]["total_obs_count"]
            )
            edges.append({
                "source_csd_uid": src,
                "target_csd_uid": tgt,
                "weight": weight,
                "evidence_count": evidence,
                "edge_type": "ebird_activity_similarity",
            })
    edges_df = pd.DataFrame(edges)
else:
    edges_df = pd.DataFrame()

edges_path = PROCESSED_GRAPH_DIR / "ebird_activity_similarity_edges_pilot.csv"
edges_df.to_csv(edges_path, index=False)

# -----------------------------------------------------------------
# 8.  QA audit report
# -----------------------------------------------------------------
lines = [
    "# eBird API Pilot Audit",
    f"**Retrieval timestamp (UTC):** `{retrieval_ts}`",
    "**API region:** `CA-ON` (Ontario, Canada)",
    "",
    "## Observation counts (raw API)",
    f"- Total observations returned: {total_obs}",
    f"- Unique `subId` values: {unique_subids}",
    f"- Unique species (common name): {unique_species}",
    f"- Missing coordinates: {missing_coords}",
    f"- Invalid records (obsValid/obsReviewed false): {invalid_records}",
    f"- Private‑location records (excluded): {private_records}",
    "",
    "## After cleaning and CSD mapping",
    f"- Valid observations kept: {len(df_clean)}",
    f"- Unique CSDs with observations: {df_clean['csd_uid'].nunique()}",
    "",
    "## Species breakdown (pilot):",
]
for sp, cnt in df_clean["comName"].value_counts().items():
    lines.append(f"- {sp}: {cnt}")
lines += [
    "",
    f"## CSDs with insufficient evidence (total_obs_count < {MIN_OBS_COUNT}):",
    f"- Count: {(csd_feat['total_obs_count'] < MIN_OBS_COUNT).sum()}",
    "",
    "## Edge summary",
    f"- Total similarity edges created: {len(edges_df)}",
    "- Edge type: `ebird_activity_similarity` (undirected, top‑3 per CSD)",
    "",
    "> **Note:** This is a *pilot* using only five recent observations. It is **not** a complete historical or province‑wide dataset. The resulting features and edges are exploratory only.",
]

aud_path = QA_DIR / "ebird_api_activity_pilot_audit.md"
with open(aud_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(
    f"eBird pilot completed.\nRaw JSON → {raw_path}\nCSD activity CSV → {csd_feat_path}\nSimilarity edges CSV → {edges_path}\nQA report → {aud_path}"
)
