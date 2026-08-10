# CFIA Wild-Bird Outbreaks File Audit Report

## 1. File Overview & Format Verification
- **File Path**: `data/raw/cfia/CFIA_outbreaks_wildbirds.csv`
- **Encoding**: `latin1` (Contains replacement characters `\ufffd` for French accented location names e.g., 'Montréal', 'Lévis')
- **Dimensions**: 4,183 rows $\times$ 11 columns
- **Total Records**: 4,183

## 2. Column Mapping & Structure
| Column Name | Identified Role | Missing Count | Missing % |
| :--- | :--- | :---: | :---: |
| `Province` | **Province** | 0 | 0.0% |
| `Census Consolidated Subdivision (CCS)` | **CCS / Location Name** | 0 | 0.0% |
| `Common Name` | **Species Name** | 0 | 0.0% |
| `Scientific Name` | Metadata | 0 | 0.0% |
| `Animal Class` | Metadata | 0 | 0.0% |
| `Bird Group` | Metadata | 226 | 5.4% |
| `Collection Date` | **Detection Date** | 0 | 0.0% |
| `Status` | Metadata | 869 | 20.8% |
| `Strain` | **HPAI Result / Status** | 0 | 0.0% |
| `Virus Lineage` | Metadata | 0 | 0.0% |
| `Result Authorised Date` | Metadata | 0 | 0.0% |

## 3. Sample Records (First 10 Rows)
```csv
Province,Census Consolidated Subdivision (CCS),Common Name,Scientific Name,Animal Class,Bird Group,Collection Date,Status,Strain,Virus Lineage,Result Authorised Date
Saskatchewan,Lost River No. 313,Mallard,Anas platyrhynchos,Avian,Waterfowl,10/4/2025,Harvest,H5N6,Reassortment EU&NA,6/15/2026
Newfoundland and Labrador,Holyrood,American Crow,Corvus brachyrhynchos,Avian,Corvid,11/25/2025,,H5N5,Fully Eurasian,3/19/2026
Newfoundland and Labrador,St. John's,Herring Gull,Larus argentatus,Avian,Seabird,11/25/2025,,H5N5,Fully Eurasian,3/19/2026
Newfoundland and Labrador,St. John's,Ring-Billed Gull,Larus delawarensis,Avian,Seabird,11/25/2025,,H5N5,Fully Eurasian,3/19/2026
Newfoundland and Labrador,St. John's,Great Black-Backed Gull,Larus marinus,Avian,Seabird,1/15/2026,,H5N5,Fully Eurasian,3/19/2026
Newfoundland and Labrador,St. John's,Great Black-Backed Gull,Larus marinus,Avian,Seabird,1/15/2026,,H5N5,Fully Eurasian,3/19/2026
Manitoba,Rockwood,Blue-Winged Teal,Spatula discors,Avian,Waterfowl,8/13/2022,Live,H5N6,Reassortment EU&NA,11/21/2022
Manitoba,Rockwood,Blue-Winged Teal,Spatula discors,Avian,Waterfowl,8/18/2022,Live,H5Nx,Unknown,4/25/2023
Alberta,Calgary,Great Horned Owl,Bubo virginianus,Avian,Raptor,12/5/2023,,H5Nx,Reassortment EU&NA,3/8/2024
Alberta,St. Paul County No. 19,Double-Crested Cormorant,Nannopterum auritum,Avian,Seabird,7/28/2022,,H5N1,Unknown,12/20/2022

```

## 4. Wild-Bird Target Verification
- **Record Type**: Confirmed true wild-bird detections (Waterfowl: 2,131, Raptors: 773, Seabirds: 535, Corvids: 445, Other/NaN: 299).
- **Poultry Keywords Check**: Zero commercial poultry keywords found ('chicken', 'commercial', 'flock', 'layer', 'broiler', 'poultry farm').
- **Evidence**:
  - Bird Group distribution: Waterfowl: 2,131, Raptor: 773, Seabird: 535, Corvid: 445, nan: 226, Other: 73
  - Top wild species detected: Canada Goose (898), Snow Goose (384), American Crow (320), Mallard (255), Great Horned Owl (185), Red-Tailed Hawk (168), Cackling Goose (132), Striped Skunk (123), Bald Eagle (121), Turkey Vulture (107)
  - Zero commercial poultry keywords found across all records. Verified true wild-bird dataset.
  - Confirmed 107 occurrences of 'Turkey Vulture' (Cathartes aura, wild raptor/scavenger), zero commercial domestic turkeys.
- **Latitude / Longitude Columns Present**: `False` (File relies on Province + Census Consolidated Subdivision (CCS) string names).

## 5. Geographic Crosswalk Audit against Modeling Table
- **Total Unique Location Pairs**: 694
- **Matched Location Pairs**: 665 (95.8%)
- **Unmatched Location Pairs**: 29 (4.2%)
- **Draft Crosswalk Output**: `[wildbird_cfia_ccs_crosswalk_draft.csv](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\data\processed\geography\wildbird_cfia_ccs_crosswalk_draft.csv)`

### Sample Unmatched Locations Requiring Manual Mapping Review:
```csv
raw_province,raw_ccs_name,match_type
Quebec,Château-Richer,UNMATCHED
Manitoba,Moosomin No. 121,UNMATCHED
Quebec,Montréal,UNMATCHED
Manitoba,"Kivalliq, Unorganized",UNMATCHED
Quebec,Saint-Octave-de-Métis,UNMATCHED
Quebec,Les Îles-de-la-Madeleine,UNMATCHED
Quebec,Saint-Jérôme,UNMATCHED
Quebec,Sainte-Adèle,UNMATCHED
Quebec,Lévis,UNMATCHED
Quebec,Saint-André-Avellin,UNMATCHED

```

## 6. Final Usability Decision

> [!IMPORTANT]
> **Decision**: **USABLE ONLY AFTER MANUAL CROSSWALK REVIEW**
>
> **Rationale**: High match rate (665/694 location pairs = 95.8% matched), but 29 location pairs (4.2%) require minor manual name/spelling crosswalk resolution before joining to the CSD-month modeling table.
