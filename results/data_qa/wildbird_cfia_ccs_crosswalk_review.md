# CFIA Wild-Bird CCS Crosswalk Review Report

## 1. Overview & Goal
Reviewed and finalized the wild-bird CFIA location crosswalk (`data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv`) to map wild-bird outbreak records to reference Canadian Census Consolidated Subdivisions (CCS) before feature engineering.

## 2. Summary Statistics & Coverage
- **Total Unique Location Pairs**: **694**
- **Successfully Matched After Review**: **694** (100.00%)
- **Still Unresolved (`manual_review_required`)**: **0** (0.00%)
- **Percent Coverage**: **100.00%**

## 3. Resolution Categorization
| Match Category | Count | Status | Description |
| :--- | :---: | :---: | :--- |
| **EXACT_MATCH / FRENCH_ACCENT_RESTORED** | 689 | `matched` | Exact province and CCS name match in reference lookup, or French accent corruption restored via bidirectional regex matching (e.g., *Montréal*, *Lévis*, *Château-Richer*, *Percé*, *Cap-Santé*, *Saint-Jérôme*, *Sainte-Adèle*, *Saint-André-de-Kamouraska*). |
| **CSD_MUNICIPALITY_TO_CCS** | 0 | `matched` | Municipal CSD names mapped to parent StatCan CCS reference. |
| **PROVINCE_MISMATCH_RESOLVED** | 5 | `matched` | Raw data province assignment errors corrected (e.g., *Moosomin No. 121* [MB $\to$ SK], *Britannia No. 502* [AB $\to$ SK], *Compton* / *Coaticook* [ON $\to$ QC], *Kivalliq, Unorganized* [MB $\to$ NU]). |
| **UNORGANIZED_TERRITORY** | 0 | `manual_review_required` | Unorganized territories (*Kivalliq, Unorganized*) flagged for manual spatial assignment. |
| **UNRESOLVED** | 0 | `manual_review_required` | Names without high-confidence candidate match. |

## 4. Sample Restored Location Mappings
```csv
raw_province,raw_ccs_name,target_ccs_name,review_notes
Saskatchewan,Lost River No. 313,Lost River No. 313,Matched CCS reference: 'Lost River No. 313' -> 'Lost River No. 313'
Newfoundland and Labrador,Holyrood,Holyrood,Matched CCS reference: 'Holyrood' -> 'Holyrood'
Manitoba,Rockwood,Rockwood,Matched CCS reference: 'Rockwood' -> 'Rockwood'
Alberta,Calgary,Calgary,Matched CCS reference: 'Calgary' -> 'Calgary'
Alberta,St. Paul County No. 19,St. Paul County No. 19,Matched CCS reference: 'St. Paul County No. 19' -> 'St. Paul County No. 19'
Alberta,Rocky View County,Rocky View County,Matched CCS reference: 'Rocky View County' -> 'Rocky View County'
Alberta,Newell County,Newell County,Matched CCS reference: 'Newell County' -> 'Newell County'
Alberta,Bonnyville No. 87,Bonnyville No. 87,Matched CCS reference: 'Bonnyville No. 87' -> 'Bonnyville No. 87'
Alberta,Lethbridge County,Lethbridge County,Matched CCS reference: 'Lethbridge County' -> 'Lethbridge County'
British Columbia,Saanich,Saanich,Matched CCS reference: 'Saanich' -> 'Saanich'
British Columbia,Delta,Delta,Matched CCS reference: 'Delta' -> 'Delta'
Alberta,Edmonton,Edmonton,Matched CCS reference: 'Edmonton' -> 'Edmonton'
Alberta,Lacombe County,Lacombe County,Matched CCS reference: 'Lacombe County' -> 'Lacombe County'
Alberta,Grande Prairie County No. 1,Grande Prairie County No. 1,Matched CCS reference: 'Grande Prairie County No. 1' -> 'Grande Prairie County No. 1'
Alberta,Warner County No. 5,Warner County No. 5,Matched CCS reference: 'Warner County No. 5' -> 'Warner County No. 5'
Alberta,Wheatland County,Wheatland County,Matched CCS reference: 'Wheatland County' -> 'Wheatland County'
Alberta,Strathcona County,Strathcona County,Matched CCS reference: 'Strathcona County' -> 'Strathcona County'
Alberta,Flagstaff County,Flagstaff County,Matched CCS reference: 'Flagstaff County' -> 'Flagstaff County'
Alberta,Leduc County,Leduc County,Matched CCS reference: 'Leduc County' -> 'Leduc County'
Alberta,Cardston County,Cardston County,Matched CCS reference: 'Cardston County' -> 'Cardston County'

```

## 5. Remaining Unresolved Location Pairs (`manual_review_required`)
All location pairs fully resolved.

## 6. Output Files
- **Reviewed Crosswalk CSV**: `[wildbird_cfia_ccs_crosswalk_reviewed.csv](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\data\processed\geography\wildbird_cfia_ccs_crosswalk_reviewed.csv)`
- **QA Review Report**: `[wildbird_cfia_ccs_crosswalk_review.md](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\results\data_qa\wildbird_cfia_ccs_crosswalk_review.md)`
