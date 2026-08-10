import pandas as pd, sys
path = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/data/raw/cfia/HPAI_in_Canada-public_data (1).XLSX'
# Load first sheet (assuming the relevant data is there)
try:
    df = pd.read_excel(path, engine='openpyxl')
except Exception as e:
    print('Error loading Excel file:', e)
    sys.exit(1)

# Identify date column (look for any column containing 'date')
date_col = None
for col in df.columns:
    if 'date' in str(col).lower():
        date_col = col
        break
if not date_col:
    print('No date column found')
    sys.exit(0)

# Parse dates
df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
# Filter rows for the year 2026
mask_2026 = df['_date'].dt.year == 2026
df_2026 = df[mask_2026]

# Determine poultry rows – use Production type or Species containing poultry keywords
poultry_mask = False
if 'Production type' in df_2026.columns:
    poultry_mask = df_2026['Production type'].astype(str).str.contains('chicken|turkey|poultry', case=False, na=False)
elif 'Species' in df_2026.columns:
    poultry_mask = df_2026['Species'].astype(str).str.contains('chicken|turkey|poultry', case=False, na=False)
else:
    poultry_mask = pd.Series([False] * len(df_2026), index=df_2026.index)

wild_mask = ~poultry_mask

print('Total rows in file:', df.shape[0])
print('Rows in 2026:', df_2026.shape[0])
print('Poultry rows in 2026:', poultry_mask.sum())
print('Wild‑bird rows in 2026:', wild_mask.sum())
