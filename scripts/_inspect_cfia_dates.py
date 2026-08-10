import pandas as pd, sys
path = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/data/raw/cfia/HPAI_in_Canada-public_data (1).XLSX'
df = pd.read_excel(path, engine='openpyxl')
# Show distinct years in CFIA Notification date
if 'CFIA Notification date' in df.columns:
    dates = pd.to_datetime(df['CFIA Notification date'], errors='coerce')
    years = dates.dt.year.dropna().unique()
    print('Years in CFIA Notification date column:', sorted(years))
    # filter 2026
    df_2026 = df[dates.dt.year == 2026]
    print('Rows with 2026 notification date:', df_2026.shape[0])
    if not df_2026.empty:
        print(df_2026[['CFIA Notification date','Production type','Species','Province']].head())
else:
    print('CFIA Notification date column not found')
