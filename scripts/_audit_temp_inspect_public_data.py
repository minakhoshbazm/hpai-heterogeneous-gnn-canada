import pandas as pd, json, sys
path = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/data/raw/cfia/HPAI_in_Canada-public_data (1).XLSX'
# read first sheet
xls = pd.ExcelFile(path)
print('Sheets:', xls.sheet_names)
# try reading each sheet to find date column
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    print('--- Sheet:', sheet)
    print('Rows:', df.shape[0])
    print('Columns:', list(df.columns))
    # Identify date columns
    date_cols = [c for c in df.columns if 'date' in str(c).lower()]
    for c in date_cols:
        try:
            dmin = pd.to_datetime(df[c], errors='coerce').min()
            dmax = pd.to_datetime(df[c], errors='coerce').max()
            print(f'Date column {c}: min={dmin}, max={dmax}')
        except Exception as e:
            print('Error parsing date column', c, e)
    # Show a few rows
    print(df.head().to_string(index=False))
    # stop after first sheet for brevity
    break
