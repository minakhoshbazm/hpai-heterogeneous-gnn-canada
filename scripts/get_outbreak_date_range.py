import pandas as pd
import sys

PATH = 'data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv'
# Load only needed columns to keep memory low
usecols = ['year','month','outbreak_next_month']
chunks = []
for chunk in pd.read_csv(PATH, usecols=usecols, chunksize=500000):
    o = chunk[chunk['outbreak_next_month']==1]
    if not o.empty:
        chunks.append(o)
if not chunks:
    print('No outbreak rows found')
    sys.exit(0)

df = pd.concat(chunks)
# Build a datetime column (first day of month)
df['date'] = pd.to_datetime(df['year'].astype(str)+'-'+df['month'].astype(str).str.zfill(2)+'-01')
min_date = df['date'].min()
max_date = df['date'].max()
print(f'Outbreaks present from {min_date.strftime("%Y-%m")} to {max_date.strftime("%Y-%m")}')
# Count per year
counts = df.groupby('year').size()
print('Counts per year:')
for yr, cnt in counts.items():
    print(f'  {yr}: {cnt}')
