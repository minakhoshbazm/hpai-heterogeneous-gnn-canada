import os
import pandas as pd
import numpy as np

def generate_mock_features(labels_path, output_path):
    """
    Generates a synthetic/mock feature dataset aligned with the real CD-month labels.
    This creates realistic seasonal/spatial patterns for environmental and species abundances.
    """
    print(f"Loading labels from: {labels_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at: {labels_path}")
        
    labels_df = pd.read_csv(labels_path)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate unique static features per CD
    unique_cds = labels_df['cd_uid'].unique()
    
    # Create static CD features dictionary
    static_features = {}
    for cd in unique_cds:
        # Poultry density proxy (number of farms per sq km)
        # Some areas (like Fraser Valley or Southern Ontario) have high poultry density
        is_high_density = np.random.rand() > 0.9
        if is_high_density:
            poultry_density = np.random.uniform(2.0, 5.0)
            road_density = np.random.uniform(1.0, 3.0)
        else:
            poultry_density = np.random.uniform(0.01, 0.5)
            road_density = np.random.uniform(0.02, 0.8)
            
        static_features[cd] = {
            'poultry_density_proxy': poultry_density,
            'road_density_proxy': road_density
        }
        
    static_df = pd.DataFrame.from_dict(static_features, orient='index').reset_index().rename(columns={'index': 'cd_uid'})
    
    # Merge static features into the main dataframe
    df = labels_df.merge(static_df, on='cd_uid', how='left')
    
    # 2. Parse months to determine seasonal patterns
    df['date'] = pd.to_datetime(df['month'] + "-01")
    df['month_num'] = df['date'].dt.month
    
    # 3. Synthesize dynamic environmental anomalies (seasonal + random noise)
    print("Generating synthetic environmental features...")
    # NDVI (peaks in summer: July, month=7)
    ndvi_seasonal = np.sin((df['month_num'] - 4) * np.pi / 6) * 0.4 + 0.4
    df['ndvi_anomaly'] = ndvi_seasonal + np.random.normal(0, 0.05, len(df))
    
    # NDWI (peaks in spring runoff: May, month=5)
    ndwi_seasonal = np.sin((df['month_num'] - 2) * np.pi / 6) * 0.3 + 0.2
    df['ndwi_anomaly'] = ndwi_seasonal + np.random.normal(0, 0.05, len(df))
    
    # Temp Anomaly (seasonal sine wave peaking in July/August)
    temp_seasonal = np.sin((df['month_num'] - 4) * np.pi / 6) * 15.0 + 10.0
    df['temp_anomaly'] = temp_seasonal + np.random.normal(0, 1.5, len(df))
    
    # Precip Anomaly (seasonal, e.g. spring and autumn rain)
    precip_seasonal = np.cos((df['month_num'] - 5) * np.pi / 6) * 30.0 + 60.0
    df['precip_anomaly'] = np.maximum(0, precip_seasonal + np.random.normal(0, 10.0, len(df)))
    
    # Surface Water Fraction (seasonal ice melt, peak in summer/spring)
    water_seasonal = np.sin((df['month_num'] - 3) * np.pi / 6) * 0.08 + 0.09
    df['water_fraction'] = np.clip(water_seasonal + np.random.normal(0, 0.01, len(df)), 0, 1)
    
    # 4. Synthesize dynamic species abundance (with spring/fall migration peaks)
    print("Generating synthetic species abundance features...")
    # Waterfowl migration peaks: Spring (April/May, month=4,5) and Fall (Sept/Oct, month=9,10)
    # Double-peak model using sum of Gaussians
    def migration_seasonal(m):
        spring_peak = np.exp(-((m - 4.5) / 1.0)**2)
        fall_peak = np.exp(-((m - 9.5) / 1.0)**2)
        return (spring_peak * 0.7 + fall_peak * 1.0)
        
    mig_pattern = df['month_num'].apply(migration_seasonal).values
    
    # Add spatial scaling (some CDs are major stopovers)
    cd_stopover_scale = {cd: (np.random.uniform(5.0, 100.0) if np.random.rand() > 0.8 else np.random.uniform(0.1, 5.0)) for cd in unique_cds}
    df['cd_scale'] = df['cd_uid'].map(cd_stopover_scale)
    
    df['mallard_abundance_proxy'] = np.maximum(0, df['cd_scale'] * mig_pattern + np.random.normal(0, 2.0, len(df)))
    df['canada_goose_abundance_proxy'] = np.maximum(0, df['cd_scale'] * mig_pattern * 1.2 + np.random.normal(0, 3.0, len(df)))
    
    # 5. Generate Lag Features
    print("Computing lag features...")
    df = df.sort_values(by=['cd_uid', 'month'])
    df['outbreak_lag_1'] = df.groupby('cd_uid')['outbreak_binary_current_month'].shift(1).fillna(0).astype(int)
    df['outbreak_lag_2'] = df.groupby('cd_uid')['outbreak_binary_current_month'].shift(2).fillna(0).astype(int)
    
    # 6. Drop intermediate columns and save
    df = df.drop(columns=['date', 'month_num', 'cd_scale'])
    
    # Ensure target folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated mock feature dataset of shape {df.shape} at: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    labels_file = os.path.join(base_dir, "data", "processed", "cfia", "cd_month_labels.csv")
    out_file = os.path.join(base_dir, "data", "processed", "features", "mock_features.csv")
    
    generate_mock_features(labels_file, out_file)
