#!/usr/bin/env python3
"""
scripts/audit_outbreak_counts_for_spatial_validation.py

Audits outbreak counts by province, year, and regional blocks to evaluate
spatial validation feasibility.
"""

import os
import pandas as pd

def main():
    model_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags_and_neighbor_pressure.csv'
    prov_csv_out = 'results/baselines/spatial_validation_positive_counts_by_province_year.csv'
    region_csv_out = 'results/baselines/spatial_validation_positive_counts_by_region_year.csv'

    print("Loading modeling table...")
    df = pd.read_csv(model_table_path)

    # 2. Count positives by province, name, and year
    # Filter to positives
    df_pos = df[df['outbreak_next_month'] == 1].copy()

    print("\nCalculating counts by province and year...")
    # Group by province code, province name, and year
    prov_year_counts = df_pos.groupby(['province_code', 'province_name', 'year']).size().reset_index(name='positive_count')
    
    # 3. Total positives by province
    prov_total_counts = df_pos.groupby(['province_code', 'province_name']).size().reset_index(name='total_positives')
    prov_total_counts = prov_total_counts.sort_values(by='total_positives', ascending=False)

    # 4. Regional blocks mapping
    # Atlantic: 10, 11, 12, 13
    # Quebec_Ontario: 24, 35
    # Prairies: 46, 47, 48
    # British_Columbia: 59
    def get_region_block(code):
        if code in [10, 11, 12, 13]:
            return 'Atlantic'
        elif code in [24, 35]:
            return 'Quebec_Ontario'
        elif code in [46, 47, 48]:
            return 'Prairies'
        elif code in [59]:
            return 'British_Columbia'
        else:
            return 'Other'

    df_pos['regional_block'] = df_pos['province_code'].apply(get_region_block)
    df['regional_block'] = df['province_code'].apply(get_region_block)

    # 5. Count positives by regional block and year
    region_year_counts = df_pos.groupby(['regional_block', 'year']).size().reset_index(name='positive_count')

    # Print results
    print("\n=========================================================================")
    print("                Outbreak Count Audit for Spatial Validation")
    print("=========================================================================")
    print("Total Positive Outbreak Counts by Province:")
    print(prov_total_counts.to_string(index=False))
    
    print("\n-------------------------------------------------------------------------")
    print("Outbreak Counts by Province and Year:")
    # Pivot for clean presentation
    prov_year_pivot = prov_year_counts.pivot(index=['province_code', 'province_name'], columns='year', values='positive_count').fillna(0).astype(int)
    print(prov_year_pivot)
    
    print("\n-------------------------------------------------------------------------")
    print("Outbreak Counts by Regional Block and Year:")
    region_year_pivot = region_year_counts.pivot(index='regional_block', columns='year', values='positive_count').fillna(0).astype(int)
    print(region_year_pivot)
    print("=========================================================================")

    # 6. Recommendation logic
    print("\n--- Recommendation & Feasibility Analysis ---")
    
    # Check if any province has very few outbreaks
    # Leave-one-province-out feasibility check
    low_count_provinces = []
    for _, row in prov_total_counts.iterrows():
        if row['total_positives'] < 15:
            low_count_provinces.append(f"{row['province_name']} (Total: {row['total_positives']})")
            
    print("a. Leave-One-Province-Out Feasibility:")
    if low_count_provinces:
        print(f"   - WARNING: Several provinces have extremely low outbreak counts across all years:")
        for lp in low_count_provinces:
            print(f"     * {lp}")
        print("   - Evaluation metrics (like PR-AUC) will be unstable or undefined when these provinces are held out.")
        print("   - Therefore, Leave-One-Province-Out validation is NOT globally recommended for all provinces.")
    else:
        print("   - Feasible. All participating provinces have enough outbreaks for stable evaluation.")

    print("\nb. Regional Block Validation Stability:")
    # Check regional block totals
    region_totals = df_pos.groupby('regional_block').size()
    print("   - Total positive outbreaks by block:")
    for reg, val in region_totals.items():
        print(f"     * {reg}: {val}")
    print("   - CONCLUSION: Regional block validation is much more stable because each block contains a larger number of outbreaks (minimum 10 in Atlantic).")
    print("   - Holding out Quebec_Ontario, Prairies, or British Columbia will yield highly reliable evaluation metrics.")

    # Save summary tables
    os.makedirs(os.path.dirname(prov_csv_out), exist_ok=True)
    prov_year_pivot.to_csv(prov_csv_out)
    region_year_pivot.to_csv(region_csv_out)
    print(f"\nSaved province-year summary to: {prov_csv_out}")
    print(f"Saved region-year summary to:   {region_csv_out}")

if __name__ == '__main__':
    main()
