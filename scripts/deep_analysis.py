"""
Deep Multi-Dataset Comparative Analysis:
1. Audit missing values in `heart_918_missing_as_blank.csv`
2. Feature importance and categorical impact in `heart_clean_full_features.csv`
3. Statistical profile and distribution analysis in `heart_master_4feature.csv`
"""

import pandas as pd
import numpy as np

def run_deep_analysis():
    print("=" * 75)
    print("     [+] COMPREHENSIVE CLINICAL DATASET ANALYSIS & AUDIT")
    print("=" * 75)

    # 1. heart_918_missing_as_blank.csv audit
    df_918 = pd.read_csv("data/raw/heart_918_missing_as_blank.csv")
    print(f"\n[1] Raw 918-Patient Cohort Analysis:")
    print(f"Total Rows: {len(df_918)} | Total Columns: {len(df_918.columns)}")
    nulls = df_918.isnull().sum()
    print("Missing fields:")
    for col, count in nulls[nulls > 0].items():
        print(f"  - {col}: {count} rows missing ({count/len(df_918)*100:.2f}%)")

    # 2. heart_master_full.csv (746 rows clean)
    df_full = pd.read_csv("data/heart_master_full.csv")
    print(f"\n[2] Clean 746 Full-Feature Analysis:")
    print("Target distribution ('high_risk'):")
    print(df_full['high_risk'].value_counts(normalize=True).round(4) * 100)

    # Categorical distributions vs high_risk
    print("\nCategorical risk rates:")
    for cat in ['sex', 'chest_pain_type', 'exercise_angina', 'st_slope']:
        risk_rate = df_full.groupby(cat)['high_risk'].mean().round(3) * 100
        print(f"  - {cat}: {dict(risk_rate)}")

    # 3. Canonical 4-Feature Master Dataset
    df_4 = pd.read_csv("data/heart_master_4feature.csv")
    print(f"\n[3] Master 4-Feature Dataset ('data/heart_master_4feature.csv'):")
    print(f"Total Clean Records: {len(df_4)}")
    print(f"Healthy (Class 0): {(df_4['high_risk'] == 0).sum()} ({(df_4['high_risk'] == 0).mean()*100:.1f}%)")
    print(f"At-Risk (Class 1): {(df_4['high_risk'] == 1).sum()} ({(df_4['high_risk'] == 1).mean()*100:.1f}%)")
    
    print("\nSummary Statistics by Diagnostic Class:")
    grouped = df_4.groupby('high_risk').agg({
        'age': ['mean', 'std'],
        'systolic_bp': ['mean', 'std'],
        'cholesterol': ['mean', 'std'],
        'max_heart_rate': ['mean', 'std']
    }).round(1)
    print(grouped)

if __name__ == "__main__":
    run_deep_analysis()
