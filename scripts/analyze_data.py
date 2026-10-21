"""
Exploratory Data Analysis of uploaded datasets:
- heart_4feature.csv
- heart_cleveland_full.csv
- demo_patients.json
"""

import json
import pandas as pd
import numpy as np

def run_analysis():
    print("=" * 70)
    print("           [+] EXPLORATORY CLINICAL DATASET ANALYSIS")
    print("=" * 70)

    # 1. heart_4feature.csv
    df4 = pd.read_csv("heart_4feature.csv")
    print(f"\n[1] 'heart_4feature.csv' Shape: {df4.shape[0]} rows, {df4.shape[1]} columns")
    print("Columns:", list(df4.columns))
    print("\nMissing values:\n", df4.isnull().sum())
    print("\nClass distribution ('high_risk'):")
    print(df4['high_risk'].value_counts(normalize=True).round(4) * 100)
    print("\nSummary Statistics:")
    print(df4.describe().round(2))

    # 2. heart_cleveland_full.csv
    df_full = pd.read_csv("heart_cleveland_full.csv")
    print(f"\n[2] 'heart_cleveland_full.csv' Shape: {df_full.shape[0]} rows, {df_full.shape[1]} columns")
    print("Columns:", list(df_full.columns))
    print("\nComparison between target and high_risk:")
    print(pd.crosstab(df_full['target'], df_full['high_risk']))

    # 3. demo_patients.json
    with open("demo_patients.json") as f:
        demo_patients = json.load(f)
    print("\n[3] 'demo_patients.json':")
    for profile, vitals in demo_patients.items():
        print(f"  - {profile}: {vitals}")

if __name__ == "__main__":
    run_analysis()
