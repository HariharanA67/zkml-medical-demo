"""
Consolidates and unifies all uploaded datasets into single clean master files.
Keeps raw datasets safely archived in `data/raw/`.
"""

import os
import shutil
import json
import pandas as pd

def consolidate_data():
    os.makedirs("data/raw", exist_ok=True)

    # 1. Archive raw files if present in root
    raw_files = [
        "heart_4feature.csv",
        "heart_clean_4feature.csv",
        "heart_cleveland_full.csv",
        "heart_clean_full_features.csv",
        "heart_918_missing_as_blank.csv",
        "demo_patients.json",
        "demo_patients1.json"
    ]

    for rf in raw_files:
        if os.path.exists(rf):
            shutil.copy(rf, os.path.join("data/raw", rf))

    # 2. Build Single Canonical 4-Feature Master File: `data/heart_master_4feature.csv`
    # Use clean 746-patient cohort with deduplication
    df_clean_4 = pd.read_csv("heart_clean_4feature.csv")
    df_clean_4 = df_clean_4.drop_duplicates().reset_index(drop=True)
    df_clean_4.to_csv("data/heart_master_4feature.csv", index=False)
    print(f"Created 'data/heart_master_4feature.csv' with {len(df_clean_4)} clean clinical records.")

    # 3. Build Single Canonical Full-Feature Master File: `data/heart_master_full.csv`
    df_clean_full = pd.read_csv("heart_clean_full_features.csv")
    df_clean_full.to_csv("data/heart_master_full.csv", index=False)
    print(f"Created 'data/heart_master_full.csv' with {len(df_clean_full)} clinical records (12 features).")

    # 4. Build Unified Master Demo Patients JSON: `data/demo_patients_master.json`
    master_demo = {
        "low_risk_athlete": {
            "name": "Patient Alpha (Young & Fit)",
            "age": 29,
            "systolic_bp": 130,
            "cholesterol": 204,
            "max_heart_rate": 202,
            "expected": "LOW_RISK_HEALTHY"
        },
        "borderline_metabolic": {
            "name": "Patient Beta (Metabolic / Elevated Chol)",
            "age": 58,
            "systolic_bp": 136,
            "cholesterol": 319,
            "max_heart_rate": 152,
            "expected": "HIGH_RISK_ANOMALY"
        },
        "borderline_elderly": {
            "name": "Patient Gamma (Elderly / Borderline)",
            "age": 67,
            "systolic_bp": 106,
            "cholesterol": 223,
            "max_heart_rate": 142,
            "expected": "HIGH_RISK_ANOMALY"
        },
        "high_risk_bradycardia": {
            "name": "Patient Delta (Severe Low Max-HR & Elderly)",
            "age": 67,
            "systolic_bp": 120,
            "cholesterol": 237,
            "max_heart_rate": 71,
            "expected": "HIGH_RISK_ANOMALY"
        },
        "high_risk_hyperlipidemia": {
            "name": "Patient Epsilon (Extreme High Cholesterol & Low HR)",
            "age": 58,
            "systolic_bp": 132,
            "cholesterol": 458,
            "max_heart_rate": 69,
            "expected": "HIGH_RISK_ANOMALY"
        }
    }

    with open("data/demo_patients_master.json", "w") as f:
        json.dump(master_demo, f, indent=2)
    print(f"Created 'data/demo_patients_master.json' with {len(master_demo)} distinct diagnostic profiles.")

if __name__ == "__main__":
    consolidate_data()
