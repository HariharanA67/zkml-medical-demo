# 🔐 Zero-Knowledge Machine Learning (zkML) Medical Inference Engine v2.0

A production-grade, verifiable zero-knowledge inference pipeline combining:
1. **AI/ML:** PyTorch Neural Network trained on the multi-hospital clinical dataset (744 clean patient records).
2. **Cybersecurity:** Zero-Knowledge Cryptography ($\text{zk-SNARKs}$ over BN254 elliptic curve) guaranteeing patient vitals privacy and cryptographic commitment schemes.
3. **Scientific Computing & Optimization:** Integer fixed-point quantization, Rank-1 Constraint System (R1CS) arithmetic circuit synthesis, and finite field polynomial evaluations.
4. **Blockchain & Distributed Systems:** Ethereum Virtual Machine (EVM) smart contract in Solidity verifying SNARK proofs using cryptographic pairing precompiles (`0x08`).

---

## 📁 Consolidated Canonical Data Structure

To eliminate confusion across multiple input files, all data is organized into single canonical master files:

* **Canonical 4-Feature Dataset:** [`data/heart_master_4feature.csv`](file:///c:/Users/eruma/Desktop/1/data/heart_master_4feature.csv) (744 clean, deduplicated multi-hospital records).
* **Canonical Full-Feature Dataset:** [`data/heart_master_full.csv`](file:///c:/Users/eruma/Desktop/1/data/heart_master_full.csv) (746 clinical records with 12 features).
* **Master Demo Profiles:** [`data/demo_patients_master.json`](file:///c:/Users/eruma/Desktop/1/data/demo_patients_master.json) (5 distinct diagnostic benchmarks from young fit to severe coronary pathology).
* **Historical Raw Archives:** Safely organized in `data/raw/` (`heart_918_missing_as_blank.csv`, etc.).

---

## 🏗️ Architecture & Circuit Specs

```
[Patient Private Input (Age, BP, Chol, Max HR)]
                      │
                      ▼
            [Fixed-Point Quantizer]
         (Float32 -> Int64, Scale S = 2^10)
                      │
                      ▼
          [R1CS Arithmetic Circuit]
      ((A · s) * (B · s) = (C · s) over F_p)
            • 139 Circuit Wires
            • 76 Quadratic Constraints
                      │
                      ▼
             [Witness Generation]
           (100% Constraints Satisfied)
                      │
                      ▼
         [Cryptographic SNARK Prover]
           Proof π = { A ∈ G1, B ∈ G2, C ∈ G1 }
           Public Signal: [Patient Commitment, Diagnosis]
                      │
                      ▼
   [Ethereum Smart Contract (Solidity / EVM)]
   `ZKMedicalDiagnosisVerifier.sol`
   • Checks BN254 bilinear pairing e(A, B) = ...
   • Emits InferenceProofVerified event
   • Records immutable diagnosis to on-chain ledger
```

---

## 📊 Model Performance Benchmark (744 Multi-Center Cohort)

| Evaluation Metric | 10-Fold Stratified Cross-Validation | Full Production Fit |
| :--- | :---: | :---: |
| **Accuracy** | **68.82% $\pm$ 3.55%** | **70.83%** |
| **ROC-AUC** | **0.7397** | **0.7727** |
| **Precision** | **67.64%** | **70.06%** |
| **Recall** | **66.80%** | **67.89%** |
| **F1-Score** | **67.06%** | **68.96%** |

---

## 🚀 Quickstart & Reproduction

### 1. Run the End-to-End Pipeline Demo
```bash
.venv\Scripts\python.exe scripts/demo_pipeline.py
```

### 2. Launch the Interactive Web Dashboard
```bash
.venv\Scripts\streamlit.exe run app.py
```
Open `http://localhost:8501` to test custom patient vitals and view real-time circuit proofs and EVM verification.
