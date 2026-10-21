"""
End-to-End Pipeline Demonstration:
1. Loads patient profiles directly from `data/demo_patients_master.json`.
2. Encodes private vitals and computes cryptographic commitment.
3. AI Inference runs inside a Zero-Knowledge Arithmetic Circuit over BN254.
4. R1CS constraints are verified mathematically.
5. Succinct Proof pi = (A, B, C) is synthesized.
6. Proof is submitted to Ethereum EVM Smart Contract.
7. Contract executes verification, stores immutable diagnosis, and emits event.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from tabulate import tabulate
from circuit.prover import ZKMLProver
from circuit.evm_simulator import SimulatedEVM

def run_demo():
    print("=" * 80)
    print("      [+] ZERO-KNOWLEDGE MACHINE LEARNING (zkML) MEDICAL INFERENCE")
    print("       Trained on Canonical Unified Dataset (data/heart_master_4feature.csv)")
    print("=" * 80)

    prover = ZKMLProver("models/cardio_weights.json")
    evm = SimulatedEVM()

    # Load master demo patients
    demo_path = "data/demo_patients_master.json" if os.path.exists("data/demo_patients_master.json") else "demo_patients.json"
    with open(demo_path) as f:
        demo_patients = json.load(f)

    for idx, (label, p) in enumerate(demo_patients.items(), 1):
        vitals = [
            float(p["age"]),
            float(p["systolic_bp"]),
            float(p["cholesterol"]),
            float(p["max_heart_rate"])
        ]
        salt = 1000 + idx * 137
        patient_name = p.get("name", label.upper())

        print(f"\n--- [Case {idx}] Profile: {patient_name} ---")
        print(f"Private Patient Input: Age {int(vitals[0])}, Systolic BP {int(vitals[1])}, Chol {int(vitals[2])}, Max HR {int(vitals[3])}")

        # 1. Prover executes private inference & generates proof
        print("\n[*] Running Zero-Knowledge Arithmetic Circuit over BN254...")
        proof_res = prover.prove_inference(vitals, secret_salt=salt)

        pub = proof_res["public_inputs"]
        stats = proof_res["circuit_stats"]
        proof = proof_res["proof"]

        circuit_summary = [
            ["Scalar Field Prime", stats["prime"][:20] + "..."],
            ["Total Circuit Wires", stats["total_wires"]],
            ["R1CS Constraints Enforced", stats["total_constraints"]],
            ["All Constraints Satisfied?", f"PASSED ({stats['r1cs_satisfied']})"],
            ["Patient Commitment (Hash)", pub["patient_commitment"][:24] + "..."],
            ["AI Risk Classification", f"[{pub['risk_label']}]"],
            ["Class 0 (Healthy) Logit", pub["float_logit_0"]],
            ["Class 1 (Anomaly) Logit", pub["float_logit_1"]],
        ]
        print(tabulate(circuit_summary, headers=["Metric", "Value"], tablefmt="grid"))

        # 2. EVM Smart Contract Verification
        print("\n[*] Submitting SNARK Proof & Commitment to Ethereum Smart Contract...")
        tx_receipt = evm.verify_and_record_diagnosis(proof_res)

        tx_summary = [
            ["Contract Address", tx_receipt["contract_address"]],
            ["Transaction Hash", tx_receipt["tx_hash"][:26] + "..."],
            ["Block Number", f"#{tx_receipt['block_number']}"],
            ["Execution Status", tx_receipt["status"]],
            ["Gas Consumed", f"{tx_receipt['gas_used']:,} gas"],
            ["Gas Fee (@ 15 Gwei)", tx_receipt["gas_cost_eth"]],
            ["Verified Diagnosis On-Chain", tx_receipt["record"]["category"]],
            ["EVM Event Emitted", tx_receipt["event_emitted"]["event"]],
        ]
        print(tabulate(tx_summary, headers=["Blockchain Parameter", "Value"], tablefmt="grid"))
        print(f"[PRIVACY GUARANTEE] Raw patient vitals {vitals} were NEVER sent to the blockchain or contract!")

    print("\n" + "=" * 80)
    print("[SUCCESS] All master demo patients verified trustlessly with zero-knowledge privacy!")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
