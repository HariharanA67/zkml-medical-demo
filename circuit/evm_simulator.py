"""
EVM State and Contract Simulator for ZKMedicalDiagnosisVerifier.
Simulates Ethereum blocks, transaction gas calculation, event emissions,
and immutable contract storage.
"""

import time
import hashlib
from typing import Dict, Any, List, Optional
from circuit.field import BN254_PRIME

class SimulatedEVM:
    def __init__(self):
        self.block_number = 19482000
        self.gas_price_gwei = 15
        self.contract_address = "0x71C8419E97305d2146907cf7c6fa713583E156A2"
        self.verified_records: Dict[str, Dict[str, Any]] = {}
        self.event_logs: List[Dict[str, Any]] = []

    def verify_and_record_diagnosis(
        self,
        proof_payload: Dict[str, Any],
        sender_address: str = "0x89205A3A3b2A69De6Dbf7f01ED13B2108B2c43e7"
    ) -> Dict[str, Any]:
        """
        Simulates executing `verifyAndRecordDiagnosis` on the Ethereum EVM.
        """
        start_time = time.time()
        self.block_number += 1
        tx_hash = "0x" + hashlib.sha256(f"{self.block_number}:{time.time()}".encode()).hexdigest()

        proof = proof_payload["proof"]
        pub = proof_payload["public_inputs"]
        commitment = pub["patient_commitment"]
        pred_class = pub["predicted_class"]
        risk_label = pub["risk_label"]

        # Parse G1 & G2 points
        a_x = int(proof["a"][0], 16)
        a_y = int(proof["a"][1], 16)
        c_x = int(proof["c"][0], 16)
        c_y = int(proof["c"][1], 16)

        # Field validation
        if not (a_x < BN254_PRIME and a_y < BN254_PRIME and c_x < BN254_PRIME and c_y < BN254_PRIME):
            return {
                "success": False,
                "error": "PointOutOfFieldRange",
                "tx_hash": tx_hash,
                "gas_used": 28500
            }

        # Simulated Gas Consumption:
        # - Base Tx: 21,000 gas
        # - Calldata expansion (proof points + inputs): ~12,400 gas
        # - Precompile 0x08 (ecPairing): ~115,000 gas
        # - SSTORE (recording diagnosis struct to storage): ~22,100 gas
        # - LOG2 (emitting InferenceProofVerified): ~1,750 gas
        total_gas = 21000 + 12400 + 115000 + 22100 + 1750
        gas_cost_eth = (total_gas * self.gas_price_gwei * 1e9) / 1e18

        # Record state on simulated blockchain
        record = {
            "patient_commitment": commitment,
            "category": risk_label,
            "predicted_class": pred_class,
            "timestamp": int(time.time()),
            "block_number": self.block_number,
            "verifier_node": sender_address,
            "tx_hash": tx_hash,
            "gas_used": total_gas,
            "gas_cost_eth": f"{gas_cost_eth:.6f} ETH"
        }
        self.verified_records[commitment] = record

        # Emit EVM event
        event = {
            "event": "InferenceProofVerified",
            "block_number": self.block_number,
            "tx_hash": tx_hash,
            "patient_commitment": commitment,
            "category": risk_label,
            "submitter": sender_address,
            "timestamp": record["timestamp"]
        }
        self.event_logs.append(event)

        return {
            "success": True,
            "status": "0x1 (SUCCESS)",
            "tx_hash": tx_hash,
            "block_number": self.block_number,
            "contract_address": self.contract_address,
            "gas_used": total_gas,
            "gas_cost_eth": f"{gas_cost_eth:.6f} ETH",
            "event_emitted": event,
            "record": record,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

    def get_record(self, patient_commitment: str) -> Optional[Dict[str, Any]]:
        return self.verified_records.get(patient_commitment)
