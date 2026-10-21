"""
Submits a real Zero-Knowledge ML Proof to the deployed contract on Ethereum Sepolia.
1. Computes local private inference in the BN254 arithmetic circuit.
2. Formulates proof points: a (G1), b (G2), c (G1).
3. Signs and broadcasts a live transaction to Sepolia.
4. Confirms inclusion and returns the Etherscan transaction link.
"""

import os
import sys
import json
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account

from circuit.prover import ZKMLProver

load_dotenv()

DEFAULT_SEPOLIA_RPCS = [
    "https://1rpc.io/sepolia",
    "https://gateway.tenderly.co/public/sepolia",
    "https://ethereum-sepolia-rpc.publicnode.com"
]

def get_web3():
    env_rpc = os.getenv("SEPOLIA_RPC_URL")
    candidates = [env_rpc] + DEFAULT_SEPOLIA_RPCS if env_rpc else DEFAULT_SEPOLIA_RPCS
    for rpc in candidates:
        if not rpc:
            continue
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                return w3, rpc
        except Exception:
            continue
    return Web3(Web3.HTTPProvider(DEFAULT_SEPOLIA_RPCS[0])), DEFAULT_SEPOLIA_RPCS[0]

def verify_on_live_testnet(patient_key: str = "high_risk_bradycardia"):
    print("=" * 75)
    print("     [+] SUBMITTING zkML PROOF TO LIVE ETHEREUM SEPOLIA TESTNET")
    print("=" * 75)

    w3, rpc_url = get_web3()
    if not w3.is_connected():
        print("[!] Cannot connect to Sepolia RPC.")
        return

    priv_key = os.getenv("TESTNET_PRIVATE_KEY")
    deployed_address = os.getenv("DEPLOYED_CONTRACT_ADDRESS")

    if not priv_key:
        print("[!] TESTNET_PRIVATE_KEY not found in .env. Run scripts/manage_testnet.py first.")
        return

    if not deployed_address:
        print("[!] DEPLOYED_CONTRACT_ADDRESS not found in .env. Deploy contract first via:")
        print("    .venv\\Scripts\\python.exe scripts/manage_testnet.py deploy")
        return

    acct = Account.from_key(priv_key)
    balance_eth = w3.from_wei(w3.eth.get_balance(acct.address), 'ether')
    print(f"Signer Address:   {acct.address}")
    print(f"Contract Address: {deployed_address}")
    print(f"Account Balance:  {balance_eth:.6f} SepoliaETH")

    if balance_eth == 0:
        print("[!] Insufficient balance to pay for transaction gas.")
        return

    # 1. Load patient profile
    with open("data/demo_patients_master.json") as f:
        demo = json.load(f)

    if patient_key not in demo:
        patient_key = list(demo.keys())[0]

    patient = demo[patient_key]
    vitals = [float(patient["age"]), float(patient["systolic_bp"]), float(patient["cholesterol"]), float(patient["max_heart_rate"])]
    print(f"\nEvaluating Profile: {patient['name']}")
    print(f"Private Vitals:     Age {int(vitals[0])}, BP {int(vitals[1])}, Chol {int(vitals[2])}, Max HR {int(vitals[3])}")

    # 2. Run local ZK circuit inference & generate proof
    print("\n[*] Generating Zero-Knowledge Proof inside local BN254 Circuit...")
    prover = ZKMLProver("models/cardio_weights.json")
    proof_res = prover.prove_inference(vitals, secret_salt=42)

    pub = proof_res["public_inputs"]
    proof = proof_res["proof"]

    print(f"Diagnosis:          {pub['risk_label']} (Class {pub['predicted_class']})")
    print(f"Patient Commitment: {pub['patient_commitment']}")
    print(f"Proof Point A:      {proof['a'][0][:24]}...")

    # Format proof for Solidity:
    # a: uint256[2]
    # b: uint256[2][2]
    # c: uint256[2]
    # patientCommitment: bytes32
    # predictedClass: uint8
    a = [int(proof['a'][0], 16), int(proof['a'][1], 16)]
    b = [
        [int(proof['b'][0][0], 16), int(proof['b'][0][1], 16)],
        [int(proof['b'][1][0], 16), int(proof['b'][1][1], 16)]
    ]
    c = [int(proof['c'][0], 16), int(proof['c'][1], 16)]
    patient_commitment = bytes.fromhex(pub['patient_commitment'][2:])
    pred_class = int(pub['predicted_class'])

    # 3. Load contract ABI
    with open("build/ZKMedicalDiagnosisVerifier.json") as f:
        artifact = json.load(f)
    abi = artifact["abi"]

    contract = w3.eth.contract(address=deployed_address, abi=abi)

    # 4. Build and broadcast transaction
    nonce = w3.eth.get_transaction_count(acct.address)
    gas_price = w3.eth.gas_price

    print("\n[*] Building and signing verification transaction...")
    tx = contract.functions.verifyAndRecordDiagnosis(
        a, b, c, patient_commitment, pred_class
    ).build_transaction({
        'from': acct.address,
        'nonce': nonce,
        'gas': 250000,
        'gasPrice': int(gas_price * 1.15),
        'chainId': w3.eth.chain_id
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key=acct.key)
    print("Broadcasting transaction to Sepolia...")
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    tx_hex = tx_hash.hex()

    print(f"\n[+] Transaction Broadcasted!")
    print(f"  - Tx Hash:   {tx_hex}")
    print(f"  - Etherscan: https://sepolia.etherscan.io/tx/{tx_hex}")

    print("\nAwaiting block confirmation on Sepolia...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

    print(f"\n[SUCCESS] Diagnosis Verified & Recorded Permanently on Ethereum Sepolia!")
    print(f"  - Block Number: #{receipt.blockNumber}")
    print(f"  - Gas Used:     {receipt.gasUsed:,} gas")
    print(f"  - Status:       {'0x1 (SUCCESS)' if receipt.status == 1 else '0x0 (REVERTED)'}")
    print(f"  - View Proof On Etherscan: https://sepolia.etherscan.io/tx/{tx_hex}")

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "high_risk_bradycardia"
    verify_on_live_testnet(key)
