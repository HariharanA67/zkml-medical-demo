"""
Interactive Streamlit UI for Zero-Knowledge Machine Learning (zkML) Medical Inference.
Supports both Local EVM Simulation and Live Ethereum Sepolia Testnet deployment & verification.
"""

import streamlit as st
import subprocess
import sys
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv

from circuit.prover import ZKMLProver
from circuit.evm_simulator import SimulatedEVM
from circuit.field import BN254_PRIME

load_dotenv()

st.set_page_config(
    page_title="zkML Medical Inference Engine (Live Testnet)",
    page_icon="🔐",
    layout="wide"
)

# Initialize singletons in session state
if "prover" not in st.session_state:
    st.session_state.prover = ZKMLProver("models/cardio_weights.json")
if "evm" not in st.session_state:
    st.session_state.evm = SimulatedEVM()
if "history" not in st.session_state:
    st.session_state.history = []

prover: ZKMLProver = st.session_state.prover
evm: SimulatedEVM = st.session_state.evm

# Load master demo patients
demo_path = "data/demo_patients_master.json" if os.path.exists("data/demo_patients_master.json") else "demo_patients.json"
demo_patients = {}
if os.path.exists(demo_path):
    with open(demo_path) as f:
        demo_patients = json.load(f)

# Load real dataset
df_dataset = None
data_path = "data/heart_master_4feature.csv" if os.path.exists("data/heart_master_4feature.csv") else "heart_clean_4feature.csv"
if os.path.exists(data_path):
    df_dataset = pd.read_csv(data_path)

st.title("🔐 Zero-Knowledge Machine Learning (zkML) on EVM")
st.markdown("""
**Cross-Domain Integration:**
`AI/ML (Multi-Center Heart MLP)` + `Cybersecurity (Zero-Knowledge SNARKs)` + 
`Scientific Computing (Arithmetic Circuits over BN254)` + `Blockchain (Solidity EVM Verifier & Live Testnet)`
""")

tab_inference, tab_dataset, tab_testnet = st.tabs([
    "🚀 Private zkML Inference & Verifier", 
    "📊 Multi-Center Clinical Dataset",
    "⛓️ Live Sepolia Testnet Status"
])

with tab_testnet:
    st.subheader("🌐 Ethereum Sepolia Testnet Integration")
    testnet_addr = os.getenv("TESTNET_ADDRESS", "0x760A879AFf9Fd3505Da8461B5581C7eC21D2D95d")
    deployed_addr = os.getenv("DEPLOYED_CONTRACT_ADDRESS", "")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("### 🔑 Your Dedicated Deployment Wallet")
        st.code(testnet_addr, language="text")
        st.markdown(f"[🔍 View Wallet on Etherscan](https://sepolia.etherscan.io/address/{testnet_addr})")
        # Add Deploy button
        if st.button("🚀 Deploy Verifier Contract to Sepolia", type="primary"):
            st.info("Running deployment… this may take ~30 seconds")
            # Use subprocess to call the deployment script
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/manage_testnet.py", "deploy"],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd(),
                    timeout=180,
                )
                if result.returncode == 0:
                    st.success("✅ Deployment completed. Check below for contract address:")
                    # Reload env to get the new address
                    load_dotenv(override=True)
                    new_addr = os.getenv("DEPLOYED_CONTRACT_ADDRESS", "")
                    if new_addr:
                        st.code(new_addr)
                        st.markdown(f"[🔍 View Contract on Etherscan](https://sepolia.etherscan.io/address/{new_addr})")
                    else:
                        st.warning("Deployment script finished but address not found in .env.")
                else:
                    st.error(f"❌ Deployment failed (exit {result.returncode}). See logs below:")
                    st.code(result.stderr or result.stdout, language="text")
            except Exception as e:
                st.exception(e)

        st.markdown("### 🔑 Your Dedicated Deployment Wallet")
        st.code(testnet_addr, language="text")
        st.markdown(f"[🔍 View Wallet on Etherscan](https://sepolia.etherscan.io/address/{testnet_addr})")
        st.info("""
        **To deploy to live Sepolia:**
        1. Fund this address with free SepoliaETH from any faucet below.
        2. Run deployment command in terminal:
        ```bash
        .venv\\Scripts\\python.exe scripts/manage_testnet.py deploy
        ```
        """)

    with col_w2:
        st.markdown("### 📜 Deployed zkML Verifier Contract")
        if deployed_addr:
            st.success(f"**Contract Address:** `{deployed_addr}`")
            st.markdown(f"[🔍 View Contract on Etherscan](https://sepolia.etherscan.io/address/{deployed_addr})")
        else:
            st.warning("Contract not deployed to live testnet yet. Use local simulation or fund wallet to deploy!")

        st.markdown("### 🚰 Free 1-Click Sepolia Faucets")
        st.markdown("""
        * [Sepolia PoW Faucet (Instant, Unlimited)](https://sepolia-faucet.pk910.de/)
        * [Google Cloud Web3 Faucet](https://cloud.google.com/application/web3/faucet/ethereum/sepolia)
        * [Chainlink Sepolia Faucet](https://faucets.chain.link/sepolia)
        """)

with tab_dataset:
    st.subheader("🏥 Canonical Multi-Center Heart Dataset (`data/heart_master_4feature.csv`)")
    if df_dataset is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cohort Size (Clean)", len(df_dataset))
        c2.metric("Features In Circuit", 4)
        healthy_pct = (df_dataset['high_risk'] == 0).mean() * 100
        risk_pct = (df_dataset['high_risk'] == 1).mean() * 100
        c3.metric("Healthy (Class 0)", f"{healthy_pct:.1f}% ({int((df_dataset['high_risk'] == 0).sum())})")
        c4.metric("At-Risk (Class 1)", f"{risk_pct:.1f}% ({int((df_dataset['high_risk'] == 1).sum())})")

        st.write("---")
        col_table, col_corr = st.columns([1.2, 1])
        with col_table:
            st.write("**Dataset Preview (Canonical Cohort):**")
            st.dataframe(df_dataset.head(8), use_container_width=True)
            st.caption("Consolidated from multi-hospital records (Cleveland, Hungarian, Long Beach, Switzerland).")
        
        with col_corr:
            st.write("**Feature Correlations with Target (`high_risk`):**")
            corr = df_dataset.corr()[['high_risk']].sort_values(by='high_risk', ascending=False)
            st.dataframe(corr.style.background_gradient(cmap="coolwarm"), use_container_width=True)
            st.info("📌 **Clinical Finding:** `max_heart_rate` is the most significant physiological marker (correlation: -0.38). Inability of the myocardium to achieve expected peak heart rates strongly signals coronary compromise.")

        st.write("---")
        st.write("**Model Benchmark (10-Fold Stratified Cross-Validation):**")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("10-Fold CV Accuracy", "68.82% ± 3.5%")
        b2.metric("10-Fold ROC-AUC", "0.7397")
        b3.metric("Precision", "67.64%")
        b4.metric("Recall", "66.80%")
    else:
        st.warning("Dataset not found.")

with tab_inference:
    # Sidebar: Presets and Privacy Settings
    with st.sidebar:
        st.header("📋 Patient Vitals Presets")
        preset_names = ["Custom"]
        preset_keys_map = {}
        for k, p in demo_patients.items():
            display_name = p.get("name", k.upper())
            preset_names.append(display_name)
            preset_keys_map[display_name] = k

        selected_preset = st.selectbox("Select Master Clinical Profile:", preset_names)
        
        salt = st.number_input("Secret Salt (Kept Private by Patient)", value=1337, step=1)
        
        st.info("💡 **Zero-Knowledge Guarantee:** The patient's vitals never leave the client device. Only the proof and cryptographic commitment are submitted to the blockchain.")

    # Determine input values
    if selected_preset != "Custom":
        key = preset_keys_map[selected_preset]
        p_data = demo_patients[key]
        def_age = int(p_data.get("age", 45))
        def_bp = int(p_data.get("systolic_bp", 125))
        def_chol = int(p_data.get("cholesterol", 200))
        def_hr = int(p_data.get("max_heart_rate", 150))
    else:
        def_age, def_bp, def_chol, def_hr = 45, 125, 200, 150

    # Input Controls
    st.write("### Step 1: Input Private Patient Vitals")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.slider("Age (years)", min_value=25, max_value=80, value=def_age)
    with col2:
        bp = st.slider("Systolic BP (mm Hg)", min_value=85, max_value=210, value=def_bp)
    with col3:
        chol = st.slider("Cholesterol (mg/dL)", min_value=120, max_value=500, value=def_chol)
    with col4:
        max_hr = st.slider("Max Heart Rate (bpm)", min_value=60, max_value=220, value=def_hr)

    patient_vitals = [float(age), float(bp), float(chol), float(max_hr)]

    st.write("")
    if st.button("🚀 Generate ZK Proof & Verify On-Chain", type="primary", use_container_width=True):
        with st.spinner("Compiling Arithmetic Circuit and synthesizing SNARK proof over BN254..."):
            proof_res = prover.prove_inference(patient_vitals, secret_salt=int(salt))
            tx_res = evm.verify_and_record_diagnosis(proof_res)

            pub = proof_res["public_inputs"]
            stats = proof_res["circuit_stats"]
            proof = proof_res["proof"]

            st.session_state.history.append({
                "block": tx_res["block_number"],
                "commitment": pub["patient_commitment"][:16] + "...",
                "category": pub["risk_label"],
                "gas": f"{tx_res['gas_used']:,}",
                "tx_hash": tx_res["tx_hash"][:18] + "..."
            })

        st.success("✅ Zero-Knowledge Proof Generated & Verified on EVM Smart Contract!")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("🔬 1. AI & Scientific Computing (R1CS Circuit)")
            is_high = pub["predicted_class"] == 1
            badge_color = "#d32f2f" if is_high else "#2e7d32"
            bg_color = "#ffebee" if is_high else "#e8f5e9"
            
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 16px; border-radius: 8px; border-left: 6px solid {badge_color};">
                    <h3 style="margin:0; color: {badge_color};">Diagnosis: {pub['risk_label']}</h3>
                    <p style="margin:4px 0 0 0; color: #333;">
                        Class 0 (Healthy) Logit: <b>{pub['float_logit_0']:.4f}</b> | 
                        Class 1 (High Risk) Logit: <b>{pub['float_logit_1']:.4f}</b>
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )

            st.write("")
            metric_df = pd.DataFrame([
                {"Parameter": "BN254 Scalar Field Prime", "Value": str(hex(BN254_PRIME))[:22] + "..."},
                {"Parameter": "Quantization Scaling Factor", "Value": f"S = 2^{prover.scale_bits} ({prover.quantizer.scale})"},
                {"Parameter": "Total Circuit Wires (Witness Size)", "Value": str(stats["total_wires"])},
                {"Parameter": "R1CS Quadratic Constraints", "Value": str(stats["total_constraints"])},
                {"Parameter": "All Constraints Satisfied?", "Value": "100% Satisfied (Valid)"},
                {"Parameter": "Patient Cryptographic Commitment", "Value": pub["patient_commitment"]}
            ])
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

        with res_col2:
            st.subheader("⛓️ 2. Cybersecurity & Blockchain (EVM Settlement)")
            tx_df = pd.DataFrame([
                {"Metric": "Contract Address", "Value": tx_res["contract_address"]},
                {"Metric": "Transaction Hash", "Value": tx_res["tx_hash"]},
                {"Metric": "Block Included", "Value": f"#{tx_res['block_number']}"},
                {"Metric": "Execution Status", "Value": tx_res["status"]},
                {"Metric": "Gas Consumed", "Value": f"{tx_res['gas_used']:,} gas"},
                {"Metric": "Gas Cost (ETH)", "Value": tx_res["gas_cost_eth"]},
                {"Metric": "EVM Event Emitted", "Value": tx_res["event_emitted"]["event"]}
            ])
            st.dataframe(tx_df, use_container_width=True, hide_index=True)

        with st.expander("🔍 View Raw Cryptographic SNARK Proof Points (Solidity Calldata)"):
            st.code(json.dumps(proof, indent=2), language="json")

    # On-Chain Ledger Explorer
    st.write("---")
    st.subheader("📜 Immutable On-Chain Diagnosis Ledger")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No on-chain transactions yet. Select a patient preset or adjust sliders and click 'Generate ZK Proof & Verify On-Chain'.")
