"""
Testnet Wallet & Deployment Manager.
- Generates or loads a dedicated testnet account.
- Queries live Sepolia network balance.
- Provides faucet guidance.
- Deploys `ZKMedicalDiagnosisVerifier.sol` to Sepolia.
"""

import os
import sys
import json
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account

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

def init_wallet():
    """Generates a dedicated testnet wallet if one doesn't already exist."""
    env_file = ".env"
    priv_key = os.getenv("TESTNET_PRIVATE_KEY")
    
    if not priv_key:
        print("[*] Generating fresh dedicated testnet wallet...")
        acct = Account.create()
        priv_key = acct.key.hex()
        address = acct.address

        with open(env_file, "a") as f:
            f.write(f"\nSEPOLIA_RPC_URL={DEFAULT_SEPOLIA_RPC}\n")
            f.write(f"TESTNET_PRIVATE_KEY={priv_key}\n")
            f.write(f"TESTNET_ADDRESS={address}\n")
        print(f"[+] Wallet generated and saved to '.env'!")
    else:
        acct = Account.from_key(priv_key)
        address = acct.address

    return acct, address

def check_status():
    print("=" * 75)
    print("         [+] ETHEREUM SEPOLIA TESTNET STATUS & WALLET")
    print("=" * 75)

    w3, rpc = get_web3()
    is_connected = w3.is_connected()
    print(f"RPC URL:         {rpc}")
    print(f"Connected:       {'SUCCESS (Connected to Sepolia)' if is_connected else 'FAILED (Cannot reach RPC)'}")

    if not is_connected:
        print("[!] Could not connect to Sepolia RPC. Check your internet connection or use an alternative RPC.")
        return

    chain_id = w3.eth.chain_id
    latest_block = w3.eth.block_number
    gas_price_gwei = w3.from_wei(w3.eth.gas_price, 'gwei')

    print(f"Chain ID:        {chain_id} (Ethereum Sepolia Testnet)")
    print(f"Latest Block:    #{latest_block:,}")
    print(f"Current Gas:     {gas_price_gwei:.2f} Gwei")

    acct, address = init_wallet()
    balance_wei = w3.eth.get_balance(address)
    balance_eth = w3.from_wei(balance_wei, 'ether')

    print(f"\nTestnet Address: {address}")
    print(f"Account Balance: {balance_eth:.6f} SepoliaETH")

    deployed_addr = os.getenv("DEPLOYED_CONTRACT_ADDRESS")
    if deployed_addr:
        print(f"Deployed Contract: {deployed_addr}")
        print(f"Etherscan Link:    https://sepolia.etherscan.io/address/{deployed_addr}")

    if balance_eth == 0:
        print("\n" + "-" * 75)
        print("[!] YOUR WALLET NEEDS TESTNET ETH FOR GAS BEFORE DEPLOYING")
        print("-" * 75)
        print(f"1. Copy your Testnet Address:")
        print(f"   {address}\n")
        print("2. Paste it into any of these free 1-click Sepolia faucets to claim free test ETH:")
        print("   - https://sepolia-faucet.pk910.de/ (PoW faucet, unlimited, works worldwide)")
        print("   - https://cloud.google.com/application/web3/faucet/ethereum/sepolia")
        print("   - https://faucets.chain.link/sepolia")
        print("   - https://www.alchemy.com/faucets/ethereum-sepolia")
        print("-" * 75)
    else:
        print("\n[+] Wallet has sufficient SepoliaETH to deploy and verify!")

    return balance_eth > 0

def deploy():
    print("=" * 75)
    print("        [+] DEPLOYING zkML VERIFIER TO ETHEREUM SEPOLIA")
    print("=" * 75)

    w3, rpc = get_web3()
    if not w3.is_connected():
        print("[!] Cannot connect to Sepolia. Aborting deployment.")
        return

    acct, address = init_wallet()
    balance_wei = w3.eth.get_balance(address)
    balance_eth = w3.from_wei(balance_wei, 'ether')

    if balance_eth == 0:
        print(f"[!] Balance is 0 SepoliaETH. Please fund wallet {address} from a faucet first.")
        return

    # Load compiled artifacts
    artifact_path = "build/ZKMedicalDiagnosisVerifier.json"
    if not os.path.exists(artifact_path):
        print("[*] Compiling contract first...")
        from scripts.compile_contract import compile_zk_contract
        compile_zk_contract()

    with open(artifact_path) as f:
        artifact = json.load(f)

    abi = artifact["abi"]
    bytecode = artifact["bytecode"]

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    # Build constructor transaction
    nonce = w3.eth.get_transaction_count(address)
    gas_price = w3.eth.gas_price
    
    print(f"\nDeploying from:   {address}")
    print(f"Account nonce:    {nonce}")
    print(f"Gas Price:        {w3.from_wei(gas_price, 'gwei'):.2f} Gwei")

    print("\nEstimating deployment gas...")
    try:
        estimated_gas = contract.constructor().estimate_gas({'from': address})
        print(f"Estimated Gas:    {estimated_gas:,} gas")
    except Exception as e:
        print(f"Gas estimation fallback (e.g. 1,500,000 gas): {e}")
        estimated_gas = 1500000

    deploy_tx = contract.constructor().build_transaction({
        'from': address,
        'nonce': nonce,
        'gas': int(estimated_gas * 1.2),
        'gasPrice': int(gas_price * 1.1),
        'chainId': w3.eth.chain_id
    })

    # Sign transaction locally with private key
    print("Signing deployment transaction...")
    signed_tx = w3.eth.account.sign_transaction(deploy_tx, private_key=acct.key)

    # Broadcast transaction to the Ethereum Sepolia network
    print("Broadcasting transaction to Sepolia mempool...")
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    tx_hex = tx_hash.hex()
    print(f"\n[+] Transaction Sent!")
    print(f"  - Tx Hash:   {tx_hex}")
    print(f"  - Etherscan: https://sepolia.etherscan.io/tx/{tx_hex}")

    print("\nWaiting for block confirmation (this takes ~12-15 seconds on Sepolia)...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

    contract_address = tx_receipt.contractAddress
    print(f"\n[SUCCESS] Contract Deployed Successfully!")
    print(f"  - Contract Address: {contract_address}")
    print(f"  - Block Included:   #{tx_receipt.blockNumber}")
    print(f"  - Gas Used:         {tx_receipt.gasUsed:,} gas")
    print(f"  - Explorer Link:    https://sepolia.etherscan.io/address/{contract_address}")

    # Save deployed address to .env
    with open(".env", "a") as f:
        f.write(f"\nDEPLOYED_CONTRACT_ADDRESS={contract_address}\n")
    print("\nSaved DEPLOYED_CONTRACT_ADDRESS to '.env'!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        deploy()
    else:
        check_status()
