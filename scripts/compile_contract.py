"""
Compiles `contracts/ZKMedicalDiagnosisVerifier.sol` using `py-solc-x`.
Installs the solc compiler binary automatically if not present.
Saves the compiled ABI and Bytecode to `build/ZKMedicalDiagnosisVerifier.json`.
"""

import os
import json
import solcx

def compile_zk_contract():
    print("=" * 70)
    print("        [+] COMPILING SOLIDITY zkML VERIFIER CONTRACT")
    print("=" * 70)

    # 1. Ensure solc compiler version 0.8.20 is installed
    solc_version = "0.8.20"
    installed_versions = [str(v) for v in solcx.get_installed_solc_versions()]
    print(f"Installed solc versions: {installed_versions}")

    if solc_version not in installed_versions:
        print(f"Downloading & installing solc v{solc_version}...")
        solcx.install_solc(solc_version)
        print(f"Successfully installed solc v{solc_version}!")

    solcx.set_solc_version(solc_version)

    # 2. Compile contract
    contract_file = "contracts/ZKMedicalDiagnosisVerifier.sol"
    print(f"\nCompiling '{contract_file}'...")
    
    with open(contract_file, "r") as f:
        source_code = f.read()

    compiled = solcx.compile_source(
        source_code,
        output_values=["abi", "bin"],
        solc_version=solc_version
    )

    contract_id = "<stdin>:ZKMedicalDiagnosisVerifier"
    if contract_id not in compiled:
        # Fallback search
        for k in compiled.keys():
            if "ZKMedicalDiagnosisVerifier" in k:
                contract_id = k
                break

    abi = compiled[contract_id]["abi"]
    bytecode = compiled[contract_id]["bin"]

    os.makedirs("build", exist_ok=True)
    output_path = "build/ZKMedicalDiagnosisVerifier.json"
    with open(output_path, "w") as f:
        json.dump({
            "contractName": "ZKMedicalDiagnosisVerifier",
            "abi": abi,
            "bytecode": bytecode
        }, f, indent=2)

    print(f"\n[SUCCESS] Compilation complete!")
    print(f"  - Artifact: {output_path}")
    print(f"  - Bytecode size: {len(bytecode) // 2:,} bytes")
    print(f"  - Functions in ABI: {len([x for x in abi if x.get('type') == 'function'])}")
    for item in abi:
        if item.get("type") == "function":
            print(f"    * {item['name']}()")

if __name__ == "__main__":
    compile_zk_contract()
