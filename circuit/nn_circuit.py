"""
Neural Network Arithmetic Circuit Compiler for zkML.
Translates neural network layers (Dense/Linear, Fixed-Point MatMul, Bias, ReLU)
into a verifiable R1CS constraint system over BN254.
"""

from typing import List, Tuple, Dict
import numpy as np
from circuit.field import FieldElement, BN254_PRIME
from circuit.quantizer import FixedPointQuantizer
from circuit.r1cs import R1CSCircuit, Variable, LinearCombination

class ZKNeuralNetworkCircuit:
    def __init__(self, quantizer: FixedPointQuantizer):
        self.quantizer = quantizer
        self.circuit = R1CSCircuit()
        self.input_vars: List[Variable] = []
        self.output_vars: List[Variable] = []
        self.weight_vars: List[Variable] = []
        self.layer_metadata: List[Dict] = []
        self.witness_map: Dict[int, FieldElement] = {}

    def set_witness_val(self, var: Variable, val: int):
        self.witness_map[var.index] = FieldElement(val, BN254_PRIME)

    def compile_model(self, layer_dims: List[int], weights: List[np.ndarray], biases: List[np.ndarray]):
        """
        layer_dims: e.g. [input_dim, hidden_dim, output_dim] -> [4, 8, 2]
        weights: List of numpy weight matrices [W1, W2]
        biases: List of numpy bias vectors [b1, b2]
        """
        # Step 1: Allocate constant ONE
        self.set_witness_val(self.circuit.one_var, 1)

        # Step 2: Allocate Private Patient Inputs
        self.input_vars = [
            self.circuit.allocate_private_input(f"patient_x_{i}")
            for i in range(layer_dims[0])
        ]

        current_activation_vars = self.input_vars

        for layer_idx, (W, b) in enumerate(zip(weights, biases)):
            in_dim, out_dim = W.shape[1], W.shape[0]
            is_last_layer = (layer_idx == len(weights) - 1)
            next_activation_vars = []

            # Quantize weights and biases
            q_W = np.round(W * self.quantizer.scale).astype(int)
            q_b = np.round(b * self.quantizer.scale).astype(int)

            for i in range(out_dim):
                # We want to compute z_raw = sum_j (W_ij * x_j) + b_i * S
                # Each W_ij * x_j is a multiplication constraint
                prod_vars = []
                for j in range(in_dim):
                    w_val = int(q_W[i, j])
                    w_var = self.circuit.allocate_private_input(f"W_{layer_idx}_{i}_{j}")
                    self.weight_vars.append(w_var)
                    self.set_witness_val(w_var, w_val)

                    prod_var = self.circuit.allocate_intermediate(f"prod_{layer_idx}_{i}_{j}")
                    # Enforce w_var * x_var == prod_var
                    self.circuit.enforce_mul(w_var, current_activation_vars[j], prod_var,
                                            f"W[{layer_idx}][{i},{j}] * act[{j}]")
                    prod_vars.append((prod_var, w_val))

                # Combine products and bias:
                # z_raw = sum(prod_vars) + b_i * scale
                # scale * z_unscaled + remainder == z_raw (fixed point rescale constraint)
                z_rescaled_var = self.circuit.allocate_intermediate(f"z_rescaled_{layer_idx}_{i}")
                rem_var = self.circuit.allocate_intermediate(f"rem_{layer_idx}_{i}")

                # Linear combination for z_raw:
                lc_z_raw_terms = {p_var.index: 1 for p_var, _ in prod_vars}
                # Add bias at scale S^2: b_i * scale
                bias_scaled = int(q_b[i] * self.quantizer.scale)
                if bias_scaled != 0:
                    lc_z_raw_terms[self.circuit.one_var.index] = bias_scaled

                # Enforce: 1 * (z_raw - rem) == scale * z_rescaled
                # which is: 1 * (sum(prod) + bias - rem) == scale * z_rescaled
                # Formulate as: (ONE) * (sum(prod) + bias - rem) == (scale * z_rescaled)
                lc_left = LinearCombination({self.circuit.one_var.index: 1})
                lc_mid = LinearCombination(lc_z_raw_terms)
                lc_mid.add_term(rem_var.index, -1)
                lc_right = LinearCombination({z_rescaled_var.index: self.quantizer.scale})

                self.circuit.add_constraint(lc_left, lc_mid, lc_right, f"rescale_layer_{layer_idx}_neuron_{i}")

                if is_last_layer:
                    # Final layer: directly output logits (as public outputs)
                    out_var = self.circuit.allocate_public_output(f"logit_{i}")
                    # Enforce 1 * z_rescaled == out_var
                    self.circuit.enforce_equal_lc(LinearCombination({z_rescaled_var.index: 1}), out_var)
                    next_activation_vars.append(out_var)
                else:
                    # Hidden layer: apply ReLU activation
                    # relu_out = sign_bit * z_rescaled
                    sign_var = self.circuit.allocate_intermediate(f"sign_{layer_idx}_{i}")
                    relu_var = self.circuit.allocate_intermediate(f"relu_{layer_idx}_{i}")

                    # 1. Enforce boolean: sign * (1 - sign) == 0
                    self.circuit.enforce_boolean(sign_var, f"sign_is_bool_{layer_idx}_{i}")
                    # 2. Enforce: sign * z_rescaled == relu_var
                    self.circuit.enforce_mul(sign_var, z_rescaled_var, relu_var, f"relu_activation_{layer_idx}_{i}")

                    next_activation_vars.append(relu_var)

            current_activation_vars = next_activation_vars

        self.output_vars = current_activation_vars
        return self.circuit

    def compute_witness(self, patient_inputs: List[float], weights: List[np.ndarray], biases: List[np.ndarray]) -> List[FieldElement]:
        """
        Executes concrete forward pass and populates every single witness wire.
        Returns full witness vector ready for verification or proof generation.
        """
        # Assign patient inputs
        q_inputs = self.quantizer.quantize_vector(patient_inputs)
        for var, val in zip(self.input_vars, q_inputs):
            self.set_witness_val(var, val)

        current_activations = q_inputs

        for layer_idx, (W, b) in enumerate(zip(weights, biases)):
            in_dim, out_dim = W.shape[1], W.shape[0]
            is_last_layer = (layer_idx == len(weights) - 1)
            next_activations = []

            q_W = np.round(W * self.quantizer.scale).astype(int)
            q_b = np.round(b * self.quantizer.scale).astype(int)

            for i in range(out_dim):
                # Compute raw dot product at scale S^2
                z_raw = int(q_b[i] * self.quantizer.scale)
                for j in range(in_dim):
                    w_val = int(q_W[i, j])
                    x_val = current_activations[j]
                    prod_val = w_val * x_val
                    prod_var = self.circuit.var_names[f"prod_{layer_idx}_{i}_{j}"]
                    self.set_witness_val(prod_var, prod_val)
                    z_raw += prod_val

                # Rescale: z_rescaled = z_raw // scale, remainder = z_raw % scale
                z_rescaled = z_raw // self.quantizer.scale
                remainder = z_raw % self.quantizer.scale

                z_rescaled_var = self.circuit.var_names[f"z_rescaled_{layer_idx}_{i}"]
                rem_var = self.circuit.var_names[f"rem_{layer_idx}_{i}"]
                self.set_witness_val(z_rescaled_var, z_rescaled)
                self.set_witness_val(rem_var, remainder)

                if is_last_layer:
                    out_var = self.circuit.var_names[f"logit_{i}"]
                    self.set_witness_val(out_var, z_rescaled)
                    next_activations.append(z_rescaled)
                else:
                    sign_val = 1 if z_rescaled > 0 else 0
                    relu_val = z_rescaled if sign_val == 1 else 0

                    sign_var = self.circuit.var_names[f"sign_{layer_idx}_{i}"]
                    relu_var = self.circuit.var_names[f"relu_{layer_idx}_{i}"]
                    self.set_witness_val(sign_var, sign_val)
                    self.set_witness_val(relu_var, relu_val)
                    next_activations.append(relu_val)

            current_activations = next_activations

        # Build ordered witness array
        witness = [FieldElement(0)] * len(self.circuit.variables)
        for idx in range(len(self.circuit.variables)):
            witness[idx] = self.witness_map.get(idx, FieldElement(0))

        return witness
