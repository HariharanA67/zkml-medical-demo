"""
Rank-1 Constraint System (R1CS) Engine for zkML.
Enforces (A * s) * (B * s) = (C * s) over finite field BN254.
Supports:
- Linear combinations & additions
- Multiplications (dense & quantized)
- Rescaling constraints
- ReLU activation gates
- Public vs Private variable assignments
"""

from typing import Dict, List, Tuple, Optional
from circuit.field import FieldElement, BN254_PRIME

class Variable:
    def __init__(self, name: str, index: int, is_public: bool = False):
        self.name = name
        self.index = index
        self.is_public = is_public

    def __repr__(self):
        return f"Var({self.name}, idx={self.index}, pub={self.is_public})"

class LinearCombination:
    """Represents a sum of terms: sum(coeff_i * var_i)."""
    def __init__(self, terms: Optional[Dict[int, FieldElement]] = None):
        # map var_index -> FieldElement
        self.terms: Dict[int, FieldElement] = {}
        if terms:
            for idx, coeff in terms.items():
                c = coeff if isinstance(coeff, FieldElement) else FieldElement(coeff)
                if c.value != 0:
                    self.terms[idx] = c

    def add_term(self, var_index: int, coeff: int):
        c = coeff if isinstance(coeff, FieldElement) else FieldElement(coeff)
        if c.value == 0:
            return
        if var_index in self.terms:
            new_c = self.terms[var_index] + c
            if new_c.value == 0:
                del self.terms[var_index]
            else:
                self.terms[var_index] = new_c
        else:
            self.terms[var_index] = c

    def evaluate(self, witness: List[FieldElement]) -> FieldElement:
        """Computes the dot product of this linear combination with the witness vector."""
        total = FieldElement(0)
        for idx, coeff in self.terms.items():
            total = total + (coeff * witness[idx])
        return total

    def __repr__(self):
        parts = [f"{coeff}*w[{idx}]" for idx, coeff in self.terms.items()]
        return " + ".join(parts) if parts else "0"

class Constraint:
    """Represents one R1CS constraint: (A * s) * (B * s) = (C * s)."""
    def __init__(self, a: LinearCombination, b: LinearCombination, c: LinearCombination, description: str = ""):
        self.a = a
        self.b = b
        self.c = c
        self.description = description

    def is_satisfied(self, witness: List[FieldElement]) -> bool:
        val_a = self.a.evaluate(witness)
        val_b = self.b.evaluate(witness)
        val_c = self.c.evaluate(witness)
        return (val_a * val_b) == val_c

    def __repr__(self):
        return f"({self.a}) * ({self.b}) == ({self.c})"

class R1CSCircuit:
    def __init__(self):
        self.variables: List[Variable] = []
        self.var_names: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        
        # Variable 0 is always the constant 1 (w[0] = 1)
        self.one_var = self._create_var("ONE", is_public=True)

    def _create_var(self, name: str, is_public: bool = False) -> Variable:
        idx = len(self.variables)
        var = Variable(name, idx, is_public)
        self.variables.append(var)
        self.var_names[name] = var
        return var

    def allocate_public_input(self, name: str) -> Variable:
        return self._create_var(name, is_public=True)

    def allocate_private_input(self, name: str) -> Variable:
        return self._create_var(name, is_public=False)

    def allocate_intermediate(self, name: str) -> Variable:
        return self._create_var(name, is_public=False)

    def allocate_public_output(self, name: str) -> Variable:
        return self._create_var(name, is_public=True)

    def add_constraint(self, a: LinearCombination, b: LinearCombination, c: LinearCombination, desc: str = ""):
        constraint = Constraint(a, b, c, desc)
        self.constraints.append(constraint)
        return constraint

    def enforce_mul(self, var_a: Variable, var_b: Variable, var_out: Variable, desc: str = ""):
        """Enforces var_a * var_b = var_out."""
        lc_a = LinearCombination({var_a.index: 1})
        lc_b = LinearCombination({var_b.index: 1})
        lc_c = LinearCombination({var_out.index: 1})
        return self.add_constraint(lc_a, lc_b, lc_c, desc or f"{var_a.name} * {var_b.name} == {var_out.name}")

    def enforce_boolean(self, var_bit: Variable, desc: str = ""):
        """Enforces var_bit * (1 - var_bit) = 0, proving var_bit in {0, 1}."""
        lc_a = LinearCombination({var_bit.index: 1})
        lc_b = LinearCombination({self.one_var.index: 1, var_bit.index: -1})
        lc_c = LinearCombination({})
        return self.add_constraint(lc_a, lc_b, lc_c, desc or f"boolean({var_bit.name})")

    def enforce_equal_lc(self, lc: LinearCombination, var_out: Variable, desc: str = ""):
        """Enforces 1 * (LinearCombination) = var_out."""
        lc_a = LinearCombination({self.one_var.index: 1})
        lc_b = lc
        lc_c = LinearCombination({var_out.index: 1})
        return self.add_constraint(lc_a, lc_b, lc_c, desc or f"lc == {var_out.name}")

    def verify_witness(self, witness: List[FieldElement]) -> Tuple[bool, List[int]]:
        """Checks if all constraints are satisfied by the given witness."""
        failed = []
        if len(witness) != len(self.variables):
            raise ValueError(f"Witness size ({len(witness)}) does not match circuit variable count ({len(self.variables)})")
        
        # Check w[0] == 1
        if witness[0].value != 1:
            failed.append(-1)

        for i, c in enumerate(self.constraints):
            if not c.is_satisfied(witness):
                failed.append(i)
        return (len(failed) == 0, failed)
