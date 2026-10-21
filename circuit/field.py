"""
Finite Field arithmetic over the BN254 (alt_bn128) scalar field.
This prime is natively supported by Ethereum EVM via EIP-196/197 precompiles.
"""

BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

class FieldElement:
    """Represents an element of F_p."""
    def __init__(self, value: int, prime: int = BN254_PRIME):
        self.prime = prime
        self.value = value % prime

    def __add__(self, other):
        other_val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value + other_val) % self.prime, self.prime)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value - other_val) % self.prime, self.prime)

    def __rsub__(self, other):
        other_val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((other_val - self.value) % self.prime, self.prime)

    def __mul__(self, other):
        other_val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value * other_val) % self.prime, self.prime)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_val = other.value if isinstance(other, FieldElement) else other
        if other_val % self.prime == 0:
            raise ZeroDivisionError("Division by zero in finite field")
        # Modular inverse using Fermat's Little Theorem: a^(p-2) mod p
        inv = pow(other_val, self.prime - 2, self.prime)
        return FieldElement((self.value * inv) % self.prime, self.prime)

    def __neg__(self):
        return FieldElement((-self.value) % self.prime, self.prime)

    def __eq__(self, other):
        other_val = other.value if isinstance(other, FieldElement) else other
        return self.value == (other_val % self.prime)

    def __int__(self):
        return self.value

    def to_signed(self) -> int:
        """
        Converts the field element back to a signed integer if it falls in the upper half of the field.
        e.g., p - 5 -> -5.
        """
        if self.value > self.prime // 2:
            return self.value - self.prime
        return self.value

    def __repr__(self):
        signed = self.to_signed()
        if abs(signed) < 100000000:
            return f"F({signed})"
        return f"F({hex(self.value)})"
