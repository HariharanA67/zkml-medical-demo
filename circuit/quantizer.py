"""
Fixed-Point Quantization Engine for zkML.
Maps real floating-point numbers (weights, inputs, biases) into integer representations
suitable for finite field arithmetic circuits.
"""

from typing import Union, List
import math
from circuit.field import FieldElement, BN254_PRIME

class FixedPointQuantizer:
    def __init__(self, scale_bits: int = 10, prime: int = BN254_PRIME):
        """
        scale_bits: The number of bits for fractional precision.
                    scale S = 2^scale_bits. (e.g. 10 bits -> S = 1024, ~0.00097 precision)
        """
        self.scale_bits = scale_bits
        self.scale = 1 << scale_bits  # 2^scale_bits
        self.prime = prime

    def quantize(self, val: Union[float, int]) -> int:
        """Convert float -> signed integer."""
        return int(round(val * self.scale))

    def dequantize(self, quantized_int: Union[int, FieldElement]) -> float:
        """Convert signed integer or FieldElement -> float."""
        if isinstance(quantized_int, FieldElement):
            val = quantized_int.to_signed()
        else:
            val = quantized_int
        return float(val) / self.scale

    def to_field(self, val: Union[float, int]) -> FieldElement:
        """Quantize float and wrap in FieldElement."""
        q = self.quantize(val)
        return FieldElement(q, self.prime)

    def rescale(self, product: int) -> int:
        """
        Rescale after a multiplication:
        product is at scale S^2, divide by S to return to scale S.
        """
        return product // self.scale

    def quantize_vector(self, vec: List[float]) -> List[int]:
        return [self.quantize(x) for x in vec]

    def dequantize_vector(self, vec: List[Union[int, FieldElement]]) -> List[float]:
        return [self.dequantize(x) for x in vec]
