import math
import unittest

from quantization.quantize import calc_scale, calc_zero_point, quantize, dequantize, quantize_all

x = [-3.0, 0.1, 3.2, -3.0, -0.3, 3.2, -2.0, 0.2, 2.0, -1.0, 0.1, 1.0, -3.2, 1.0, 3.0]

class test_quantize(unittest.TestCase):
    def test_quantize_all(self):
        _type = "uint8"
        print("quantize_all", quantize_all(x, _type))

    def test_quantize(self):
        _type = "uint8"
        scale_molecule, scale_denominator = calc_scale(x, _type)
        scale = math.ceil(scale_molecule / scale_denominator)
        _zero_point = calc_zero_point(x, scale, _type)
        scale = scale_molecule / scale_denominator
        print("quantize", quantize(x, scale, _zero_point, _type))

    def test_dequantize(self):
        _type = "uint8"
        scale_molecule, scale_denominator = calc_scale(x, _type)
        scale = math.ceil(scale_molecule / scale_denominator)
        zero_point = calc_zero_point(x, scale, _type)
        print("dequantize", dequantize(
            quantize(x, scale, zero_point, _type),
            scale,
            zero_point,
            _type))


if __name__ == '__main__':
    unittest.main()
