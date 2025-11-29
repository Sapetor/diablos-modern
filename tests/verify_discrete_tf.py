
import unittest
import numpy as np
from lib import functions

class TestDiscreteTransferFunction(unittest.TestCase):
    def test_step_response(self):
        # H(z) = z / (z - 0.5)
        # y[k] = 0.5 y[k-1] + u[k]
        # Step input u[k] = 1
        # y[0] = 1
        # y[1] = 1.5
        # y[2] = 1.75
        
        params = {
            'numerator': [1.0, 0.0],
            'denominator': [1.0, -0.5],
            '_init_start_': True,
            '_name_': 'TestBlock',
            'init_conds': [0.0]
        }
        
        # Time 0
        inputs = {0: 1.0}
        out = functions.discrete_transfer_function(0.0, inputs, params)
        self.assertAlmostEqual(out[0], 1.0)
        
        # Time 1
        inputs = {0: 1.0}
        out = functions.discrete_transfer_function(1.0, inputs, params)
        self.assertAlmostEqual(out[0], 1.5)
        
        # Time 2
        inputs = {0: 1.0}
        out = functions.discrete_transfer_function(2.0, inputs, params)
        self.assertAlmostEqual(out[0], 1.75)

    def test_strictly_proper(self):
        # H(z) = 1 / (z - 0.5) = z^-1 / (1 - 0.5 z^-1)
        # y[k] = 0.5 y[k-1] + u[k-1]
        # Step input u[k] = 1
        # y[0] = 0 (strictly proper)
        # y[1] = 1
        # y[2] = 0.5(1) + 1 = 1.5
        
        params = {
            'numerator': [1.0],
            'denominator': [1.0, -0.5],
            '_init_start_': True,
            '_name_': 'TestBlockStrictlyProper',
            'init_conds': [0.0]
        }
        
        # Time 0
        inputs = {0: 1.0}
        out = functions.discrete_transfer_function(0.0, inputs, params)
        self.assertAlmostEqual(out[0], 0.0)
        
        # Time 1
        inputs = {0: 1.0}
        out = functions.discrete_transfer_function(1.0, inputs, params)
        self.assertAlmostEqual(out[0], 1.0)
        
        # Time 2
        inputs = {0: 1.0}
        out = functions.discrete_transfer_function(2.0, inputs, params)
        self.assertAlmostEqual(out[0], 1.5)

if __name__ == '__main__':
    unittest.main()
