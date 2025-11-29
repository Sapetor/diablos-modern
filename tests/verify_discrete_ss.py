
import unittest
import numpy as np
from lib import functions

class TestDiscreteStateSpace(unittest.TestCase):
    def test_accumulator(self):
        # x[k+1] = x[k] + u[k]
        # y[k] = x[k]
        params = {
            'A': [[1.0]],
            'B': [[1.0]],
            'C': [[1.0]],
            'D': [[0.0]],
            'init_conds': [0.0],
            '_init_start_': True,
            '_name_': 'TestAccumulator'
        }
        
        # k=0, u=1.0
        inputs = {0: 1.0}
        out = functions.discrete_statespace(0.0, inputs, params)
        self.assertEqual(out[0], 0.0) # y[0] = x[0] = 0
        self.assertEqual(params['_x_'][0,0], 1.0) # x[1] = 0 + 1 = 1
        
        # k=1, u=1.0
        inputs = {0: 1.0}
        out = functions.discrete_statespace(1.0, inputs, params)
        self.assertEqual(out[0], 1.0) # y[1] = x[1] = 1
        self.assertEqual(params['_x_'][0,0], 2.0) # x[2] = 1 + 1 = 2

    def test_direct_feedthrough(self):
        # x[k+1] = 0.5*x[k] + u[k]
        # y[k] = x[k] + u[k]
        params = {
            'A': [[0.5]],
            'B': [[1.0]],
            'C': [[1.0]],
            'D': [[1.0]],
            'init_conds': [0.0],
            '_init_start_': True,
            '_name_': 'TestDirectFeedthrough'
        }
        
        # k=0, u=1.0
        inputs = {0: 1.0}
        out = functions.discrete_statespace(0.0, inputs, params)
        self.assertEqual(out[0], 1.0) # y[0] = 0 + 1 = 1
        self.assertEqual(params['_x_'][0,0], 1.0) # x[1] = 0 + 1 = 1
        
        # k=1, u=1.0
        inputs = {0: 1.0}
        out = functions.discrete_statespace(1.0, inputs, params)
        self.assertEqual(out[0], 2.0) # y[1] = 1 + 1 = 2
        self.assertEqual(params['_x_'][0,0], 1.5) # x[2] = 0.5*1 + 1 = 1.5

if __name__ == '__main__':
    unittest.main()
