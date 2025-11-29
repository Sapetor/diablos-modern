
import unittest
from lib import functions

class TestZeroOrderHold(unittest.TestCase):
    def test_zoh_logic(self):
        # Sampling time = 0.5
        params = {
            'sampling_time': 0.5,
            '_init_start_': True,
            '_name_': 'TestZOH'
        }
        
        # t=0.0: Should sample input 10.0
        inputs = {0: 10.0}
        out = functions.zero_order_hold(0.0, inputs, params)
        self.assertEqual(out[0], 10.0)
        self.assertEqual(params['_held_value_'], 10.0)
        self.assertAlmostEqual(params['_next_sample_time_'], 0.5)
        
        # t=0.2: Should hold 10.0 even if input changes to 20.0
        inputs = {0: 20.0}
        out = functions.zero_order_hold(0.2, inputs, params)
        self.assertEqual(out[0], 10.0)
        
        # t=0.5: Should sample input 30.0
        inputs = {0: 30.0}
        out = functions.zero_order_hold(0.5, inputs, params)
        self.assertEqual(out[0], 30.0)
        self.assertAlmostEqual(params['_next_sample_time_'], 1.0)
        
        # t=0.7: Should hold 30.0
        inputs = {0: 40.0}
        out = functions.zero_order_hold(0.7, inputs, params)
        self.assertEqual(out[0], 30.0)

if __name__ == '__main__':
    unittest.main()
