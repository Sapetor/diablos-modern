"""
Example: Discrete-Time Control System

This example demonstrates discrete-time blocks:
- Continuous signal input (sine wave)
- Zero-Order Hold (ZOH) for sampling
- Discrete Transfer Function (digital filter)
- Scope for comparison

Illustrates sampled-data systems.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.diagram_builder import DiagramBuilder


def create_discrete_example():
    """Create a discrete-time control example."""
    builder = DiagramBuilder(sim_time=5.0, sim_dt=0.001)  # Fine dt for continuous
    
    # Input: Sine wave (continuous)
    builder.add_block('Sine', 50, 150, name='sine', params={
        'amplitude': 1.0,
        'frequency': 2.0,
        'phase': 0.0
    })
    
    # Zero-Order Hold (samples at slower rate)
    builder.add_block('ZeroOrderHold', 200, 200, name='zoh', params={
        'sampling_time': 0.1  # 10 Hz sampling
    })
    
    # Discrete Transfer Function (low-pass filter)
    # H(z) = 0.5 / (z - 0.5) - simple first-order
    builder.add_block('DiscreteTF', 350, 200, name='dtf', params={
        'numerator': [0.5],
        'denominator': [1.0, -0.5]
    })
    
    # Scope for original signal
    builder.add_block('Scope', 400, 100, name='scope_orig', params={
        'title': 'Original Signal',
        'labels': 'continuous'
    })
    
    # Scope for ZOH output
    builder.add_block('Scope', 400, 200, name='scope_zoh', params={
        'title': 'ZOH Output',
        'labels': 'sampled'
    })
    
    # Scope for filtered output
    builder.add_block('Scope', 500, 300, name='scope_filt', params={
        'title': 'Filtered Output',
        'labels': 'filtered'
    })
    
    # Connections
    builder.connect('sine', 0, 'zoh', 0)
    builder.connect('sine', 0, 'scope_orig', 0)
    builder.connect('zoh', 0, 'dtf', 0)
    builder.connect('zoh', 0, 'scope_zoh', 0)
    builder.connect('dtf', 0, 'scope_filt', 0)
    
    # Save the diagram
    output_path = os.path.join(os.path.dirname(__file__), 'discrete_controller.json')
    builder.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_discrete_example()
