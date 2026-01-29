"""
Example: Nonlinear Control System

This example demonstrates nonlinear blocks:
- Sine wave input
- Saturation (limits amplitude)
- Hysteresis (on/off behavior with thresholds)
- Deadband (zero output near zero input)
- Rate Limiter (limits rate of change)
- Scope for comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.diagram_builder import DiagramBuilder


def create_nonlinear_example():
    """Create a nonlinear blocks demonstration."""
    builder = DiagramBuilder(sim_time=5.0, sim_dt=0.01)
    
    # Input: Sine wave
    builder.add_block('Sine', 50, 200, name='input', params={
        'amplitude': 2.0,
        'frequency': 0.5,
        'phase': 0.0
    })
    
    # Saturation block (-1 to +1)
    builder.add_block('Saturation', 200, 100, name='sat', params={
        'upper_limit': 1.0,
        'lower_limit': -1.0
    })
    
    # Hysteresis block 
    builder.add_block('Hysteresis', 200, 200, name='hyst', params={
        'upper': 0.5,
        'lower': -0.5,
        'high': 1.0,
        'low': -1.0
    })
    
    # Deadband block
    builder.add_block('Deadband', 200, 300, name='dead', params={
        'start': -0.5,
        'end': 0.5
    })
    
    # Rate Limiter
    builder.add_block('RateLimiter', 200, 400, name='rate', params={
        'rising_slew': 2.0,
        'falling_slew': -2.0
    })
    
    # Scopes for each output
    builder.add_block('Scope', 400, 100, name='scope_sat', params={
        'title': 'Saturation',
        'labels': 'saturated'
    })
    builder.add_block('Scope', 400, 200, name='scope_hyst', params={
        'title': 'Hysteresis',
        'labels': 'hysteresis'
    })
    builder.add_block('Scope', 400, 300, name='scope_dead', params={
        'title': 'Deadband',
        'labels': 'deadband'
    })
    builder.add_block('Scope', 400, 400, name='scope_rate', params={
        'title': 'Rate Limiter',
        'labels': 'rate_limited'
    })
    
    # Connections: connect(src_block, src_port, dst_block, dst_port)
    builder.connect('input', 0, 'sat', 0)
    builder.connect('input', 0, 'hyst', 0)
    builder.connect('input', 0, 'dead', 0)
    builder.connect('input', 0, 'rate', 0)
    builder.connect('sat', 0, 'scope_sat', 0)
    builder.connect('hyst', 0, 'scope_hyst', 0)
    builder.connect('dead', 0, 'scope_dead', 0)
    builder.connect('rate', 0, 'scope_rate', 0)
    
    # Save the diagram
    output_path = os.path.join(os.path.dirname(__file__), 'nonlinear_blocks.json')
    builder.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_nonlinear_example()
