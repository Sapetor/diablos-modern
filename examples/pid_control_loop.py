"""
Example: PID Control Loop with Plant

This example demonstrates a classic feedback control system:
- Reference step input
- PID controller
- First-order plant (transfer function)
- Scope for visualization

The closed-loop system regulates the plant output to track the reference.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.diagram_builder import DiagramBuilder


def create_pid_control_example():
    """Create a PID feedback control loop example."""
    builder = DiagramBuilder(sim_time=10.0, sim_dt=0.01)
    
    # Reference signal (step from 0 to 1 at t=1)
    builder.add_block('Step', 50, 200, name='ref', params={
        'step_time': 1.0,
        'initial_value': 0.0,
        'final_value': 1.0
    })
    
    # Error calculation (reference - feedback)
    builder.add_block('Sum', 180, 200, name='error', params={
        'sign': '+-'
    })
    
    # PID Controller
    builder.add_block('PID', 300, 200, name='pid', params={
        'Kp': 5.0,
        'Ki': 2.0,
        'Kd': 0.5,
        'N': 20.0
    })
    
    # Plant: First-order system G(s) = 1/(s+1)
    builder.add_block('TranFn', 450, 200, name='plant', params={
        'numerator': [1.0],
        'denominator': [1.0, 1.0]
    })
    
    # Scope to view output
    builder.add_block('Scope', 600, 200, name='scope', params={
        'title': 'PID Response',
        'labels': 'output'
    })
    
    # Connections: connect(src_block, src_port, dst_block, dst_port)
    builder.connect('ref', 0, 'error', 0)    # ref -> error (+)
    builder.connect('error', 0, 'pid', 0)    # error -> PID
    builder.connect('pid', 0, 'plant', 0)    # PID -> plant
    builder.connect('plant', 0, 'scope', 0)  # plant -> scope
    builder.connect('plant', 0, 'error', 1)  # feedback loop (plant -> error -)
    
    # Save the diagram
    output_path = os.path.join(os.path.dirname(__file__), 'pid_control_loop.json')
    builder.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_pid_control_example()
