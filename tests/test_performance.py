"""
Performance Test Script for DiaBloS

Tests simulation performance with diagrams of various sizes.
Reports execution times and identifies potential bottlenecks.
"""

import sys
import os
import time
import cProfile
import pstats
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.diagram_builder import DiagramBuilder


def create_large_diagram(num_chains=10, chain_length=5):
    """Create a large diagram with multiple parallel signal chains.
    
    Args:
        num_chains: Number of parallel processing chains
        chain_length: Number of blocks in each chain
    
    Returns:
        DiagramBuilder with the created blocks
    """
    builder = DiagramBuilder(sim_time=10.0, sim_dt=0.01)
    
    y_spacing = 80
    x_spacing = 120
    
    for chain in range(num_chains):
        y_base = 50 + chain * y_spacing
        
        # Source block
        src_name = f"src_{chain}"
        builder.add_block('Sine', 50, y_base, name=src_name, params={
            'amplitude': 1.0,
            'frequency': 0.5 + chain * 0.1,
            'phase': 0.0
        })
        
        prev_block = src_name
        
        # Chain of processing blocks
        for i in range(chain_length):
            x = 50 + (i + 1) * x_spacing
            
            # Alternate block types
            if i % 3 == 0:
                block_name = f"gain_{chain}_{i}"
                builder.add_block('Gain', x, y_base, name=block_name, params={
                    'gain': 1.1
                })
            elif i % 3 == 1:
                block_name = f"int_{chain}_{i}"
                builder.add_block('Integrator', x, y_base, name=block_name, params={
                    'init_conds': 0.0
                })
            else:
                block_name = f"sat_{chain}_{i}"
                builder.add_block('Saturation', x, y_base, name=block_name, params={
                    'upper_limit': 10.0,
                    'lower_limit': -10.0
                })
            
            builder.connect(prev_block, 0, block_name, 0)
            prev_block = block_name
        
        # Scope at the end
        scope_name = f"scope_{chain}"
        builder.add_block('Scope', 50 + (chain_length + 1) * x_spacing, y_base, 
                         name=scope_name, params={
                             'title': f'Chain {chain}',
                             'labels': f'output_{chain}'
                         })
        builder.connect(prev_block, 0, scope_name, 0)
    
    return builder


def time_diagram_creation(num_chains, chain_length):
    """Time the creation of a diagram."""
    start = time.perf_counter()
    builder = create_large_diagram(num_chains, chain_length)
    elapsed = time.perf_counter() - start
    
    total_blocks = num_chains * (1 + chain_length + 1)  # src + chain + scope
    total_connections = num_chains * (chain_length + 1)
    
    return elapsed, total_blocks, total_connections, builder


def profile_simulation():
    """Profile simulation execution on a medium-sized diagram."""
    from lib.lib import DSim
    import json
    
    # Create a test diagram
    builder = create_large_diagram(num_chains=5, chain_length=4)
    
    # Save to temp file
    temp_path = 'perf_test_temp.json'
    builder.save(temp_path)
    
    # Load into DSim (headless check)
    try:
        # Profile the functions module instead of full simulation
        from lib import functions
        import numpy as np
        
        print("\nProfiling individual block functions:")
        
        # Profile integrator
        profiler = cProfile.Profile()
        profiler.enable()
        
        params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, 'dtime': 0.01}
        for i in range(1000):
            functions.integrator(i * 0.01, {0: np.array([1.0])}, params, dtime=0.01)
        
        profiler.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        print(s.getvalue())
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_benchmark():
    """Run comprehensive benchmarks on different diagram sizes."""
    print("=" * 60)
    print("DiaBloS Performance Benchmark")
    print("=" * 60)
    
    configs = [
        (2, 3),    # Small: ~10 blocks
        (5, 4),    # Medium: ~35 blocks
        (10, 5),   # Large: ~70 blocks
        (20, 5),   # XL: ~140 blocks
    ]
    
    print("\n1. Diagram Creation Performance:")
    print("-" * 50)
    print(f"{'Config':>15} {'Blocks':>8} {'Conns':>8} {'Time (ms)':>12}")
    print("-" * 50)
    
    for num_chains, chain_length in configs:
        elapsed, blocks, conns, builder = time_diagram_creation(num_chains, chain_length)
        config_str = f"{num_chains}x{chain_length}"
        print(f"{config_str:>15} {blocks:>8} {conns:>8} {elapsed*1000:>12.2f}")
    
    print("\n2. Block Execution Performance:")
    print("-" * 50)
    
    import numpy as np
    
    # Test individual block performance
    blocks_to_test = [
        ('Integrator', {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True, 'dtime': 0.01}),
        ('Gain', {'gain': 2.0}),
        ('Sum', {'sign': '++'}, 2),
        ('Saturation', {'min': -1.0, 'max': 1.0}),
    ]
    
    iterations = 10000
    
    for block_info in blocks_to_test:
        if len(block_info) == 2:
            block_type, params = block_info
            num_inputs = 1
        else:
            block_type, params, num_inputs = block_info
        
        # Import the block
        try:
            module = __import__(f'blocks.{block_type.lower()}', fromlist=[f'{block_type}Block'])
            block_class = getattr(module, f'{block_type}Block')
            block = block_class()
            
            inputs = {i: np.array([1.0]) for i in range(num_inputs)}
            
            start = time.perf_counter()
            for i in range(iterations):
                block.execute(i * 0.01, inputs, params.copy())
            elapsed = time.perf_counter() - start
            
            per_call = elapsed / iterations * 1e6  # microseconds
            print(f"  {block_type:>15}: {per_call:.2f} µs/call ({iterations} iterations)")
            
        except Exception as e:
            print(f"  {block_type:>15}: Error - {e}")
    
    print("\n3. Memory Overhead Check:")
    print("-" * 50)
    
    # Check diagram size in memory
    import json
    
    for num_chains, chain_length in [(5, 4), (20, 5)]:
        _, blocks, conns, builder = time_diagram_creation(num_chains, chain_length)
        
        # Serialize to JSON to check size
        temp_path = f'perf_temp_{num_chains}_{chain_length}.json'
        builder.save(temp_path)
        size = os.path.getsize(temp_path) / 1024  # KB
        os.remove(temp_path)
        
        print(f"  {num_chains}x{chain_length} ({blocks} blocks): {size:.1f} KB")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
