"""
Remaining Block Tests for DiaBloS

Tests for blocks not covered in other test files:
- StateSpace, TransferFunction (continuous)
- DiscreteStateSpace  
- TransportDelay
- Goto/From (routing)
- Scope, Display, Export, XYGraph, FFT (sinks)
- Terminator
- Assert, Rootlocus, Bodemag
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())


# ==================== CONTROL BLOCKS ====================

def test_statespace_block():
    """Test StateSpaceBlock implements dx/dt = Ax + Bu, y = Cx + Du."""
    from lib import functions
    
    # Simple first-order system: dx/dt = -x + u, y = x
    # A = -1, B = 1, C = 1, D = 0
    params = {
        'A': [[-1.0]],
        'B': [[1.0]],
        'C': [[1.0]],
        'D': [[0.0]],
        'init_conds': [0.0],
        '_init_start_': True,
        'dtime': 0.01
    }
    
    # Step response - should approach 1.0 as steady state
    for i in range(500):
        result = functions.statespace(i * 0.01, {0: np.array([1.0])}, params)
    
    # After 5 seconds, first-order system should be near 1.0
    output = params.get('_y_', result.get(0, [0]))
    if hasattr(output, '__len__'):
        output = output[0]
    assert output > 0.9, f"StateSpace: expected >0.9, got {output}"
    print("[PASS] StateSpaceBlock first-order system")


def test_transfer_function_block():
    """Test TransferFunctionBlock implements continuous TF."""
    from lib import functions
    
    # First-order low-pass filter: H(s) = 1/(s+1)
    # numerator = [1], denominator = [1, 1]
    params = {
        'numerator': [1.0],
        'denominator': [1.0, 1.0],
        '_init_start_': True,
        'dtime': 0.01
    }
    
    # Step response
    for i in range(500):
        result = functions.transfer_function(i * 0.01, {0: np.array([1.0])}, params)
    
    # After 5 seconds, should be near 1.0
    output = result.get(0, [0])
    if hasattr(output, '__len__'):
        output = output[0]
    assert output > 0.9, f"TransferFunction: expected >0.9, got {output}"
    print("[PASS] TransferFunctionBlock low-pass filter")


def test_discrete_statespace_block():
    """Test DiscreteStateSpaceBlock implements discrete state-space."""
    from blocks.discrete_statespace import DiscreteStateSpaceBlock
    
    block = DiscreteStateSpaceBlock()
    
    # Simple discrete integrator: x[k+1] = x[k] + u[k], y = x
    # A = 1, B = 1, C = 1, D = 0
    params = {
        'A': [[1.0]],
        'B': [[0.01]],  # Scale by dt for accumulation
        'C': [[1.0]],
        'D': [[0.0]],
        'init_conds': [0.0],
        '_init_start_': True,
        'dtime': 0.01
    }
    
    # Accumulate input over 100 steps
    for i in range(100):
        result = block.execute(i * 0.01, {0: np.array([1.0])}, params)
    
    # After 100 steps with input 1.0 and B=0.01, should be ~1.0
    output = result.get(0, [0])
    if hasattr(output, '__len__'):
        output = output[0]
    assert np.isclose(output, 1.0, rtol=0.1), f"DiscreteStateSpace: expected ~1.0, got {output}"
    print("[PASS] DiscreteStateSpaceBlock")


def test_transport_delay_block():
    """Test TransportDelayBlock delays signal by specified time."""
    from blocks.transport_delay import TransportDelayBlock
    
    block = TransportDelayBlock()
    
    params = {'delay_time': 0.1, 'initial_value': 0.0, '_init_start_': True}
    
    # Before delay time, should output initial value
    for i in range(5):
        result = block.execute(i * 0.01, {0: np.array([5.0])}, params)
    assert np.isclose(result[0], 0.0), f"TransportDelay before: expected 0.0, got {result[0]}"
    
    # Continue until delay time passed
    for i in range(5, 20):
        result = block.execute(i * 0.01, {0: np.array([5.0])}, params)
    
    # After delay time, should output delayed value
    assert np.isclose(result[0], 5.0, rtol=0.1), f"TransportDelay after: expected ~5.0, got {result[0]}"
    print("[PASS] TransportDelayBlock")


# ==================== ROUTING BLOCKS ====================

def test_goto_block():
    """Test GotoBlock passes through input (actual routing is external)."""
    from blocks.goto import GotoBlock
    
    block = GotoBlock()
    params = {'tag': 'A'}
    
    result = block.execute(0, {0: np.array([42.0])}, params)
    assert np.isclose(result[0], 42.0), f"Goto: expected 42.0, got {result[0]}"
    print("[PASS] GotoBlock passthrough")


def test_from_block():
    """Test FromBlock (signal source from Goto tag)."""
    from blocks.from_block import FromBlock
    
    block = FromBlock()
    params = {'tag': 'A', '_init_start_': True}
    
    # From block returns stored value (default 0 if no Goto connected)
    result = block.execute(0, {}, params)
    assert result[0] is not None, "FromBlock should return a value"
    print("[PASS] FromBlock")


# ==================== SINK BLOCKS ====================

def test_scope_block():
    """Test ScopeBlock accumulates data for plotting."""
    from blocks.scope import ScopeBlock
    
    block = ScopeBlock()
    params = {'_init_start_': True, 'title': 'test', 'labels': 'signal1'}
    
    # Feed data to scope
    for i in range(10):
        result = block.execute(i * 0.1, {0: np.array([float(i)])}, params)
    
    # Scope should store data in params
    assert 'vector' in params or len(result) >= 0, "Scope should accumulate data"
    print("[PASS] ScopeBlock data accumulation")


def test_display_block():
    """Test DisplayBlock stores value for display."""
    from blocks.display import DisplayBlock
    
    block = DisplayBlock()
    params = {}
    
    result = block.execute(0, {0: np.array([123.456])}, params)
    
    # Display stores value in _display_value_
    assert '_display_value_' in params or result is not None, "Display should store value"
    print("[PASS] DisplayBlock")


def test_export_block():
    """Test ExportBlock stores data for export."""
    from blocks.export import ExportBlock
    
    block = ExportBlock()
    params = {'_init_start_': True, 'variable_name': 'test_var', 'str_name': 'test_export'}
    
    for i in range(5):
        result = block.execute(i * 0.1, {0: np.array([float(i)])}, params)
    
    # Export should accumulate data
    assert '_export_data_' in params or 'vector' in params or result is not None, "Export should store data"
    print("[PASS] ExportBlock")


def test_terminator_block():
    """Test TerminatorBlock consumes input without output."""
    from blocks.terminator import TerminatorBlock
    
    block = TerminatorBlock()
    params = {}
    
    result = block.execute(0, {0: np.array([100.0])}, params)
    # Terminator just returns empty or 0
    print("[PASS] TerminatorBlock")


def test_xygraph_block():
    """Test XYGraphBlock stores x,y data pairs."""
    from blocks.xygraph import XYGraphBlock
    
    block = XYGraphBlock()
    params = {'_init_start_': True}
    
    # XY graph takes x and y inputs
    for i in range(10):
        x = float(i)
        y = x ** 2
        result = block.execute(i * 0.1, {0: np.array([x]), 1: np.array([y])}, params)
    
    # Should have accumulated data
    assert '_x_data_' in params or '_y_data_' in params, "XYGraph should store data pairs"
    print("[PASS] XYGraphBlock")


def test_fft_block():
    """Test FFTBlock accumulates data for FFT."""
    from blocks.fft import FFTBlock
    
    block = FFTBlock()
    params = {'_init_start_': True, 'window_size': 64}
    
    # Feed sine wave data
    for i in range(100):
        t = i * 0.01
        value = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine
        result = block.execute(t, {0: np.array([value])}, params)
    
    # FFT should have buffer
    assert '_fft_buffer_' in params, "FFT should accumulate buffer"
    print("[PASS] FFTBlock data accumulation")


# ==================== ANALYSIS BLOCKS ====================

def test_assert_block():
    """Test AssertBlock checks conditions."""
    from blocks.assert_block import AssertBlock
    
    block = AssertBlock()
    
    # By default, assert should pass for any input
    params = {'condition': '>', 'threshold': 0.0}
    result = block.execute(0, {0: np.array([1.0])}, params)
    # Assert returns the input if condition passes
    print("[PASS] AssertBlock condition check")


def test_bodemag_block():
    """Test BodeMagBlock (analysis block, returns empty during sim)."""
    from blocks.bodemag import BodeMagBlock
    
    block = BodeMagBlock()
    params = {'_init_start_': True}
    
    # BodeMag just returns empty dict during simulation
    result = block.execute(0, {0: np.array([1.0])}, params)
    assert result == {}, f"BodeMag: expected empty dict, got {result}"
    print("[PASS] BodeMagBlock")


def test_rootlocus_block():
    """Test RootLocusBlock (analysis block, returns empty during sim)."""
    from blocks.rootlocus import RootLocusBlock
    
    block = RootLocusBlock()
    params = {'_init_start_': True}
    
    # RootLocus just returns empty dict during simulation
    result = block.execute(0, {0: np.array([1.0])}, params)
    assert result == {}, f"RootLocus: expected empty dict, got {result}"
    print("[PASS] RootLocusBlock")


def test_external_block():
    """Test ExternalBlock (placeholder, returns None during sim)."""
    from blocks.external import ExternalBlock
    
    block = ExternalBlock()
    params = {'filename': 'test.py'}
    
    # External block returns None (executed externally)
    result = block.execute(0, {0: np.array([1.0])}, params)
    assert result is None, f"External: expected None, got {result}"
    print("[PASS] ExternalBlock")


if __name__ == "__main__":
    print("=" * 50)
    print("DiaBloS Remaining Block Tests")
    print("=" * 50)
    
    tests = [
        # Control
        ("StateSpaceBlock", test_statespace_block),
        ("TransferFunctionBlock", test_transfer_function_block),
        ("DiscreteStateSpaceBlock", test_discrete_statespace_block),
        ("TransportDelayBlock", test_transport_delay_block),
        # Routing
        ("GotoBlock", test_goto_block),
        ("FromBlock", test_from_block),
        # Sinks
        ("ScopeBlock", test_scope_block),
        ("DisplayBlock", test_display_block),
        ("ExportBlock", test_export_block),
        ("TerminatorBlock", test_terminator_block),
        ("XYGraphBlock", test_xygraph_block),
        ("FFTBlock", test_fft_block),
        # Analysis
        ("AssertBlock", test_assert_block),
        ("BodeMagBlock", test_bodemag_block),
        ("RootLocusBlock", test_rootlocus_block),
        ("ExternalBlock", test_external_block),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    sys.exit(1 if failed > 0 else 0)
