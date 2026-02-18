# b_type Property Fix Plan

## Issue
Saved block properties in `.diablos` files have different `b_type` values than what the palette generates.

**Saved values:**
- Step (source): b_type = 0
- TranFn (control): b_type = 1 (for proper transfer functions)
- Sum/Gain (math): b_type = 2
- Scope (sink): b_type = 3

**Current palette values:** All default to b_type = 2

## Root Cause
Block classes don't define `b_type` as a property, so `simulation_model.py` defaults all to 2.

## b_type Meaning
- **0**: Source blocks (no inputs, generate output)
- **1**: Memory/control blocks (outputs initial state before requiring inputs)
- **2**: Process/math blocks (normal processing)
- **3**: Sink blocks (consume output, no outputs)

## Fix Strategy

### 1. Add b_type to Source Blocks (b_type = 0)
- Constant, Noise, PRBS, Ramp, Sine, Step, WaveGenerator

### 2. Keep Memory/Control Blocks (b_type = 1)
- StateSpace, DiscreteStateSpace (memory blocks)
- Integrator (accumulates state)
- Delay, TransportDelay (hold history)
- StateVariable, Momentum, Adam (optimization primitives)
- Already correctly set: TranFn (conditional), DiscreteTranFn (conditional)

### 3. Add b_type to Sink Blocks (b_type = 3)
- Scope, Display, Term (Terminator), Assert, XYGraph, FFT
- Export (if outputs data)
- FieldScope, FieldScope2D (field sinks)

### 4. Math/Control Blocks remain b_type = 2
- Sum, Gain, Product, MathFunction, Saturation, etc.

## Implementation Order
1. Source blocks (7 blocks)
2. Sink blocks (8+ blocks)
3. Verify palette loads correctly
4. Test with compare_block_properties.py
5. Run example simulation to confirm no regressions
