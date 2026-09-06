
================================================================================
DIABLOS BLOCK AUDIT REPORT
================================================================================

PROJECT: diablos-modern
DATE: 2026-02-11
SCOPE: Compare available blocks in palette vs. blocks used in example diagrams

================================================================================
CRITICAL FINDINGS
================================================================================

ISSUE #1: MISSING BLOCKS - Block Names Mismatch
-----------

The examples use block names that don't directly match the palette registration.
This is caused by a naming discrepancy in the block system:

  • Examples use "block_fn" field from .diablos files
  • Palette registers blocks by class name (with "Block" suffix removed)
  • Some blocks define a different "block_name" property for serialization

AFFECTED BLOCKS (2):

  1. SgProd (SignalProduct)
     Location: blocks/sigproduct.py
     Class: SigProductBlock
     Used in: 2 files
       - gradient_descent_verification.diablos
       - optimization_basic_demo.diablos
     Root Cause: Class name→SigProduct, but block_name→SgProd (serialization name)

  2. TranFn (Transfer Function)
     Location: blocks/transfer_function.py
     Class: TransferFunctionBlock
     Used in: 8 files
       - c01_tank_feedback.diablos
       - c02_string_instability.diablos
       - c02_vehicle_single_agent.diablos
       - c03_bode_frequency_response.diablos
       - c05_lqr_vs_open_loop.diablos
       - c05_mass_spring_state_space.diablos
       - c06_observer_estimation.diablos
       - c08_consensus_ring.diablos
     Root Cause: Class name→TransferFunction, but block_name→TranFn (serialization name)

ROOT CAUSE:
  The block_loader.py extracts blocks by class name but doesn't use the 
  block_name property. When diagrams are loaded, they use block_fn which 
  comes from the block_name property. This creates a mismatch.

SOLUTION REQUIRED:
  Update main_window.py or dialog.py block instantiation to:
  1. Use block.block_name instead of class.__name__ for matching
  2. Or update load_block_from_type() to use the block_name property


================================================================================
BLOCKS IN PALETTE (80 total)
================================================================================

Available for use (all working, registered):

 1. Abs                      (Math)
 2. Adam                      (Optimization Primitives)
 3. AdvectionEquation1D       (PDE)
 4. AdvectionEquation2D       (PDE)
 5. Assert                    (Sinks)
 6. BodeMag                   (Control)
 7. BodePhase                 (Control)
 8. Constant                  (Sources)
 9. Constraint                (Optimization)
10. CostFunction              (Optimization)
11. DataFit                   (Optimization)
12. Deadband                  (Control)
13. Delay                     (Control)
14. Demux                     (Routing)
15. Derivative                (Control)
16. DiffusionReaction1D       (PDE)
17. DiscreteStateSpace        (Control)
18. DiscreteTransferFunction  (Control)
19. Display                   (Sinks)
20. Exponential               (Math)
21. Export                    (Sinks)
22. External                  (Other)
23. FFT                       (Math)
24. FieldGradient             (Field Processing)
25. FieldIntegral             (Field Processing)
26. FieldLaplacian            (Field Processing)
27. FieldMax                  (Field Processing)
28. FieldProbe                (Field Processing)
29. FieldProbe2D              (Field Processing)
30. FieldScope                (Field Processing)
31. FieldScope2D              (Field Processing)
32. FieldSlice                (Field Processing)
33. FirstOrderHold            (Control)
34. From                      (Routing)
35. Gain                      (Math)
36. Goto                      (Routing)
37. HeatEquation1D            (PDE)
38. HeatEquation2D            (PDE)
39. Hysteresis                (Control)
40. Integrator                (Control)
41. LinearSystemSolver        (Optimization Primitives)
42. MathFunction              (Math)
43. Momentum                  (Optimization Primitives)
44. Mux                       (Routing)
45. Noise                     (Sources)
46. NumericalGradient         (Optimization Primitives)
47. Nyquist                   (Control)
48. ObjectiveFunction          (Optimization Primitives)
49. Optimizer                 (Optimization)
50. PID                       (Control)
51. PRBS                      (Sources)
52. Parameter                 (Optimization)
53. Product                   (Math)
54. Ramp                      (Sources)
55. RateLimiter               (Control)
56. RateTransition            (Control)
57. ResidualNorm              (Optimization Primitives)
58. RootFinder                (Optimization Primitives)
59. RootLocus                 (Control)
60. Saturation                (Control)
61. Scope                     (Sinks)
62. Selector                  (Routing)
63. SigProduct                (Math) ← HIDDEN (block_name="SgProd")
64. Sine                      (Sources)
65. StateSpace                (Control)
66. StateVariable             (Optimization Primitives)
67. Step                      (Sources)
68. Sum                       (Math)
69. Switch                    (Routing)
70. Terminator                (Sinks)
71. TransferFunction          (Control) ← HIDDEN (block_name="TranFn")
72. TransportDelay            (Control)
73. VectorGain                (Optimization Primitives)
74. VectorPerturb             (Optimization Primitives)
75. VectorSum                 (Optimization Primitives)
76. WaveEquation1D            (PDE)
77. WaveEquation2D            (PDE)
78. WaveGenerator             (Sources)
79. XYGraph                   (Sinks)
80. ZeroOrderHold             (Control)


================================================================================
BLOCKS USED IN EXAMPLES (45 total)
================================================================================

All example blocks are available in palette (when accounting for name mapping):

 1. Adam                      (used in 2 files)
 2. AdvectionEquation1D       (used in 3 files)
 3. Constant                  (used in 41 files)
 4. Constraint                (used in 1 file)
 5. CostFunction              (used in 3 files)
 6. DiffusionReaction1D       (used in 2 files)
 7. Display                   (used in 15 files)
 8. FieldIntegral             (used in 1 file)
 9. FieldMax                  (used in 1 file)
10. FieldProbe                (used in 20 files)
11. FieldProbe2D              (used in 3 files)
12. FieldScope                (used in 12 files)
13. FieldScope2D              (used in 2 files)
14. FirstOrderHold            (used in 1 file)
15. Gain                      (used in 16 files)
16. HeatEquation1D            (used in 4 files)
17. HeatEquation2D            (used in 2 files)
18. Integrator                (used in 1 file)
19. LinearSystemSolver        (used in 2 files)
20. MathFunction              (used in 23 files)
21. Momentum                  (used in 2 files)
22. Mux                       (used in 2 files)
23. NumericalGradient         (used in 8 files)
24. ObjectiveFunction         (used in 30 files)
25. Optimizer                 (used in 3 files)
26. PID                       (used in 2 files)
27. Parameter                 (used in 6 files)
28. Product                   (used in 1 file)
29. Ramp                      (used in 6 files)
30. RateTransition            (used in 1 file)
31. ResidualNorm              (used in 6 files)
32. RootFinder                (used in 2 files)
33. Scope                     (used in 63 files)
34. SgProd ⚠️ MISSING MAPPING (used in 2 files) → maps to SigProduct
35. Sine                      (used in 2 files)
36. StateSpace                (used in 9 files)
37. StateVariable             (used in 16 files)
38. Step                      (used in 9 files)
39. Sum                       (used in 18 files)
40. TranFn ⚠️ MISSING MAPPING (used in 8 files) → maps to TransferFunction
41. VectorGain                (used in 4 files)
42. VectorPerturb             (used in 16 files)
43. VectorSum                 (used in 11 files)
44. WaveEquation1D            (used in 3 files)
45. ZeroOrderHold             (used in 1 file)


================================================================================
BLOCKS IN PALETTE BUT NOT USED IN EXAMPLES (37 total)
================================================================================

These blocks are available but have no examples. Good candidates for documentation:

 1. Abs                      - Absolute value
 2. AdvectionEquation2D      - 2D advection equation solver
 3. Assert                   - Signal assertion/validation
 4. BodeMag                  - Bode magnitude response
 5. BodePhase                - Bode phase response
 6. DataFit                  - Data fitting optimization
 7. Deadband                 - Dead zone
 8. Delay                    - Time delay
 9. Demux                    - Signal demultiplexer
10. Derivative               - Numerical differentiation
11. DiscreteStateSpace       - Discrete state-space system
12. DiscreteTransferFunction - Discrete transfer function
13. Exponential              - e^x function
14. Export                   - Export data to file
15. External                 - External function/library interface
16. FFT                      - Fast Fourier Transform
17. FieldGradient            - Field spatial gradient
18. FieldLaplacian           - Field Laplacian operator
19. FieldSlice               - Field slicing/extraction
20. From                     - Signal routing (source)
21. Goto                     - Signal routing (sink)
22. Hysteresis               - Hysteresis nonlinearity
23. Noise                    - Random noise generator
24. Nyquist                  - Nyquist frequency response
25. PRBS                     - Pseudo-random binary signal
26. RateLimiter              - Rate limiting
27. RootLocus                - Root locus plot
28. Saturation               - Saturation nonlinearity
29. Selector                 - Signal selector/multiplexer
30. Switch                   - Conditional switch
31. Terminator               - Signal terminator (no output)
32. TransportDelay           - Transport/dead time
33. WaveEquation2D           - 2D wave equation solver
34. WaveGenerator            - Arbitrary waveform generator
35. XYGraph                  - XY plane plot
36. (SigProduct)             - Already counted above
37. (TransferFunction)       - Already counted above


================================================================================
RECOMMENDATIONS
================================================================================

URGENT (Block Mapping Bug):
  □ Fix blocks/base_block.py or main_window.py to use block_name property
    instead of class name for diagram loading
  □ Create mapping between block_name and class name in registry
  □ Add test to verify all loaded examples can be instantiated

HIGH PRIORITY (Example Coverage):
  □ Add example for TransportDelay (frequently used in control)
  □ Add example for DiscreteTransferFunction (control education)
  □ Add example for Switch block (conditional logic)
  □ Add example for FieldGradient (PDE analysis)

MEDIUM PRIORITY (Documentation):
  □ Document unused blocks in wiki with use cases
  □ Add tutorials for Bode/Nyquist/RootLocus analysis blocks
  □ Create control system design example collection

LOW PRIORITY (Cleanup):
  □ Consider aliasing SigProduct → Product for consistency
  □ Review if 80 blocks is appropriate for education (might be too many)

================================================================================
