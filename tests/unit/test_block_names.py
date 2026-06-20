"""Unit tests for lib.engine.block_names.canonical_fn.

These pin the canonical dispatch names and prove that the extracted helper is
behavior-equivalent to the four inline normalization ladders it replaces.
"""
import pytest

from lib.engine.block_names import canonical_fn


def _legacy_full_ladder(block_fn):
    """Verbatim copy of the most complete inline ladder (system_compiler).

    Used to assert canonical_fn() is byte-for-byte equivalent to the code it
    replaced, so the refactor is provably behavior-preserving.
    """
    fn = block_fn.title() if block_fn else ''
    if fn == 'Statespace':
        fn = 'StateSpace'
    if fn in ('Transferfcn', 'Tranfn'):
        fn = 'TransferFcn'
    if block_fn == 'PID':
        fn = 'PID'
    if fn == 'Ratelimiter':
        fn = 'RateLimiter'
    if fn == 'Heatequation1d':
        fn = 'Heatequation1D'
    if fn == 'Waveequation1d':
        fn = 'Waveequation1D'
    if fn == 'Advectionequation1d':
        fn = 'Advectionequation1D'
    if fn == 'Diffusionreaction1d':
        fn = 'Diffusionreaction1D'
    return fn


# Every real block_fn value found in the example diagrams + the engine ladders.
REAL_BLOCK_FNS = [
    'Integrator', 'Gain', 'Sum', 'Step', 'Sine', 'Constant', 'Ramp',
    'Product', 'SgProd', 'Exponential', 'Saturation', 'Switch', 'Noise',
    'Abs', 'Selector', 'Hysteresis', 'Mux', 'Demux', 'Scope', 'Terminator',
    'StateSpace', 'Statespace', 'TranFn', 'Transferfcn', 'PID', 'RateLimiter',
    'MathFunction', 'WaveGenerator', 'Deadband', 'LogicalOperator',
    'HeatEquation1D', 'WaveEquation1D', 'AdvectionEquation1D',
    'DiffusionReaction1D', 'HeatEquation2D', 'WaveEquation2D',
    'AdvectionEquation2D', 'FieldProbe', 'FieldScope', 'FieldProbe2D',
    'FieldScope2D', 'FieldSlice', 'FieldIntegral', 'FieldMax',
]


@pytest.mark.unit
class TestCanonicalFn:
    @pytest.mark.parametrize("raw,expected", [
        ('Statespace', 'StateSpace'),
        ('StateSpace', 'StateSpace'),
        ('TranFn', 'TransferFcn'),
        ('Transferfcn', 'TransferFcn'),
        ('PID', 'PID'),
        ('RateLimiter', 'RateLimiter'),
        ('ratelimiter', 'RateLimiter'),
        ('Integrator', 'Integrator'),
        ('Gain', 'Gain'),
    ])
    def test_known_repairs(self, raw, expected):
        assert canonical_fn(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ('HeatEquation1D', 'Heatequation1D'),
        ('WaveEquation1D', 'Waveequation1D'),
        ('AdvectionEquation1D', 'Advectionequation1D'),
        ('DiffusionReaction1D', 'Diffusionreaction1D'),
        ('HeatEquation2D', 'Heatequation2D'),
        ('WaveEquation2D', 'Waveequation2D'),
        ('AdvectionEquation2D', 'Advectionequation2D'),
    ])
    def test_pde_names_match_dispatch_branches(self, raw, expected):
        # These are the exact strings the replay/compiler `elif fn == ...`
        # branches test against.
        assert canonical_fn(raw) == expected

    @pytest.mark.parametrize("falsy", ['', None])
    def test_falsy_input(self, falsy):
        assert canonical_fn(falsy) == ''

    @pytest.mark.parametrize("raw", REAL_BLOCK_FNS)
    def test_equivalent_to_legacy_ladder(self, raw):
        # Proof of behavior preservation against the inline ladder it replaced.
        assert canonical_fn(raw) == _legacy_full_ladder(raw)
