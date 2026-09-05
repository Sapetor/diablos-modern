"""Invariants of the shared block base classes introduced by the 2026-09-05
"duplicated block logic" cleanup (audit section 2).

Four families of blocks used to carry copy-pasted bodies:

* ``TranFn``/``StateSpace``/``DiscreteTranFn``/``DiscreteStateSpace`` repeated
  the ``_process_input -> _compute_output -> _update_state -> _format_output``
  sequence; it now lives in ``StateSpaceBaseBlock._step``.
* ``TransportDelay``/``VariableTransportDelay`` repeated the history-buffer
  interpolation; it now lives in ``DelayBufferBlock._interpolate``, with the
  one genuine difference between them selected by ``_HOLD_LAST_ON_TIE``.
* the five analysis markers repeated a purple, single-input, zero-output,
  inert-``execute`` declaration; it now lives in ``AnalysisMarkerBlock``.
* ``Goto``/``From`` repeated the tag parameter pair and the pass-through
  ``execute``; both now live in ``TagRoutingBlock``.

These tests pin what the shared code must keep doing, plus the property that
makes the sharing safe: a base class must stay *abstract*, because
``lib.block_loader`` registers every concrete ``BaseBlock`` subclass it can see
in a block module and would otherwise put the base in the palette.
"""

import inspect

import numpy as np
import pytest

from blocks.analysis_marker_base import AnalysisMarkerBlock
from blocks.delay_base import DelayBufferBlock
from blocks.discrete_statespace import DiscreteStateSpaceBlock
from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock
from blocks.statespace import StateSpaceBlock
from blocks.statespace_base import StateSpaceBaseBlock
from blocks.tag_base import TagRoutingBlock
from blocks.transfer_function import TransferFunctionBlock

SHARED_BASES = [
    StateSpaceBaseBlock,
    DelayBufferBlock,
    AnalysisMarkerBlock,
    TagRoutingBlock,
]


@pytest.mark.unit
@pytest.mark.parametrize("base", SHARED_BASES, ids=lambda c: c.__name__)
def test_shared_bases_stay_abstract(base):
    """``_collect_block_classes`` skips abstract classes and nothing else.

    A base that accidentally implements every abstract member becomes
    instantiable and is then imported into the palette (and into the frozen
    build) once per block module that imports it.
    """
    assert inspect.isabstract(base), f"{base.__name__} would register as a block"


@pytest.mark.unit
def test_shared_bases_are_not_registered_as_blocks():
    from lib.block_loader import load_blocks

    registered = {cls.__name__ for cls in load_blocks()}
    for base in SHARED_BASES:
        assert base.__name__ not in registered


@pytest.mark.unit
class TestStateSpaceSharedStep:
    """All four A/B/C/D blocks route their sample through the same ``_step``."""

    FAMILY = [
        TransferFunctionBlock,
        StateSpaceBlock,
        DiscreteTransferFunctionBlock,
        DiscreteStateSpaceBlock,
    ]

    @pytest.mark.parametrize("block_cls", FAMILY, ids=lambda c: c.__name__)
    def test_no_block_redefines_the_shared_step(self, block_cls):
        assert "_step" not in vars(block_cls)
        assert block_cls._step is StateSpaceBaseBlock._step

    def test_step_outputs_before_it_advances_the_state(self):
        """y must be formed from the state on entry, then x advanced once."""
        block = DiscreteStateSpaceBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        # x[k+1] = x[k] + u[k], y[k] = x[k]  -- a pure accumulator, so the
        # output lags the input by exactly one sample iff the order is right.
        params.update({"A": [[1.0]], "B": [[1.0]], "C": [[1.0]], "D": [[0.0]]})

        outs = [block.execute(t * 0.1, {0: np.array([1.0])}, params)[0] for t in range(4)]
        assert outs == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_output_only_reads_without_consuming_or_advancing(self):
        block = DiscreteStateSpaceBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        params.update({"A": [[1.0]], "B": [[1.0]], "C": [[1.0]], "D": [[0.0]]})

        block.execute(0.0, {0: np.array([2.0])}, params)  # x -> 2
        before = params["_x_"].copy()

        peek = block.execute(0.1, {0: np.array([5.0])}, params, output_only=True)

        assert peek[0] == pytest.approx(2.0)
        assert params["_x_"] == pytest.approx(before)

    @pytest.mark.parametrize(
        "block_cls", [TransferFunctionBlock, DiscreteTransferFunctionBlock], ids=["tranfn", "dtf"]
    )
    def test_siso_blocks_reject_a_vector_input_with_an_error_dict(self, block_cls):
        """Both used to reach ``y.item()`` and raise on a vector; the shared
        input check turns that into the same dimension-mismatch error dict."""
        block = block_cls()
        params = {k: v["default"] for k, v in block.params.items()}

        result = block.execute(0.0, {0: np.array([1.0, 2.0, 3.0])}, params, dtime=0.01)

        assert result.get("E") is True
        assert "dimension" in result["error"].lower()


@pytest.mark.unit
class TestDiscreteSampleGating:
    """The discrete pair share one sample-instant epsilon with every other
    rate-gated block (``blocks.input_helpers.SAMPLE_TIME_EPS``)."""

    @pytest.mark.parametrize(
        "block_cls", [DiscreteTransferFunctionBlock, DiscreteStateSpaceBlock], ids=["dtf", "dss"]
    )
    def test_output_is_held_between_sample_instants(self, block_cls):
        block = block_cls()
        params = {k: v["default"] for k, v in block.params.items()}
        params["sampling_time"] = 0.1

        first = block.execute(0.0, {0: np.array([1.0])}, params, dtime=0.01)[0]
        held = block.execute(0.05, {0: np.array([99.0])}, params, dtime=0.01)[0]

        assert held == pytest.approx(first)
        # ... and the schedule has moved past the instant just consumed.
        assert params["_next_sample_time_"] > 0.0


@pytest.mark.unit
class TestDelayBufferTieBreak:
    """``_HOLD_LAST_ON_TIE`` is the only behavioural difference between the two
    delay blocks; it decides what a request answers when every recorded
    timestamp is equal (the interpreter's four RK4 sub-steps at a fixed t)."""

    def test_fixed_delay_reports_the_initial_value_on_a_tie(self):
        from blocks.transport_delay import TransportDelayBlock

        block = TransportDelayBlock()
        assert block._HOLD_LAST_ON_TIE is False
        buf_t, buf_v = [1.0, 1.0], [np.array([7.0]), np.array([9.0])]

        out = block._interpolate(buf_t, buf_v, 1.0, initial_value=-1.0)

        assert out == pytest.approx([-1.0])

    def test_variable_delay_holds_the_newest_sample_on_a_tie(self):
        from blocks.variable_transport_delay import VariableTransportDelayBlock

        block = VariableTransportDelayBlock()
        assert block._HOLD_LAST_ON_TIE is True
        buf_t, buf_v = [1.0, 1.0], [np.array([7.0]), np.array([9.0])]

        out = block._interpolate(buf_t, buf_v, 1.0, initial_value=-1.0)

        assert out == pytest.approx([9.0])

    def test_both_interpolate_linearly_between_bracketing_samples(self):
        from blocks.transport_delay import TransportDelayBlock
        from blocks.variable_transport_delay import VariableTransportDelayBlock

        buf_t, buf_v = [0.0, 1.0], [np.array([0.0]), np.array([10.0])]
        for block in (TransportDelayBlock(), VariableTransportDelayBlock()):
            out = block._interpolate(buf_t, buf_v, 0.25, initial_value=-1.0)
            assert out == pytest.approx([2.5])


@pytest.mark.unit
class TestTagRoutingBase:
    """Goto and From are one connection: same tag params, opposite ports."""

    def test_the_pair_declare_identical_tag_parameters(self):
        from blocks.from_block import FromBlock
        from blocks.goto import GotoBlock

        assert GotoBlock().params == FromBlock().params
        assert set(GotoBlock().params) == {"tag", "signal_name"}

    def test_ports_are_mirrored(self):
        from blocks.from_block import FromBlock
        from blocks.goto import GotoBlock

        assert len(GotoBlock().inputs) == 1 and GotoBlock().outputs == []
        assert FromBlock().inputs == [] and len(FromBlock().outputs) == 1

    def test_both_forward_the_routed_signal_untouched(self):
        from blocks.from_block import FromBlock
        from blocks.goto import GotoBlock

        signal = np.array([1.0, 2.0, 3.0])
        for block in (GotoBlock(), FromBlock()):
            out = block.execute(0.0, {0: signal}, {"tag": "A"})
            assert out[0] is signal, "routing must not copy or normalize"

    def test_an_unrouted_tag_yields_the_bare_zero_default(self):
        from blocks.from_block import FromBlock

        assert FromBlock().execute(0.0, {}, {"tag": "A"}) == {0: 0}


@pytest.mark.unit
class TestAnalysisMarkerBase:
    MARKERS = ["bodemag", "bodephase", "nyquist", "rootlocus", "lqr"]

    @pytest.mark.parametrize("module_name", MARKERS)
    def test_every_marker_uses_the_shared_base(self, module_name):
        import importlib

        module = importlib.import_module(f"blocks.{module_name}")
        cls = next(
            obj
            for _, obj in vars(module).items()
            if inspect.isclass(obj)
            and issubclass(obj, AnalysisMarkerBlock)
            and not inspect.isabstract(obj)
        )
        assert cls.execute is AnalysisMarkerBlock.execute
        assert cls.draw_icon is AnalysisMarkerBlock.draw_icon
        assert cls().outputs == []
        assert cls().color == "purple"

    def test_the_base_refuses_to_draw_without_a_subclass_icon(self):
        class Bare(AnalysisMarkerBlock):
            @property
            def block_name(self):
                return "Bare"

        with pytest.raises(NotImplementedError):
            Bare().draw_icon(None)
