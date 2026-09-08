"""One diagram, both solver paths, the same Scope signal names.

Ensemble and sweep results are keyed by the names ``harvest_scope_signals``
produces, and a study can be re-run on either path (the compiled path falls
back to the interpreter for unsupported blocks). A single-input Scope used to
come back as ``y`` when compiled but as ``scope1`` when interpreted; the
user's ``labels`` entry decides on both now.
"""

import numpy as np
import pytest

from lib.analysis.resim import harvest_scope_signals
from tests.validation import _harness as H


def _diagram():
    b = H.build(0.5, 0.01)
    H.add(b, "Step", "src", {"value": 1.0, "delay": 0.0})
    H.add(b, "Sine", "osc", {})
    H.add(b, "Scope", "labelled", {"labels": "y"})
    H.add(b, "Scope", "unlabelled", {"labels": "default"})
    H.add(b, "Scope", "pair", {"labels": "a,b"}, in_ports=2)
    b.connect("src", 0, "labelled", 0)
    b.connect("src", 0, "unlabelled", 0)
    b.connect("src", 0, "pair", 0)
    b.connect("osc", 0, "pair", 1)
    return b


@pytest.mark.regression
def test_signal_names_match_across_paths(qapp, tmp_path):
    runs = {}
    for compiled in (False, True):
        result = H.run(_diagram(), tmp_path=tmp_path, compiled=compiled)
        assert result.compiled is compiled
        harvested = harvest_scope_signals(result.dsim)
        assert harvested is not None
        runs[compiled] = harvested["signals"]

    interp, comp = runs[False], runs[True]
    assert sorted(interp) == sorted(comp)
    # The user label, the ``<scope>-0`` default and per-channel labels all
    # survive, and nothing is keyed by the bare block name.
    default_name = next(k for k in comp if k.endswith("-0"))
    assert set(comp) == {"y", default_name, "a", "b"}
    for name in comp:
        n = min(len(interp[name]), len(comp[name]))
        assert n > 10
        np.testing.assert_allclose(interp[name][:n], comp[name][:n], atol=1e-6)
