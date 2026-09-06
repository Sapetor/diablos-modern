"""Validation cases that currently fail, each pinned as a strict xfail.

Nothing belongs here that is a tolerance quibble: this module is for a wrong
answer, a crash, or a documented invariant the engine does not hold. A case is
marked ``xfail(strict=True)`` so the suite stays green while the defect stands
and turns red the moment it is fixed -- at which point the marker comes off, the
test moves to the validation module it belongs in, and the case joins the table
in ``docs/VALIDATION.md``.

Each test is written as the check that *should* pass, with the observed failure
described in its docstring, so the test doubles as the reproducer.

**There are currently no known defects.** The three this module was created for
have been fixed and their tests now live in ``test_closed_loop.py``
(interpreted PID convergence), ``test_convergence.py`` (the integration methods
that pair two input samples) and ``test_events.py`` (repeated location of a
state-dependent crossing). See the "Known defects" section of
``docs/VALIDATION.md`` for what each of them was.
"""

import pytest

pytestmark = pytest.mark.validation
