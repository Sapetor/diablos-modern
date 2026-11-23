"""UI scale preference (src/diablos/gui/ui_scale.py).

Before 2026-11 Preferences > UI Scale only set the app font to 10pt * factor.
Widgets styled from the fixed TYPE point sizes (menu bar, palette, Properties
panel, transport, toolbar) ignored it, so 150% looked patchy. The factor was
written to config/default_config.json, which in a source checkout is the
bundled file, so a developer's 125% had leaked into the shipped default. The
scale is now QT_SCALE_FACTOR, read from QSettings before QApplication exists.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from diablos.gui import ui_scale

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private settings INI and no inherited QT_SCALE_FACTOR."""
    ini = tmp_path / "settings.ini"
    monkeypatch.setenv("DIABLOS_SETTINGS_INI", str(ini))
    monkeypatch.delenv("QT_SCALE_FACTOR", raising=False)
    return ini


def _write(ini, value):
    from PyQt6.QtCore import QSettings

    s = QSettings(str(ini), QSettings.Format.IniFormat)
    s.setValue(ui_scale.SETTINGS_KEY, value)
    s.sync()


@pytest.mark.unit
class TestStoredScale:
    def test_default_is_100_percent(self, store):
        assert ui_scale.stored_ui_scale() == 1.0

    @pytest.mark.parametrize("factor", ui_scale.CHOICES)
    def test_round_trip(self, store, factor):
        assert ui_scale.store_ui_scale(factor)
        assert ui_scale.stored_ui_scale() == factor

    @pytest.mark.parametrize("bad", ["abc", "10", "0", "-1"])
    def test_invalid_stored_value_falls_back_to_100_percent(self, store, bad):
        _write(store, bad)
        assert ui_scale.stored_ui_scale() == 1.0

    def test_store_rejects_out_of_range(self, store):
        with pytest.raises(ValueError):
            ui_scale.store_ui_scale(12)

    def test_source_checkout_ignores_the_bundled_config(self, store):
        """In a checkout the "user" config is the bundled file: not a choice."""
        assert ui_scale._legacy_config_scale() is None
        bundled = json.load(open(os.path.join(REPO, "config", "default_config.json")))
        assert "scaling_factor" not in bundled.get("display", {})

    def test_packaged_build_keeps_a_previous_choice(self, store, tmp_path, monkeypatch):
        from diablos.core import app_paths

        user = tmp_path / "user"
        (user / "config").mkdir(parents=True)
        (user / "config" / "default_config.json").write_text(
            json.dumps({"display": {"scaling_factor": 1.5}})
        )
        monkeypatch.setattr(app_paths, "user_data_path", lambda rel: str(user / rel))
        assert ui_scale.stored_ui_scale() == 1.5
        # An explicit QSettings value wins over the old config key.
        ui_scale.store_ui_scale(1.25)
        assert ui_scale.stored_ui_scale() == 1.25


@pytest.mark.unit
class TestApply:
    def test_sets_qt_scale_factor(self, store):
        ui_scale.store_ui_scale(1.25)
        assert ui_scale.apply_ui_scale() == 1.25
        assert os.environ["QT_SCALE_FACTOR"] == "1.25"

    def test_100_percent_leaves_the_environment_alone(self, store):
        assert ui_scale.apply_ui_scale() == 1.0
        assert "QT_SCALE_FACTOR" not in os.environ

    def test_explicit_environment_wins(self, store, monkeypatch):
        ui_scale.store_ui_scale(1.5)
        monkeypatch.setenv("QT_SCALE_FACTOR", "2")
        assert ui_scale.apply_ui_scale() == 2.0
        assert os.environ["QT_SCALE_FACTOR"] == "2"


@pytest.mark.slow
@pytest.mark.parametrize("scale", [1.25, 1.5])
def test_startup_scales_the_whole_ui(store, scale):
    """End to end in a fresh process: the stored scale reaches Qt's device
    pixel ratio, and the app font is not scaled a second time."""
    _write(store, scale)
    script = textwrap.dedent(
        """
        import os
        import diablos.main as diablos_modern

        setup_application = diablos_modern.setup_application

        def check_application():
            assert "QT_AUTO_SCREEN_SCALE_FACTOR" not in os.environ
            app = setup_application()
            print(app.primaryScreen().devicePixelRatio(), app.font().pointSize())
            raise SystemExit(0)

        diablos_modern.setup_application = check_application
        raise SystemExit(diablos_modern.main())
        """
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    env.pop("QT_AUTO_SCREEN_SCALE_FACTOR", None)
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    ratio, points = out.stdout.split()[-2:]
    assert float(ratio) == scale
    assert int(points) == 10
