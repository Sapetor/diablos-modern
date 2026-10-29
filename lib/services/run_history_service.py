import json
import logging
import numpy as np
from pathlib import Path

from lib.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


class RunHistoryService:
    """
    Manages the history of simulation runs, including storage, persistence, and limits.
    Extracted from DSim.
    """

    #: File name inside the writable ``saves/`` folder used when no explicit
    #: ``persist_path`` is given.
    DEFAULT_FILENAME = "run_history.json"

    def __init__(self, limit=5, persist_path=None):
        self.history = []
        self.limit = limit
        self.persist_enabled = False
        # ``None`` means "resolve lazily via lib.app_paths.user_saves_path", so
        # merely constructing a DSim does not create directories. The old
        # default was the relative literal "saves/run_history.json", so enabling
        # persistence in a packaged build (CWD read-only, often "/") raised
        # [Errno 30] Read-only file system: 'saves'.
        self._persist_path = Path(persist_path) if persist_path is not None else None

    @property
    def persist_path(self) -> Path:
        """The JSON file backing the history, always in a writable location."""
        if self._persist_path is not None:
            return self._persist_path
        from lib.app_paths import user_saves_dir

        # create=False: merely reading the path must not make directories;
        # save_history() mkdirs the parent right before it writes. Resolved on
        # every access so the location follows the user data dir.
        return Path(user_saves_dir(create=False)) / self.DEFAULT_FILENAME

    @persist_path.setter
    def persist_path(self, value):
        self._persist_path = Path(value) if value is not None else None

    def load_history(self):
        """Load persisted run history if enabled."""
        try:
            if not self.persist_path.exists():
                return
            with open(self.persist_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.persist_enabled = bool(payload.get("persist", False))
            if not self.persist_enabled:
                return

            runs = payload.get("runs", [])
            loaded = []
            for run in runs:
                timeline = np.array(run.get("timeline", []), dtype=float)
                traces = []
                for tr in run.get("traces", []):
                    traces.append(
                        {
                            "name": tr.get("name", ""),
                            "y": np.array(tr.get("y", []), dtype=float),
                            "step": bool(tr.get("step", False)),
                        }
                    )
                loaded.append(
                    {
                        "name": run.get("name", "Run"),
                        "timeline": timeline,
                        "traces": traces,
                        "sim_dt": run.get("sim_dt"),
                        "sim_time": run.get("sim_time"),
                        "pinned": bool(run.get("pinned", False)),
                    }
                )

            # The same rule as record_run: the limit counts unpinned runs only,
            # so a pinned baseline survives a restart.
            self.history = self._apply_limit(loaded)
            logger.info(f"Loaded {len(self.history)} runs from history.")
        except Exception as e:
            logger.warning(f"Could not load run history: {e}")

    def save_history(self):
        """Persist run history to disk if enabled."""
        try:
            if not self.persist_enabled:
                return

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            serializable_runs = []
            for run in self.history:
                # Convert numpy arrays to lists for JSON serialization
                serializable_runs.append(
                    {
                        "name": run.get("name", ""),
                        "timeline": run.get("timeline", np.array([])).tolist()
                        if hasattr(run.get("timeline"), "tolist")
                        else run.get("timeline"),
                        "traces": [
                            {
                                "name": tr.get("name", ""),
                                "y": tr["y"].tolist()
                                if hasattr(tr.get("y"), "tolist")
                                else (tr.get("y") or []),
                                "step": bool(tr.get("step", False)),
                            }
                            for tr in run.get("traces", [])
                        ],
                        "sim_dt": run.get("sim_dt"),
                        "sim_time": run.get("sim_time"),
                        "pinned": bool(run.get("pinned", False)),
                    }
                )

            payload = {"persist": self.persist_enabled, "runs": serializable_runs}
            # Atomic: an interrupted write must not truncate the saved history.
            atomic_write_json(str(self.persist_path), payload, indent=None)
            logger.debug("Run history saved.")
        except Exception as e:
            logger.warning(f"Could not save run history: {e}")

    def set_persist(self, enabled: bool):
        """Toggle persistence of waveform run history."""
        self.persist_enabled = bool(enabled)
        if not enabled:
            # Optionally remove file to avoid stale data
            try:
                if self.persist_path.exists():
                    self.persist_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove run history file: {e}")
        else:
            self.save_history()

    def _apply_limit(self, runs):
        """``runs`` with only the newest ``limit`` unpinned entries; pinned ones all stay."""
        drop = sum(1 for r in runs if not r.get("pinned", False)) - self.limit
        if drop <= 0:
            return list(runs)
        kept = []
        for r in runs:
            if not r.get("pinned", False) and drop > 0:
                drop -= 1
                continue
            kept.append(r)
        return kept

    def record_run(self, timeline, traces, sim_dt, sim_time, run_name=None):
        """
        Record a new simulation run.

        Args:
            timeline (np.ndarray): Time vector
            traces (list): List of trace dictionaries
            sim_dt (float): Simulation time step
            sim_time (float): Total simulation time
            run_name (str, optional): Name of the run. Defaults to timestamp.
        """
        import datetime

        if run_name is None:
            run_name = f"Run {datetime.datetime.now().strftime('%H:%M:%S')}"

        # Defensively copy caller-owned buffers so later runs that reuse/mutate
        # the same timeline/trace arrays cannot corrupt previously recorded history.
        timeline_copy = np.array(timeline)
        traces_copy = []
        for tr in traces or []:
            tr_copy = dict(tr)
            if "y" in tr_copy:
                tr_copy["y"] = np.array(tr_copy["y"])
            traces_copy.append(tr_copy)

        run_entry = {
            "name": run_name,
            "timeline": timeline_copy,
            "traces": traces_copy,
            "sim_dt": sim_dt,
            "sim_time": sim_time,
            "pinned": False,
        }

        self.history.append(run_entry)
        self.history = self._apply_limit(self.history)

        # Persist if enabled
        self.save_history()
