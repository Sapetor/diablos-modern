import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class RunHistoryService:
    """
    Manages the history of simulation runs, including storage, persistence, and limits.
    Extracted from DSim.
    """
    def __init__(self, limit=5, persist_path="saves/run_history.json"):
        self.history = []
        self.limit = limit
        self.persist_enabled = False
        self.persist_path = Path(persist_path)
        
    def load_history(self):
        """Load persisted run history if enabled."""
        try:
            if not self.persist_path.exists():
                return
            with open(self.persist_path, "r") as f:
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
                    traces.append({
                        "name": tr.get("name", ""),
                        "y": np.array(tr.get("y", []), dtype=float),
                        "step": bool(tr.get("step", False))
                    })
                loaded.append({
                    "name": run.get("name", "Run"),
                    "timeline": timeline,
                    "traces": traces,
                    "sim_dt": run.get("sim_dt"),
                    "sim_time": run.get("sim_time"),
                    "pinned": bool(run.get("pinned", False))
                })
            
            self.history = loaded[-self.limit:] if loaded else []
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
                serializable_runs.append({
                    "name": run.get("name", ""),
                    "timeline": run.get("timeline", np.array([])).tolist() if hasattr(run.get("timeline"), 'tolist') else run.get("timeline"),
                    "traces": [
                        {
                            "name": tr.get("name", ""),
                            "y": tr.get("y").tolist() if hasattr(tr.get("y"), 'tolist') else tr.get("y"),
                            "step": bool(tr.get("step", False))
                        }
                        for tr in run.get("traces", [])
                    ],
                    "sim_dt": run.get("sim_dt"),
                    "sim_time": run.get("sim_time"),
                    "pinned": bool(run.get("pinned", False))
                })
            
            payload = {
                "persist": self.persist_enabled,
                "runs": serializable_runs
            }
            with open(self.persist_path, "w") as f:
                json.dump(payload, f)
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
            except Exception:
                pass
        else:
            self.save_history()

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
            
        run_entry = {
            "name": run_name,
            "timeline": timeline,
            "traces": traces,
            "sim_dt": sim_dt,
            "sim_time": sim_time,
            "pinned": False,
        }
        
        self.history.append(run_entry)
        
        # Enforce limit on unpinned runs while keeping pinned entries
        unpinned_count = sum(1 for r in self.history if not r.get("pinned", False))
        if unpinned_count > self.limit:
            drop = unpinned_count - self.limit
            new_history = []
            for r in self.history:
                if not r.get("pinned", False) and drop > 0:
                    drop -= 1
                    continue
                new_history.append(r)
            self.history = new_history

        # Persist if enabled
        self.save_history()
