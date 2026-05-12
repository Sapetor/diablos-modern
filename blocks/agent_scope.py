"""AgentScope: animated 2D scatter visualizer for multi-agent systems.

Captures a flat position vector [x1, y1, x2, y2, ..., xN, yN] at every time
step and stores the time series for post-simulation playback. On simulation
end, ScopePlotter opens a matplotlib window with the agents as a scatter
plot, with optional trails and a time slider + Export button (GIF / MP4).
"""

import logging
import numpy as np

from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class AgentScopeBlock(BaseBlock):
    @property
    def block_name(self):
        return "AgentScope"

    @property
    def category(self):
        return "Sinks"

    @property
    def b_type(self):
        return 3

    @property
    def color(self):
        return "red"

    @property
    def doc(self):
        return (
            "Multi-agent scope: animated 2D scatter of N agents over time."
            "\n\nInput: flat positions vector of length 2*N, layout [x1, y1, x2, y2, ...]."
            "\n\nParameters:"
            "\n- n_agents: number of agents (input length must equal 2*n_agents)."
            "\n- show_trails: whether to draw a fading line behind each agent."
            "\n- trail_length: max samples in each trail (0 = full history)."
            "\n- title: plot title."
            "\n\nOn simulation end opens a window with a time slider and an"
            " Export button (GIF / MP4)."
        )

    @property
    def params(self):
        return {
            "n_agents": {"type": "int", "default": 4, "doc": "Number of agents"},
            "show_trails": {"type": "bool", "default": True, "doc": "Draw trails"},
            "trail_length": {"type": "int", "default": 0,
                             "doc": "Max trail samples (0 = full history)"},
            "title": {"type": "string", "default": "Agent trajectories",
                      "doc": "Plot title"},
            "_init_start_": {"type": "bool", "default": True,
                             "doc": "Internal init flag"},
        }

    @property
    def inputs(self):
        return [{"name": "positions", "type": "array",
                 "doc": "Flat positions vector [x1, y1, x2, y2, ...]"}]

    @property
    def outputs(self):
        return []

    @property
    def requires_outputs(self):
        return False

    def draw_icon(self, block_rect):
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.addRect(0.15, 0.15, 0.7, 0.7)
        for cx, cy in [(0.3, 0.35), (0.55, 0.6), (0.7, 0.3), (0.4, 0.7)]:
            path.addEllipse(cx - 0.05, cy - 0.05, 0.1, 0.1)
        return path

    def execute(self, time, inputs, params, **kwargs):
        if params.get("_init_start_", True):
            params["_pos_history_"] = []
            params["_time_history_"] = []
            params["_init_start_"] = False

        u = np.atleast_1d(np.asarray(inputs.get(0, 0), dtype=float)).flatten()
        params["_pos_history_"].append(u.copy())
        params["_time_history_"].append(float(time))
        return {'E': False}
