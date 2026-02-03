"""
PDE Blocks Package - Distributed Parameter Model Blocks

This package provides blocks for solving Partial Differential Equations (PDEs)
using the Method of Lines (MOL) approach. Space is discretized while time
remains continuous, converting PDEs into large ODE systems that the existing
solver infrastructure can handle.

Supported equations (1D):
- HeatEquation1D: 1D heat/diffusion equation (∂T/∂t = α∇²T + q)
- WaveEquation1D: 1D wave equation (∂²u/∂t² = c²∇²u)
- AdvectionEquation1D: 1D advection equation (∂c/∂t + v∂c/∂x = 0)
- DiffusionReaction1D: 1D diffusion-reaction (∂c/∂t = D∇²c - kc)

Supported equations (2D):
- HeatEquation2D: 2D heat/diffusion equation (∂T/∂t = α∇²T + q)
- WaveEquation2D: 2D wave equation (∂²u/∂t² = c²∇²u)
- AdvectionEquation2D: 2D advection-diffusion (∂c/∂t = -v·∇c + D∇²c)

Boundary condition types:
- Dirichlet: Fixed value (T = T₀)
- Neumann: Fixed flux (∂T/∂x = q)
- Robin: Mixed/convective (h(T - T∞))
"""

from blocks.pde.heat_equation_1d import HeatEquation1DBlock
from blocks.pde.wave_equation_1d import WaveEquation1DBlock
from blocks.pde.advection_equation_1d import AdvectionEquation1DBlock
from blocks.pde.diffusion_reaction_1d import DiffusionReaction1DBlock
from blocks.pde.heat_equation_2d import HeatEquation2DBlock
from blocks.pde.wave_equation_2d import WaveEquation2DBlock
from blocks.pde.advection_equation_2d import AdvectionEquation2DBlock
from blocks.pde.field_processing import (
    FieldProbeBlock,
    FieldIntegralBlock,
    FieldMaxBlock,
    FieldScopeBlock,
    FieldGradientBlock,
    FieldLaplacianBlock,
)
from blocks.pde.field_processing_2d import (
    FieldProbe2DBlock,
    FieldScope2DBlock,
    FieldSliceBlock,
)

__all__ = [
    # 1D PDE blocks
    'HeatEquation1DBlock',
    'WaveEquation1DBlock',
    'AdvectionEquation1DBlock',
    'DiffusionReaction1DBlock',
    # 2D PDE blocks
    'HeatEquation2DBlock',
    'WaveEquation2DBlock',
    'AdvectionEquation2DBlock',
    # 1D Field processing
    'FieldProbeBlock',
    'FieldIntegralBlock',
    'FieldMaxBlock',
    'FieldScopeBlock',
    'FieldGradientBlock',
    'FieldLaplacianBlock',
    # 2D Field processing
    'FieldProbe2DBlock',
    'FieldScope2DBlock',
    'FieldSliceBlock',
]
