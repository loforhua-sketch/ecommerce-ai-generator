"""Semi-automatic perspective overlay tools for stair tread products."""

from .compositor import alpha_blend, compose_stair_overlay
from .geometry import TreadEdges, get_tread_edges, load_treads, save_treads, shrink_quad
from .geometry_model import MatDimensions, StairMatRenderConfig

__all__ = [
    "alpha_blend",
    "compose_stair_overlay",
    "MatDimensions",
    "StairMatRenderConfig",
    "TreadEdges",
    "get_tread_edges",
    "load_treads",
    "save_treads",
    "shrink_quad",
]
