"""Semi-automatic perspective overlay tools for stair tread products."""

from .compositor import alpha_blend, compose_stair_overlay
from .geometry import TreadEdges, get_tread_edges, load_treads, save_treads, shrink_quad
from .geometry_model import MatDimensions, StairMatRenderConfig
from .geometry_v2 import TreadGeometry, TreadGeometryDocument, load_tread_geometry
from .renderer_v2 import compose_overlay_v2

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
    "TreadGeometry",
    "TreadGeometryDocument",
    "load_tread_geometry",
    "compose_overlay_v2",
]
