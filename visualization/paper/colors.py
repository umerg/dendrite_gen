"""Global semantic color scheme for the paper figures."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


# Method identity colors. Use the base color for primary curves and marks, and
# the light/dark variants when a second tone of the same method is needed.
REFERENCE_COLOR = "#4C78A8"
REFERENCE_LIGHT_COLOR = "#AFCBE3"
REFERENCE_DARK_COLOR = "#2D5B86"

SEMLAFLOW_COLOR = "#F58518"
SEMLAFLOW_LIGHT_COLOR = "#FBC48B"
SEMLAFLOW_DARK_COLOR = "#B95D08"

OURS_COLOR = "#54A86B"
OURS_LIGHT_COLOR = "#B4E2BD"
OURS_DARK_COLOR = "#2B7042"


# Tree drawings intentionally use one neutral-to-method-independent green
# family in every qualitative panel. Method identity is carried by labels and
# statistical marks, not by recoloring the biological object itself.
TREE_COLOR = "#54A86B"
TREE_DEPTH_BACK_COLOR = "#0B3D24"
TREE_DEPTH_ROOT_COLOR = TREE_COLOR
TREE_DEPTH_FRONT_COLOR = "#B4E2BD"
TREE_DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "paper_tree_depth",
    [
        (0.0, TREE_DEPTH_BACK_COLOR),
        (0.5, TREE_DEPTH_ROOT_COLOR),
        (1.0, TREE_DEPTH_FRONT_COLOR),
    ],
)


# Supporting paper colors.
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#374151"
NEUTRAL_COLOR = "#4B5563"
LIGHT_GRID_COLOR = "#D1D5DB"
DIAGONAL_COLOR = "#9CA3AF"
GUIDE_COLOR = "#CBD5E1"
GUIDE_ACCENT_COLOR = "#DC2626"


METHOD_COLORS = {
    "reference": REFERENCE_COLOR,
    "semlaflow": SEMLAFLOW_COLOR,
    "ours": OURS_COLOR,
}
METHOD_LIGHT_COLORS = {
    "reference": REFERENCE_LIGHT_COLOR,
    "semlaflow": SEMLAFLOW_LIGHT_COLOR,
    "ours": OURS_LIGHT_COLOR,
}
METHOD_DARK_COLORS = {
    "reference": REFERENCE_DARK_COLOR,
    "semlaflow": SEMLAFLOW_DARK_COLOR,
    "ours": OURS_DARK_COLOR,
}

PAPER_COLORS = {
    **METHOD_COLORS,
    "reference-light": REFERENCE_LIGHT_COLOR,
    "reference-dark": REFERENCE_DARK_COLOR,
    "semlaflow-light": SEMLAFLOW_LIGHT_COLOR,
    "semlaflow-dark": SEMLAFLOW_DARK_COLOR,
    "ours-light": OURS_LIGHT_COLOR,
    "ours-dark": OURS_DARK_COLOR,
    "tree": TREE_COLOR,
    "tree-back": TREE_DEPTH_BACK_COLOR,
    "tree-root": TREE_DEPTH_ROOT_COLOR,
    "tree-middle": TREE_DEPTH_ROOT_COLOR,
    "tree-front": TREE_DEPTH_FRONT_COLOR,
    "background": BACKGROUND_COLOR,
    "text": TEXT_COLOR,
    "neutral": NEUTRAL_COLOR,
    "grid": LIGHT_GRID_COLOR,
    "guide": GUIDE_COLOR,
    "guide-accent": GUIDE_ACCENT_COLOR,
}


def resolve_paper_color(value: str) -> str:
    """Resolve a semantic paper color name, leaving literal colors untouched."""
    key = value.strip().lower().replace("_", "-")
    return PAPER_COLORS.get(key, value)


# Compatibility aliases for older paper scripts and notebooks.
BASELINE_COLOR = SEMLAFLOW_COLOR
DEPTH_GREEN_CMAP = TREE_DEPTH_CMAP
TREE_DEPTH_MIDDLE_COLOR = TREE_DEPTH_ROOT_COLOR
