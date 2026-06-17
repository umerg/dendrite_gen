"""Generated README text for visualization output folders."""

from __future__ import annotations

from pathlib import Path


VISUALIZATION_OUTPUT_README = """# Visualization Outputs

This folder contains generated plots for comparing reference trees and model samples. Some plot families below may be absent if the corresponding runner or option was not used.

## Conventions

In paired plots, "Reference" or "GT" means the ground-truth tree and "Ours" or "Pred" means the generated tree. Blue/burgundy usually distinguish these two sources; plots with a colorbar use color for the shown attribute value instead.

## Qualitative 2D Plots (`qualitative/`)

- `*_pair_{projection}.png`: side-by-side 2D projections of one reference tree and the corresponding generated tree. These are useful for a quick shape comparison, but they assume the two trees are meaningfully paired by index.
- `*_overlay_{projection}.png`: overlays the reference and generated trees in the same 2D coordinate system. This emphasizes local geometric mismatch when pairwise comparison is meaningful.
- `*_offset_{projection}.png`: shows reference and generated trees in one panel with a small offset between them. This keeps both silhouettes visible while still making the shapes easy to compare.
- `gallery2d_{projection}.png`: a compact gallery of several selected reference/generated examples. This is mainly for qualitative browsing of a small subset.
- `offset_gallery2d_{projection}.png`: a gallery version of the offset plot. It is useful when direct overlays would visually obscure one tree.

## Tree-Level Statistics (`tree_stats/`)

- `treelevel_hist.png`: compares distributions of scalar tree-level metrics such as height, span, path length, and branch angle. For unconditioned models, this is more meaningful than pairwise scatter because it compares populations.
- `treelevel_scatter.png`: plots paired reference-vs-generated values for scalar metrics. Interpret this only when generated sample `i` is intended to match reference tree `i`.

## Within-Tree Distribution Statistics (`distribution_stats/`)

- `distribution_hist_pooled.png`: compares pooled distributions of branch length, bifurcation angle, path distance, radial distance, and branch order across all selected trees. Large trees contribute more values, so this view emphasizes the population of branches/nodes.
- `distribution_hist_tree_average.png`: compares the same within-tree quantities after averaging each tree's histogram equally. This reduces the dominance of very large trees.
- `*_distribution_hist.png`: per-tree versions of the within-tree distribution plots. These are paired diagnostics for individual examples.

## TMD Figures (`tmd_figures/`)

- `*_tmd_grid.png`: persistence-diagram grids for individual reference/generated pairs and selected filtrations. These show the topological summaries used by the TMD diagnostics.
- `tmd_mean_pi_*.png`: mean persistence-image comparisons for reference and generated populations. These summarize average topological structure under a chosen filtration.
- `tmd_embedding_*.png`: 2D embeddings of persistence-image vectors for reference and generated trees. Nearby points have similar TMD summaries, though the axes themselves are embedding coordinates rather than directly interpretable measurements.
- `tmd_embedding_*_color_*.png`: the same TMD embedding colored by a tree-level scalar attribute. These help reveal whether visible clusters or gradients correspond to interpretable tree properties.
- `tmd_diagram_distance_*_by_gt_*.png`: compares TMD distance to a chosen ground-truth scalar attribute. These are paired diagnostics and should be interpreted only when GT/pred pairings are meaningful.
- `tmd_embedding_*_points.csv`: tabular embedding coordinates and metadata used to make the corresponding TMD embedding plots.

## Cylinder Trees (`cylinders*/`)

- `*_gt_cylinder_*.html` or `*.png`: 3D cylinder renderings of reference trees. Plotly `.html` files are interactive and can be rotated in the browser.
- `*_pred*_cylinder_*.html` or `*.png`: 3D cylinder renderings of generated trees. Thickness may come from stored radii or from visualization-only synthesized radii, depending on the command options.
- `*_cylinder_pair_*.png`: static side-by-side cylinder comparisons from the Matplotlib backend. Plotly pair mode writes separate GT and prediction HTML files instead.
- Folders containing `curved` use endpoint-preserving curved branch paths for display, and folders containing `leaves` add translucent low-poly canopy blobs. These are visualization aids, not measured tree geometry.

## Unconditional Diagnostics (`unconditional/`)

- `tree_feature_pca.png`: PCA of per-tree feature vectors for reference and generated populations. Each vector uses mean and standard deviation summaries of branch length, bifurcation angle, path distance, radial distance, and branch order, plus height, XY span, and bounding-box diagonal.
- `tree_feature_pca_by_feature/pca_color_*.png`: the same PCA coordinates colored by one feature value at a time. Color depends only on the attribute value; marker shape distinguishes reference from generated samples.
"""


def write_visualization_readme(out_root: Path, *, filename: str = "README.md") -> Path:
    """Write the explanatory README into a visualization output root."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / filename
    out_path.write_text(VISUALIZATION_OUTPUT_README, encoding="utf-8")
    return out_path
