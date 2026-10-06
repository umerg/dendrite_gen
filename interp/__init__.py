"""Interpretability tooling for the SO(2)-EGNN expansion generator.

Probes P1–P6 and steering experiments; see docs in the interp plan. Everything here is
read-only with respect to the training stack: models are built through
``main.get_expansion_items`` so the architecture matches the checkpoints exactly.
"""
