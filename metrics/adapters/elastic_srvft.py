"""Boundary for the external Elastic SRVFT implementation.

The external repository has not been selected or audited yet. In particular,
its published formulation removes full SO(3) rotations, whereas this project
needs an SO(2)-only quotient around the preferred axis. This module deliberately
does not guess at the repository's API.
"""

from __future__ import annotations


class ElasticSRVFTNotConfigured(RuntimeError):
    """Raised until a concrete Elastic SRVFT package/API has been configured."""


def elastic_srvft_distance(*_args, **_kwargs) -> float:
    """Raise an actionable error until the external implementation is integrated."""
    raise ElasticSRVFTNotConfigured(
        "Elastic SRVFT is not configured. Clone/install the selected implementation, "
        "then audit every internal rotation alignment so it quotients only SO(2), not SO(3), "
        "before completing metrics.adapters.elastic_srvft."
    )
