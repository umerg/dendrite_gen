# Local external metric checkouts

This directory is reserved for temporary, local checkouts of research metric
implementations, including Elastic SRVFT. Its checkout contents are ignored so
an ordinary nested clone is not accidentally committed as an embedded Git
repository.

For initial inspection, clone the repository below this directory and install
it into the `trees` environment if it exposes a Python package. Do not add its
directory to `sys.path` from project code.

Once the implementation and license have been audited, make it reproducible by
pinning an installable Git revision or by intentionally converting the checkout
to a Git submodule. Keep all project-facing calls behind an adapter in
`metrics/adapters/`.

The Elastic SRVFT implementation needs an additional scientific audit: the
paper optimizes over full SO(3), while this project must preserve the preferred
axis and optimize only over rotations around it.
