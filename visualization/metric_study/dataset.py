"""Class-labelled ground-truth SWC dataset utilities.

The dataset location is deliberately supplied by the caller.  Class metadata
is read from the comment header of each SWC rather than inferred from file
order or from a separate, potentially stale manifest.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class TreeRecord:
    """One class-labelled ground-truth tree."""

    tree_id: str
    swc_path: Path
    split: str
    cell_class: int
    cell_type: str


def parse_swc_labels(swc_path: str | Path) -> tuple[int, str]:
    """Return ``(cell_class, cell_type)`` from an SWC comment header.

    Expected metadata lines are ``# cell_class <integer>`` and
    ``# cell_type <non-empty name>``.  Other header comments and blank lines
    are ignored.  Metadata after the first SWC data row is intentionally not
    treated as header metadata.
    """

    path = Path(swc_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SWC file does not exist: {path}")

    values: dict[str, tuple[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break

            comment = stripped[1:].strip()
            parts = comment.split(maxsplit=1)
            key = parts[0] if parts else ""
            if key not in {"cell_class", "cell_type"}:
                continue
            value = parts[1].strip() if len(parts) == 2 else ""
            if not value:
                raise ValueError(
                    f"Empty {key!r} metadata in {path} at line {line_number}"
                )
            if key in values:
                previous_value, previous_line = values[key]
                raise ValueError(
                    f"Duplicate {key!r} metadata in {path} at lines "
                    f"{previous_line} and {line_number} "
                    f"({previous_value!r}, {value!r})"
                )
            values[key] = (value, line_number)

    missing = [key for key in ("cell_class", "cell_type") if key not in values]
    if missing:
        raise ValueError(
            f"Missing SWC header metadata in {path}: {', '.join(missing)}"
        )

    class_text = values["cell_class"][0]
    try:
        cell_class = int(class_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid cell_class {class_text!r} in {path}; expected an integer"
        ) from exc
    if cell_class < 0:
        raise ValueError(
            f"Invalid cell_class {cell_class} in {path}; "
            "expected a non-negative integer"
        )

    return cell_class, values["cell_type"][0]


def validate_tree_records(records: Iterable[TreeRecord]) -> None:
    """Validate identifiers and the one-to-one class/type label mapping."""

    id_paths: dict[str, Path] = {}
    class_to_type: dict[int, str] = {}
    type_to_class: dict[str, int] = {}

    for record in records:
        if not record.tree_id:
            raise ValueError(f"Tree record has an empty tree_id: {record.swc_path}")
        if not record.split.strip():
            raise ValueError(f"Tree record has an empty split: {record.tree_id!r}")
        if (
            isinstance(record.cell_class, bool)
            or not isinstance(record.cell_class, int)
            or record.cell_class < 0
        ):
            raise ValueError(
                f"Tree record {record.tree_id!r} has invalid cell_class "
                f"{record.cell_class!r}"
            )
        if not record.cell_type.strip():
            raise ValueError(
                f"Tree record {record.tree_id!r} has an empty cell_type"
            )
        if record.tree_id in id_paths:
            raise ValueError(
                f"Duplicate tree_id {record.tree_id!r}: "
                f"{id_paths[record.tree_id]} and {record.swc_path}"
            )
        id_paths[record.tree_id] = record.swc_path

        expected_type = class_to_type.setdefault(record.cell_class, record.cell_type)
        if expected_type != record.cell_type:
            raise ValueError(
                f"Inconsistent labels for cell_class {record.cell_class}: "
                f"{expected_type!r} and {record.cell_type!r}"
            )

        expected_class = type_to_class.setdefault(record.cell_type, record.cell_class)
        if expected_class != record.cell_class:
            raise ValueError(
                f"Inconsistent labels for cell_type {record.cell_type!r}: "
                f"classes {expected_class} and {record.cell_class}"
            )


def discover_tree_records(
    dataset_root: str | Path,
    *,
    split_dirs: Sequence[str] | Mapping[str, str | Path] = (
        "train",
        "val",
        "test",
    ),
) -> tuple[TreeRecord, ...]:
    """Discover labelled SWCs in caller-configured split directories.

    ``split_dirs`` may be a sequence of directory names relative to
    ``dataset_root``, or a mapping from the desired split label to a relative
    or absolute directory path.  SWCs are discovered directly within each
    split directory and returned in deterministic split/filename order.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    if isinstance(split_dirs, Mapping):
        configured_splits = list(split_dirs.items())
    elif isinstance(split_dirs, str):
        configured_splits = [(split_dirs, split_dirs)]
    else:
        configured_splits = [(name, name) for name in split_dirs]
    if not configured_splits:
        raise ValueError("At least one split directory must be configured")

    split_names: set[str] = set()
    records: list[TreeRecord] = []
    for raw_split, raw_directory in configured_splits:
        split = str(raw_split).strip()
        if not split:
            raise ValueError("Split labels must be non-empty")
        if split in split_names:
            raise ValueError(f"Duplicate split label: {split!r}")
        split_names.add(split)

        directory = Path(raw_directory).expanduser()
        if not directory.is_absolute():
            directory = root / directory
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Configured split directory for {split!r} does not exist: {directory}"
            )

        swc_paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and not path.name.startswith("._")
            and path.name.endswith(".swc")
        )
        for swc_path in swc_paths:
            cell_class, cell_type = parse_swc_labels(swc_path)
            records.append(
                TreeRecord(
                    tree_id=swc_path.name[: -len(".swc")],
                    swc_path=swc_path,
                    split=split,
                    cell_class=cell_class,
                    cell_type=cell_type,
                )
            )

    validate_tree_records(records)
    return tuple(records)


def select_balanced_sample(
    records: Iterable[TreeRecord],
    per_class: int | None,
    *,
    seed: int = 0,
) -> tuple[TreeRecord, ...]:
    """Select the same deterministic number of records from every class.

    The selection is independent of input order.  ``per_class=None`` uses the
    size of the smallest class.  A seeded SHA-256 rank makes the chosen subset
    reproducible without depending on global random state.
    """

    record_tuple = tuple(records)
    if not record_tuple:
        raise ValueError("Cannot sample from an empty record collection")
    validate_tree_records(record_tuple)

    by_class: dict[int, list[TreeRecord]] = defaultdict(list)
    for record in record_tuple:
        by_class[record.cell_class].append(record)

    if per_class is None:
        sample_size = min(len(group) for group in by_class.values())
    elif isinstance(per_class, bool) or not isinstance(per_class, int):
        raise TypeError("per_class must be an integer or None")
    elif per_class <= 0:
        raise ValueError("per_class must be positive")
    else:
        sample_size = per_class

    undersized = {
        cell_class: len(group)
        for cell_class, group in by_class.items()
        if len(group) < sample_size
    }
    if undersized:
        details = ", ".join(
            f"class {cell_class}: {count}"
            for cell_class, count in sorted(undersized.items())
        )
        raise ValueError(
            f"Requested {sample_size} records per class, but some classes "
            f"are smaller ({details})"
        )

    def selection_rank(record: TreeRecord) -> tuple[bytes, str, str, str]:
        identity = "\0".join((str(seed), str(record.cell_class), record.tree_id))
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        return digest, record.tree_id, record.split, record.swc_path.as_posix()

    selected: list[TreeRecord] = []
    for cell_class in sorted(by_class):
        chosen = sorted(by_class[cell_class], key=selection_rank)[:sample_size]
        selected.extend(sorted(chosen, key=lambda item: (item.tree_id, item.split)))

    return tuple(selected)
