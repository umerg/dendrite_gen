"""Tests for class-labelled metric-study dataset discovery and sampling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from visualization.metric_study.dataset import (
    TreeRecord,
    discover_tree_records,
    parse_swc_labels,
    select_balanced_sample,
    validate_tree_records,
)


def _write_swc(
    path: Path,
    *,
    cell_class: int | str | None,
    cell_type: str | None,
    extra_header: str = "# generated fixture\n",
) -> None:
    lines = [extra_header]
    if cell_class is not None:
        lines.append(f"# cell_class {cell_class}\n")
    if cell_type is not None:
        lines.append(f"# cell_type {cell_type}\n")
    lines.extend(("1 1 0 0 0 1 -1\n", "2 3 0 0 1 1 1\n"))
    path.write_text("".join(lines), encoding="utf-8")


def test_parse_swc_labels_reads_header_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tree.swc"
    path.write_text(
        "# generated fixture\n"
        "#\tcell_class\t5\n"
        "#   cell_type   6P-IT   \n"
        "1 1 0 0 0 1 -1\n",
        encoding="utf-8",
    )

    assert parse_swc_labels(path) == (5, "6P-IT")


@pytest.mark.parametrize(
    ("cell_class", "cell_type", "message"),
    [
        (None, "23P", "cell_class"),
        (0, None, "cell_type"),
        ("not-an-int", "23P", "expected an integer"),
        (-1, "23P", "non-negative"),
    ],
)
def test_parse_swc_labels_rejects_missing_or_invalid_metadata(
    tmp_path: Path,
    cell_class: int | str | None,
    cell_type: str | None,
    message: str,
) -> None:
    path = tmp_path / "bad.swc"
    _write_swc(path, cell_class=cell_class, cell_type=cell_type)

    with pytest.raises(ValueError, match=message):
        parse_swc_labels(path)


def test_parse_swc_labels_rejects_duplicate_header_key(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.swc"
    _write_swc(
        path,
        cell_class=0,
        cell_type="23P",
        extra_header="# cell_class 1\n",
    )

    with pytest.raises(ValueError, match="Duplicate 'cell_class'"):
        parse_swc_labels(path)


def test_discover_tree_records_supports_configurable_splits(
    tmp_path: Path,
) -> None:
    fit = tmp_path / "source_fit"
    holdout = tmp_path / "source_holdout"
    fit.mkdir()
    holdout.mkdir()
    _write_swc(fit / "b.swc", cell_class=1, cell_type="4P")
    _write_swc(fit / "a.swc", cell_class=0, cell_type="23P")
    _write_swc(holdout / "c.csv.swc", cell_class=1, cell_type="4P")
    (fit / "notes.txt").write_text("ignored", encoding="utf-8")
    _write_swc(fit / "._resource.swc", cell_class=0, cell_type="23P")

    records = discover_tree_records(
        tmp_path,
        split_dirs={"train": "source_fit", "test": holdout},
    )

    assert [record.tree_id for record in records] == ["a", "b", "c.csv"]
    assert [record.split for record in records] == ["train", "train", "test"]
    assert [(record.cell_class, record.cell_type) for record in records] == [
        (0, "23P"),
        (1, "4P"),
        (1, "4P"),
    ]
    assert all(record.swc_path.is_absolute() for record in records)


def test_discover_tree_records_rejects_duplicate_ids_across_splits(
    tmp_path: Path,
) -> None:
    for split in ("train", "test"):
        directory = tmp_path / split
        directory.mkdir()
        _write_swc(directory / "same.swc", cell_class=0, cell_type="23P")

    with pytest.raises(ValueError, match="Duplicate tree_id 'same'"):
        discover_tree_records(tmp_path, split_dirs=("train", "test"))


@pytest.mark.parametrize(
    "records",
    [
        (
            TreeRecord("a", Path("a.swc"), "train", 0, "23P"),
            TreeRecord("b", Path("b.swc"), "train", 0, "4P"),
        ),
        (
            TreeRecord("a", Path("a.swc"), "train", 0, "23P"),
            TreeRecord("b", Path("b.swc"), "train", 1, "23P"),
        ),
    ],
)
def test_validate_tree_records_rejects_inconsistent_label_mapping(
    records: tuple[TreeRecord, TreeRecord],
) -> None:
    with pytest.raises(ValueError, match="Inconsistent labels"):
        validate_tree_records(records)


def test_select_balanced_sample_is_balanced_seeded_and_input_order_independent(
    tmp_path: Path,
) -> None:
    records = tuple(
        TreeRecord(
            tree_id=f"class-{cell_class}-tree-{index}",
            swc_path=tmp_path / f"{cell_class}-{index}.swc",
            split="train",
            cell_class=cell_class,
            cell_type=f"type-{cell_class}",
        )
        for cell_class in range(3)
        for index in range(8)
    )

    first = select_balanced_sample(records, 3, seed=17)
    repeated = select_balanced_sample(reversed(records), 3, seed=17)
    other_seed = select_balanced_sample(records, 3, seed=18)

    assert first == repeated
    assert Counter(record.cell_class for record in first) == {0: 3, 1: 3, 2: 3}
    assert {record.tree_id for record in first} != {
        record.tree_id for record in other_seed
    }


def test_select_balanced_sample_can_use_smallest_class_and_rejects_shortage(
    tmp_path: Path,
) -> None:
    records = (
        TreeRecord("a", tmp_path / "a.swc", "train", 0, "23P"),
        TreeRecord("b", tmp_path / "b.swc", "train", 0, "23P"),
        TreeRecord("c", tmp_path / "c.swc", "train", 1, "4P"),
    )

    selected = select_balanced_sample(records, None)
    assert Counter(record.cell_class for record in selected) == {0: 1, 1: 1}

    with pytest.raises(ValueError, match="class 1: 1"):
        select_balanced_sample(records, 2)
