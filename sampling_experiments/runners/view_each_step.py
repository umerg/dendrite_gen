"""Utilities to evaluate and persist step-by-step predictions for one graph."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import torch as th
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from sampling_experiments.eval import ExpansionStepEvaluator, StepEvalRecord
from sampling_experiments.loaders import SequenceSetupResult, prepare_sequence_setup


@dataclass
class StepwiseEvaluationResult:
    """Container bundling setup context, predictions, and artifact location."""

    setup: SequenceSetupResult
    records: List[StepEvalRecord]
    artifact_dir: Path | None = None

    @property
    def bundle(self):
        return self.setup.reduction_bundle

    @property
    def context(self):
        return self.setup.context


def _chunks(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def evaluate_sequence_records(
    setup: SequenceSetupResult,
    *,
    batch_size: int = 4,
    evaluator: ExpansionStepEvaluator | None = None,
) -> List[StepEvalRecord]:
    """Run the model over every reduction step and return ordered prediction records."""
    context = setup.context
    model = context.model
    device = context.device
    model = model.to(device).eval()

    evaluator = evaluator or context.method
    if not isinstance(evaluator, ExpansionStepEvaluator):
        raise TypeError("context.method must be an ExpansionStepEvaluator for stepwise evaluation.")
    if hasattr(evaluator, "to"):
        evaluator = evaluator.to(device)
    if hasattr(evaluator, "eval"):
        evaluator.eval()

    all_records: List[StepEvalRecord] = []

    with th.no_grad():
        for chunk in _chunks(setup.reduction_bundle.steps, batch_size):
            batch = Batch.from_data_list(chunk)
            batch = batch.to(device)
            batch_records = evaluator.collect_step_predictions(batch, model)
            all_records.extend(batch_records)

    all_records.sort(key=lambda r: (r.sequence_idx if r.sequence_idx is not None else r.graph_batch_idx))
    return all_records


def _prepare_run_dir(base_dir: Path, graph_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{graph_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_stepwise_results(
    result: StepwiseEvaluationResult,
    *,
    output_dir: Path = Path("sampling_experiments/artifacts/view_each_step"),
) -> Path:
    """Persist evaluation artifacts to disk for notebook consumption."""
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = result.bundle.graph_path
    graph_name = graph_path.stem if graph_path is not None else "graph"
    run_dir = _prepare_run_dir(output_dir, graph_name)

    sequence_path = run_dir / "reduction_sequence.pkl"
    records_path = run_dir / "step_records.pkl"
    with open(sequence_path, "wb") as f:
        pickle.dump(result.bundle, f)
    with open(records_path, "wb") as f:
        pickle.dump(result.records, f)

    summary = []
    for record in result.records:
        summary.append(
            {
                "sequence_idx": record.sequence_idx,
                "step_idx": record.step_idx,
                "reduction_level": record.reduction_level,
                "num_nodes": record.num_nodes,
                "num_leaves": record.num_leaves,
                "losses": record.losses,
            }
        )
    with open(run_dir / "step_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    metadata = {
        "graph_path": str(graph_path) if graph_path is not None else None,
        "num_steps": len(result.bundle.steps),
        "checkpoint": str(result.context.checkpoint_path),
        "ema_beta": result.context.ema_beta,
        "device": str(result.context.device),
        "reduction_seed": result.bundle.reduction_seed,
        "records_path": records_path.name,
        "sequence_path": sequence_path.name,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if OmegaConf is not None:
        with open(run_dir / "config_resolved.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(result.setup.cfg))

    result.artifact_dir = run_dir
    return run_dir


def run_view_each_step(
    *,
    config_path: Path | str,
    checkpoint_path: Path | str,
    graph_path: Path | str,
    overrides: Sequence[str] | None = None,
    ema_beta: float | None = None,
    device: str = "cpu",
    reduction_seed: int = 0,
    batch_size: int = 4,
    output_dir: Path = Path("sampling_experiments/artifacts/view_each_step"),
) -> StepwiseEvaluationResult:
    """High-level helper: load config/model, evaluate sequence, and save artifacts."""
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    graph_path = Path(graph_path)
    setup = prepare_sequence_setup(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        graph_path=graph_path,
        overrides=overrides,
        ema_beta=ema_beta,
        device=device,
        reduction_seed=reduction_seed,
        method_cls=ExpansionStepEvaluator,
    )
    records = evaluate_sequence_records(setup, batch_size=batch_size)
    result = StepwiseEvaluationResult(setup=setup, records=records)
    save_stepwise_results(result, output_dir=output_dir)
    return result
