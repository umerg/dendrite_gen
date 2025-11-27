import pickle
from pathlib import Path
from time import time
import logging

import psutil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch as th
from matplotlib.figure import Figure
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Optional / guarded imports (Hydra, OmegaConf, wandb)
try:  # Hydra runtime config access
    from hydra.core.hydra_config import HydraConfig  # type: ignore
except Exception:  # pragma: no cover
    HydraConfig = None  # fallback when running outside hydra

try:  # Config serialization for wandb
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    OmegaConf = None

try:  # Experiment tracking (optional)
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from .metrics import Metric
from .model import EMA, EMA1


class Trainer:
    def __init__(
        self,
        model,
        method,
        train_dataloader,
        train_graphs: list[nx.Graph],
        validation_graphs: list[nx.Graph],
        test_graphs: list[nx.Graph],
        metrics: list[Metric],
        cfg,
    ):
        self.train_iterator = iter(train_dataloader)
        self.train_graphs = train_graphs
        self.validation_graphs = validation_graphs
        self.test_graphs = test_graphs
        self.metrics = metrics
        self.cfg = cfg

        self.rng = np.random.default_rng(0)
        # Prefer CUDA, fallback to CPU (MPS has stability issues with PyG)
        if not cfg.debugging:
            if th.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
        print(f"Selected device: {self.device}")
        self.method = method.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), cfg.training.lr)

        # Optional LR scheduler (Cosine Annealing over training horizon)
        self.scheduler = None
        scheduler_name = getattr(cfg.training, "lr_scheduler", None)
        if scheduler_name is not None:
            scheduler_name = str(scheduler_name).lower()
        if scheduler_name in ("cosine", "cosine_annealing"):
            # By default, anneal over the full training horizon
            T_max = getattr(cfg.training, "scheduler_T_max", cfg.training.num_steps)
            eta_min = getattr(cfg.training, "scheduler_eta_min", 0.0)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )

        # EMA - lets keep EMA for future use but keep beta=1 for now
        self.ema_models = {
            beta: EMA(
                model=self.model, beta=beta, gamma=cfg.ema.gamma, power=cfg.ema.power
            )
            if beta != 1
            else EMA1(model=self.model)
            for beta in cfg.ema.betas
        }

        self.all_models = {
            "model": self.model,
            **{f"model_ema_{c}": m for c, m in self.ema_models.items()},
        }

        # checkpoint / artifact directory (Hydra-aware fallback)
        if HydraConfig is not None:
            try:
                self.output_dir = Path(HydraConfig.get().runtime.output_dir)
            except Exception:  # pragma: no cover
                self.output_dir = Path("./outputs")
        else:
            # Running outside Hydra (e.g., direct script execution)
            self.output_dir = Path("./outputs")

        # Resume from checkpoint
        if cfg.training.resume:
            self.resume_from_checkpoint(cfg.training.resume)
            print(f"Resuming training from step {self.step}")
        else:
            self.step = 0
            self.best_validation_scores = {beta: -1 for beta in cfg.ema.betas} # what are EMA betas? TODO
            self.run_id = None

        # Wandb (only if requested AND available AND OmegaConf present)
        if cfg.wandb.logging and wandb is not None and OmegaConf is not None:
            try:
                self.wandb_run = wandb.init(
                    project="tree-generation",
                    config=OmegaConf.to_container(cfg, resolve=True),
                    name=cfg.name,
                    resume=self.run_id,
                )
                self.run_id = self.wandb_run.id
            except Exception as e:  # pragma: no cover
                print(f"[wandb disabled] {e}")
                self.wandb_run = None
                self.run_id = None
        else:
            self.wandb_run = None
            self.run_id = None

        num_parameters = sum(p.numel() for p in model.parameters())
        print(f"Total number of model parameters: {num_parameters / 1e6} Million")
        
        # Set up logging to file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            log_file = self.output_dir / "training.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.info(f"Training initialized on device: {self.device}")
        self.logger.info(f"Total model parameters: {num_parameters / 1e6:.6f} Million")

    def save_checkpoint(self):
        checkpoint = {
            name: model.state_dict()
            for name, model in self.all_models.items()
            if model is not None
        }
        checkpoint["optimizer"] = self.optimizer.state_dict()
        if getattr(self, "scheduler", None) is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        checkpoint["step"] = self.step
        checkpoint["best_validation_scores"] = self.best_validation_scores
        checkpoint["run_id"] = self.run_id

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        th.save(checkpoint, checkpoint_dir / f"step_{self.step}.pt")

    def resume_from_checkpoint(self, resume):
        checkpoint_dir = self.output_dir / "checkpoints"
        assert checkpoint_dir.exists(), "No checkpoints found."
        if isinstance(resume, bool):
            # resume from latest checkpoint
            checkpoint_path = max(
                checkpoint_dir.glob("step_*.pt"),
                key=lambda f: int(f.stem.split("_")[1]),
            )
        else:
            # resume from specific checkpoint
            checkpoint_path = checkpoint_dir / f"step_{resume}.pt"

        checkpoint = th.load(checkpoint_path)
        for name, model in self.all_models.items():
            if model is not None:
                model.load_state_dict(checkpoint[name])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and getattr(self, "scheduler", None) is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = checkpoint["step"]
        self.best_validation_scores = checkpoint["best_validation_scores"]
        self.run_id = checkpoint["run_id"]

    def train(self):
        print(f"Training model on {self.device}")
        if hasattr(self, 'logger'):
            self.logger.info(f"Starting training on {self.device}")
        self.model.train()

        last_step = False
        while not last_step:
            # print(f"Starting step {self.step + 1}/{self.cfg.training.num_steps}")
            self.step += 1
            last_step = self.step == self.cfg.training.num_steps

            step_start_time = time()
            batch = next(self.train_iterator)
            loss_terms = self.run_step(batch)
            if self.cfg.training.log_interval > 0 and (
                self.step % self.cfg.training.log_interval == 0 or last_step
            ):
                loss_terms["step_time"] = time() - step_start_time
                self.log({"training": loss_terms})

            if self.cfg.validation.interval > 0 and (
                self.step >= self.cfg.validation.first_step
                and self.step % self.cfg.validation.interval == 0
                or last_step
            ):
                if self.device == "cuda":
                    th.cuda.empty_cache()
                self.run_validation()

                if self.cfg.training.save_checkpoint:
                    self.save_checkpoint()

                if self.device == "cuda":
                    th.cuda.empty_cache()

    def test(self):
        print(f"Testing model at {self.step} steps on {self.device}")

        # Test for all EMA beta values
        test_results = {}
        for beta in self.cfg.ema.betas:
            test_results[f"ema_{beta}"] = self.evaluate(self.test_graphs, beta)

        # Log results
        self.log({"test": test_results})

        # Dump results
        if self.cfg.training.save_checkpoint:
            test_dir = self.output_dir / "test"
            test_dir.mkdir(exist_ok=True)
            with open(test_dir / f"step_{self.step}.pkl", "wb") as f:
                pickle.dump(test_results, f)

    def run_step(self, batch):
        # # print memory usage - for batch sizing etc
        # print(f"Memory allocated before step: {th.cuda.memory_allocated(self.device) / 1024 ** 2:.2f} MB")
        # print(f"Memory cached before step: {th.cuda.memory_reserved(self.device) / 1024 ** 2:.2f} MB")
        # # print RAM usage
        # process = psutil.Process()
        # print(f"RAM usage before step: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        batch = batch.to(self.device, non_blocking=True)
        loss, loss_terms = self.method.get_loss(
            batch=batch, model=self.model,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            # Step LR scheduler once per global training step
            self.scheduler.step()

        for model in list(self.ema_models.values()):
            if model is not None:
                model.update(step=self.step)
        # Optionally log current LR
        loss_terms["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return loss_terms

    def run_validation(self):
        print(f"Running validation at {self.step} steps.")
        if hasattr(self, 'logger'):
            self.logger.info(f"Running validation at step {self.step}")

        # --- VALIDATION LOOP (metrics + optional plots) ---
        # We gate metric computation & test-trigger logic with cfg.validation.enable_metrics.
        # We gate example plotting with cfg.validation.enable_plots.
        # Original code retained inside conditionals for future reactivation.
        val_results = {}
        test_results = {}
        enable_metrics = getattr(self.cfg.validation, 'enable_metrics', True)
        enable_plots = getattr(self.cfg.validation, 'enable_plots', True)

        for beta in self.cfg.ema.betas:
            # Always generate graphs (needed for plots & potential metrics later)
            val_results[f"ema_{beta}"] = self.evaluate(self.validation_graphs, beta)

            if enable_metrics:
                # --- METRIC VALIDATION SCORE BLOCK (original logic) ---
                unique_novel_valid_keys = [
                    str(m) for m in self.metrics if "UniqueNovelValid" in str(m)
                ]
                if len(unique_novel_valid_keys) > 0:
                    validation_score = val_results[f"ema_{beta}"][
                        unique_novel_valid_keys[0]
                    ]
                else:
                    # Ratio metric used as inverse score previously
                    validation_score = 1 / val_results[f"ema_{beta}"]["Ratio"]

                # Evaluate on test set if validation score improved
                if validation_score >= self.best_validation_scores[beta]:
                    self.best_validation_scores[beta] = validation_score
                    test_results[f"ema_{beta}"] = self.evaluate(self.test_graphs, beta)
            else:
                # Metrics disabled: insert placeholder
                val_results[f"ema_{beta}"]["metrics_disabled"] = True

        # Log results (test_results empty if metrics disabled & no improvements tracked)
        self.log({"validation": val_results, "test": test_results})

        # Dump results (persist even if metrics disabled to keep artifacts of generated graphs/plots)
        if self.cfg.training.save_checkpoint:
            val_dir = self.output_dir / "validation"
            val_dir.mkdir(exist_ok=True)
            with open(val_dir / f"step_{self.step}.pkl", "wb") as f:
                pickle.dump(val_results, f)
            if test_results:
                test_dir = self.output_dir / "test"
                test_dir.mkdir(exist_ok=True)
                with open(test_dir / f"step_{self.step}.pkl", "wb") as f:
                    pickle.dump(test_results, f)

    @th.no_grad()
    def evaluate(self, eval_graphs: list[nx.Graph], beta):
        """Evaluate model for given beta on given graphs."""
        model = self.ema_models[beta]

        # Shuffle prediction order to make size distribution more uniform
        pred_perm = self.rng.permutation(np.arange(len(eval_graphs)))

        # Select target number of nodes and split into batches
        target_size = np.array([len(g) for g in eval_graphs])[pred_perm]
        bs = (
            self.cfg.validation.batch_size
            if self.cfg.validation.batch_size is not None
            else self.cfg.training.batch_size
        )
        batches = [target_size[i : i + bs] for i in range(0, len(target_size), bs)]

        results = {}

        # Generate graphs
        pred_graphs = []
        for batch in batches:
            pred_graphs_batch = self.method.sample_graphs(
                target_size=th.tensor(batch, device=self.device),
                model=model,
            )  # returns list[nx.Graph] with geometric node attrs
            pred_graphs += pred_graphs_batch
        # Reorder according to original permutation
        results["pred_graphs"] = [pred_graphs[i] for i in pred_perm]
        if self.device == "cuda":
            th.cuda.empty_cache()

        # Consistency assertions: all graphs must have 'pos' attribute per node
        def _assert_geometric(graphs: list[nx.Graph]):
            if not graphs:
                return
            # Check dimensionality consistency
            first_node = next(iter(graphs[0].nodes()))
            ref_dim = len(graphs[0].nodes[first_node]['pos']) if 'pos' in graphs[0].nodes[first_node] else None
            for G in graphs:
                for n in G.nodes():
                    assert 'pos' in G.nodes[n], "Graph node missing 'pos' attribute"
                    assert isinstance(G.nodes[n]['pos'], (list, tuple, np.ndarray)), "'pos' must be list/tuple/ndarray"
                    assert len(G.nodes[n]['pos']) == ref_dim, "Inconsistent position dimensionality across graphs"
        _assert_geometric(eval_graphs)
        _assert_geometric(results["pred_graphs"])

        # Metric computation gated
        enable_metrics = getattr(self.cfg.validation, 'enable_metrics', True)
        if enable_metrics:
            # Validate graphs (original metric loop)
            for metric in self.metrics:
                results[str(metric)] = metric(
                    reference_graphs=eval_graphs,
                    predicted_graphs=pred_graphs,
                    train_graphs=self.train_graphs,
                )

            if self.cfg.validation.per_graph_size:
                for n in set(target_size):
                    eval_graphs_n = [g for g in eval_graphs if len(g) == n]
                    pred_graphs_n = [g for g in pred_graphs if len(g) == n]
                    results[f"size_{n}"] = {}
                    for metric in self.metrics:
                        results[f"size_{n}"][str(metric)] = metric(
                            reference_graphs=eval_graphs_n,
                            predicted_graphs=pred_graphs_n,
                            train_graphs=self.train_graphs,
                        )
        else:
            results['metrics_disabled'] = True

        # Example plot generation gated
        enable_plots = getattr(self.cfg.validation, 'enable_plots', True)

        if enable_plots:
            # Plot using stored geometric positions (first two coordinates projected to XY)
            max_examples = min(8, len(results["pred_graphs"]))
            cols = min(4, max_examples)
            rows = int(np.ceil(max_examples / cols))
            if max_examples > 0:
                fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
                if isinstance(axs, plt.Axes):
                    axs = np.array([[axs]])
                axs = np.atleast_2d(axs)
                for i in range(max_examples):
                    G = results["pred_graphs"][i]
                    r = i // cols; c = i % cols
                    ax = axs[r][c]
                    # Extract positions
                    pos_dict = {n: G.nodes[n]['pos'][:2] for n in G.nodes()}
                    # Draw edges manually for consistent style
                    for u, v in G.edges():
                        p1 = pos_dict[u]; p2 = pos_dict[v]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=1.0, zorder=1)
                    # Scatter nodes (uniform color since no leaf attribute)
                    xs = [pos_dict[n][0] for n in G.nodes()]
                    ys = [pos_dict[n][1] for n in G.nodes()]
                    ax.scatter(xs, ys, c='tab:blue', s=30, edgecolors='k', linewidths=0.5, zorder=2)
                    ax.set_title(f"N={G.number_of_nodes()}")
                    ax.set_xticks([]); ax.set_yticks([])
                # Hide unused axes
                for j in range(max_examples, rows * cols):
                    r = j // cols; c = j % cols
                    axs[r][c].axis('off')
                fig.tight_layout()
                results["examples"] = fig
                # Save figure to eval_plots directory
                eval_plots_dir = self.output_dir / 'eval_plots'
                eval_plots_dir.mkdir(exist_ok=True)
                fig_path = eval_plots_dir / f"step_{self.step}_beta_{beta}.png"
                fig.savefig(fig_path, dpi=150)
                results["examples_path"] = str(fig_path)

                # NEW: side-by-side comparison plots (reference vs predicted) for same permutation order
                eval_graphs_perm = [eval_graphs[i] for i in pred_perm]
                comp_rows = max_examples
                comp_cols = 2  # reference | predicted
                comp_fig, comp_axs = plt.subplots(comp_rows, comp_cols, figsize=(comp_cols * 5, comp_rows * 3.5))
                if isinstance(comp_axs, plt.Axes):
                    comp_axs = np.array([[comp_axs]])
                comp_axs = np.atleast_2d(comp_axs)
                for i in range(max_examples):
                    refG = eval_graphs_perm[i]
                    predG = results["pred_graphs"][i]
                    for col_idx, (Gcur, title_prefix) in enumerate([(refG, 'Eval'), (predG, 'Pred')]):
                        axc = comp_axs[i][col_idx]
                        pos_cur = {n: Gcur.nodes[n]['pos'][:2] for n in Gcur.nodes()}
                        # edges
                        for u, v in Gcur.edges():
                            p1 = pos_cur[u]; p2 = pos_cur[v]
                            axc.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=1.0, zorder=1)
                        xs = [pos_cur[n][0] for n in Gcur.nodes()]
                        ys = [pos_cur[n][1] for n in Gcur.nodes()]
                        axc.scatter(xs, ys, c='tab:blue' if title_prefix=='Eval' else 'tab:orange', s=30, edgecolors='k', linewidths=0.5, zorder=2)
                        axc.set_title(f"{title_prefix} N={Gcur.number_of_nodes()}")
                        axc.set_xticks([]); axc.set_yticks([])
                comp_fig.tight_layout()
                comp_path = eval_plots_dir / f"step_{self.step}_beta_{beta}_compare.png"
                comp_fig.savefig(comp_path, dpi=150)
                results["examples_compare"] = comp_fig
                results["examples_compare_path"] = str(comp_path)
            else:
                results["examples"] = None
                results["examples_path"] = None
                results["examples_compare"] = None
                results["examples_compare_path"] = None
        else:
            results["examples"] = None
            results["examples_path"] = None
            results["examples_compare"] = None
            results["examples_compare_path"] = None

        return results

    def log(self, log_dict: dict, prefix: str = "", indent: int = 0):
        """Logs an arbitrarily nested dict to the console and wandb."""
        for key, value in log_dict.items():
            if isinstance(value, dict):
                print(f"{'   ' * indent}{key}:")
                self.log(value, prefix=f"{prefix}{key}/", indent=indent + 1)
            elif isinstance(value, float):
                log_msg = f"{'   ' * indent}{key}: {value}"
                print(log_msg)
                # Also log to file
                if hasattr(self, 'logger'):
                    self.logger.info(f"{prefix}{key}: {value}")
                if self.cfg.wandb.logging and self.wandb_run is not None:
                    self.wandb_run.log({f"{prefix}{key}": value}, step=self.step)
            elif isinstance(value, Figure):
                # Wandb logging for figures currently disabled or wandb import commented out.
                # Keeping placeholder for future reactivation.
                if getattr(self.cfg.wandb, 'logging', False) and self.wandb_run is not None and wandb is not None:
                    try:
                        self.wandb_run.log({f"{prefix}{key}": wandb.Image(value)}, step=self.step)
                    except Exception as e:
                        print(f"[wandb logging skipped] {e}")
