from asyncio.log import logger
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

from utils.tmd import compute_tmd_mixed, compute_tmd_embedding
from validation.dist_metrics import compute_distribution_metrics, build_gt_cache
from validation.plot import plot_graph_grid_angles, DEFAULT_ANGLES
# NOTE: validation.teacher_forced_eval is imported lazily inside the methods that use it
# (_tf_batches_for / evaluate) to avoid a circular import — that module pulls in
# graph_generation.method.helpers, which re-enters the graph_generation package during init.

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


def _maybe_add_alias(alias_cfg: dict, source_obj, source_key: str, alias_key: str):
    if source_obj is None or not hasattr(source_obj, source_key):
        return
    value = getattr(source_obj, source_key)
    if value is not None:
        alias_cfg[alias_key] = value


def build_wandb_config(cfg):
    base_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(base_cfg, dict):
        base_cfg = {}

    alias_cfg = {}

    model_cfg = getattr(cfg, "model", None)
    model_aliases = {
        "num_layers": "model_num_layers",
        "feats_dim": "model_feats_dim",
        "m_dim": "model_m_dim",
        "tmd_hidden_dim": "model_tmd_hidden_dim",
        "offset_head_hidden": "model_offset_head_hidden",
        "global_linear_attn_heads": "model_global_linear_attn_heads",
        "global_linear_attn_dim_head": "model_global_linear_attn_dim_head",
        "num_global_tokens": "model_num_global_tokens",
    }
    for source_key, alias_key in model_aliases.items():
        _maybe_add_alias(alias_cfg, model_cfg, source_key, alias_key)

    training_cfg = getattr(cfg, "training", None)
    _maybe_add_alias(alias_cfg, training_cfg, "num_steps", "training_num_steps")
    if training_cfg is not None:
        # Current configs use `training.lr`; keep a fallback for `learning_rate`.
        if hasattr(training_cfg, "lr") and getattr(training_cfg, "lr") is not None:
            alias_cfg["training_learning_rate"] = getattr(training_cfg, "lr")
        elif hasattr(training_cfg, "learning_rate") and getattr(training_cfg, "learning_rate") is not None:
            alias_cfg["training_learning_rate"] = getattr(training_cfg, "learning_rate")

    return {**base_cfg, **alias_cfg}


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
        pos_scale_factor: float | None = None,
    ):
        self.pos_scale_factor = pos_scale_factor
        self.train_iterator = iter(train_dataloader)
        self.train_graphs = train_graphs
        self.validation_graphs = validation_graphs
        self.test_graphs = test_graphs
        self.metrics = metrics
        self.cfg = cfg

        self.rng = np.random.default_rng(0)
        # Per-eval-set caches for the distribution metrics: the GT-fit objects
        # (morpho mean/std, TMD PCA, MMD bandwidths, Sholl radii) and the
        # real-vs-real floor are model-independent, so computed once per eval set.
        self._eval_cache: dict[int, dict] = {}
        self._floor_cache: dict[int, dict] = {}
        self._tf_batch_cache: dict[int, list] = {}  # GT reduction batches for TF validation
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
            print(f"Resumed from step {self.step}, LR={self.optimizer.param_groups[0]['lr']}")
            # Fork into a FRESH wandb run instead of rejoining the checkpoint's
            # original run. Keeps the loaded model/optimizer/scheduler/step but
            # drops the stored run_id so wandb.init() starts a new experiment —
            # prevents a resumed run (e.g. branching off step 75k to test a config
            # change) from overwriting the source run's logged history.
            if getattr(cfg.wandb, "new_run", False):
                print(f"[wandb] new_run=True -> forking a fresh wandb run (dropping run_id={self.run_id})")
                self.run_id = None
        else:
            self.step = 0
            self.best_validation_scores = {beta: -1 for beta in cfg.ema.betas} # what are EMA betas? TODO
            self.run_id = None

        # Wandb (only if requested AND available AND OmegaConf present)
        if cfg.wandb.logging and wandb is not None and OmegaConf is not None:
            try:
                self.wandb_run = wandb.init(
                    project="tree_gen",
                    config=build_wandb_config(cfg),
                    name=cfg.name,
                    id=self.run_id,
                    resume="allow" if self.run_id else None,
                )
                self.run_id = self.wandb_run.id
                # Plot everything against our own training step instead of
                # wandb's internal global step. On resume, wandb restores its
                # internal step to the previous run's MAX logged step and
                # silently drops any log() at an earlier/equal step ("ignoring
                # partial history record"). Using a custom step metric makes
                # records land immediately regardless of the resumed step.
                self.wandb_run.define_metric("train_step")
                self.wandb_run.define_metric("*", step_metric="train_step")
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
        if isinstance(resume, str) and (resume.endswith(".pt") or Path(resume).is_file()):
            # resume from explicit file path
            checkpoint_path = Path(resume)
        else:
            checkpoint_dir = self.output_dir / "checkpoints"
            assert checkpoint_dir.exists(), "No checkpoints found."
            if isinstance(resume, bool):
                # resume from latest checkpoint
                checkpoint_path = max(
                    checkpoint_dir.glob("step_*.pt"),
                    key=lambda f: int(f.stem.split("_")[1]),
                )
            else:
                # resume from specific step number
                checkpoint_path = checkpoint_dir / f"step_{resume}.pt"

        checkpoint = th.load(checkpoint_path, map_location=self.device)
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
            _t0_data = time()
            batch = next(self.train_iterator)
            t_data_load = time() - _t0_data
            loss_terms = self.run_step(batch)
            loss_terms["t_data_load"] = t_data_load
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

        _t0_transfer = time()
        batch = batch.to(self.device, non_blocking=True)
        t_gpu_transfer = time() - _t0_transfer

        if getattr(self.cfg, "debugging", False):
            batch_vec = getattr(batch, "batch", None)
            if batch_vec is not None and batch_vec.numel() > 0:
                sizes = th.bincount(batch_vec.detach().cpu())
                sizes = sizes[sizes > 0]
                if sizes.numel() > 0:
                    unique_sizes = sorted({int(size) for size in sizes.tolist()})

        _t0 = time()
        loss, loss_terms = self.method.get_loss(
            batch=batch, model=self.model,
        )
        loss_terms["t_forward"] = time() - _t0

        self.optimizer.zero_grad(set_to_none=True)
        _t0 = time()
        loss.backward()
        loss_terms["t_backward"] = time() - _t0

        # Gradient-norm logging (always on) + optional clipping. clip_grad_norm_
        # returns the PRE-clip total norm: with max_norm=inf it computes the norm
        # without clipping (behavior unchanged when grad_clip_norm is null), and
        # with a finite value it clips and still returns the pre-clip norm — so a
        # single run both surfaces gradient spikes (training/grad_norm) and tames
        # them. No AMP/GradScaler is in use, so no unscale_ is needed.
        grad_clip = getattr(self.cfg.training, "grad_clip_norm", None)
        max_norm = float(grad_clip) if grad_clip is not None else float("inf")
        total_norm = th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        loss_terms["grad_norm"] = float(total_norm)

        _t0 = time()
        self.optimizer.step()
        if self.scheduler is not None:
            # Step LR scheduler once per global training step
            self.scheduler.step()
        loss_terms["t_optimizer"] = time() - _t0

        _t0_ema = time()
        for model in list(self.ema_models.values()):
            if model is not None:
                model.update(step=self.step)
        loss_terms["t_ema"] = time() - _t0_ema
        loss_terms["t_gpu_transfer"] = t_gpu_transfer
        # Optionally log current LR
        loss_terms["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return loss_terms

    def run_validation(self):
        print(f"Running validation at {self.step} steps.")
        if hasattr(self, 'logger'):
            self.logger.info(f"Running validation at step {self.step}")
        _t_val_start = time()

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

        self.logger.info("[validation step=%d] total=%.1fs", self.step, time() - _t_val_start)
        # Log results (test_results empty if metrics disabled & no improvements tracked)
        self.log({"validation": val_results, "test": test_results})

        # Strip Figure objects from results before pickling: PNGs are already
        # saved to eval_plots/ and their paths are stored in *_path keys. Keeping
        # live Figure objects in the pickle payload bloats artifacts and is
        # fragile across matplotlib versions.
        def _strip_figures(results_dict):
            for sub in results_dict.values():
                if isinstance(sub, dict):
                    sub.pop("examples", None)
                    sub.pop("examples_compare", None)

        # Dump results (persist even if metrics disabled to keep artifacts of generated graphs/plots)
        if self.cfg.training.save_checkpoint:
            val_dir = self.output_dir / "validation"
            val_dir.mkdir(exist_ok=True)
            _strip_figures(val_results)
            with open(val_dir / f"step_{self.step}.pkl", "wb") as f:
                pickle.dump(val_results, f)
            if test_results:
                test_dir = self.output_dir / "test"
                test_dir.mkdir(exist_ok=True)
                _strip_figures(test_results)
                with open(test_dir / f"step_{self.step}.pkl", "wb") as f:
                    pickle.dump(test_results, f)

        # Release matplotlib figure buffers. Figures created via plt.subplots()
        # are registered in matplotlib._pylab_helpers.Gcf and are NOT freed by
        # Python GC when their references drop — they must be closed explicitly.
        # Without this, canvas buffers accumulate ~10-20 MB per validation and
        # cause a linear RSS leak over long runs.
        plt.close('all')

    def _eval_embed_fn(self):
        """Euclidean-from-root TMD persistence-image embedding used for joint metrics."""
        tmd_bins = getattr(self.cfg.validation, "tmd_eval_bins", 16)
        filtration = getattr(self.cfg.validation, "tmd_eval_filtration", "radial_root")
        return lambda G: compute_tmd_embedding(G, filtration=filtration, n_bins=tmd_bins)

    def _gt_cache_for(self, eval_graphs: list[nx.Graph], uhat_np: np.ndarray) -> dict:
        """Build (once, then cache) the GT-fit objects for the distribution metrics."""
        key = id(eval_graphs)
        cache = self._eval_cache.get(key)
        if cache is None:
            cache = build_gt_cache(
                eval_graphs,
                uhat=tuple(np.asarray(uhat_np, dtype=float).reshape(3).tolist()),
                embed_fn=self._eval_embed_fn(),
                tmd_pca_ncomp=getattr(self.cfg.validation, "tmd_pca_ncomp", 32),
            )
            self._eval_cache[key] = cache
        return cache

    def _tf_batches_for(self, eval_graphs: list[nx.Graph]) -> list:
        """Build (once, then cache) GT reduction batches for teacher-forced validation.

        Mirrors the training data pipeline: same `cfg.reduction` factory and the same position
        scaling (1/pos_scale_factor), so each sample is what `get_loss` consumes in training.
        """
        from validation.teacher_forced_eval import build_reduction_batches_from_graphs
        key = id(eval_graphs)
        batches = self._tf_batch_cache.get(key)
        if batches is None:
            psf = float(self.pos_scale_factor) if self.pos_scale_factor is not None else 1.0
            tf_bs = int(getattr(self.cfg.validation, "tf_batch_size", 512))
            # Bound cost: the full ODE runs over every reduction level of every graph, so cap the
            # number of graphs (deterministic prefix; eval order is fixed). None -> use all.
            tf_max = getattr(self.cfg.validation, "tf_max_graphs", None)
            graphs = eval_graphs[: int(tf_max)] if tf_max else eval_graphs
            batches = build_reduction_batches_from_graphs(
                [G.copy() for G in graphs], self.cfg, tf_bs, psf
            )
            self._tf_batch_cache[key] = batches
        return batches

    def _floor_for(self, eval_graphs: list[nx.Graph], cache: dict, uhat_np: np.ndarray) -> dict:
        """Real-vs-real floor: a train subset (matched to N) vs the eval/GT set, cached once."""
        key = id(eval_graphs)
        floor = self._floor_cache.get(key)
        if floor is None:
            n = len(eval_graphs)
            train = self.train_graphs or []
            if not train:
                floor = {}
            else:
                rng = np.random.default_rng(0)
                if len(train) > n:
                    idx = rng.choice(len(train), size=n, replace=False)
                    train_sub = [train[i] for i in idx]
                else:
                    train_sub = list(train)
                floor = compute_distribution_metrics(
                    train_sub,
                    eval_graphs,
                    uhat=uhat_np,
                    gt_cache=cache,
                    embed_fn=self._eval_embed_fn(),
                    ged_enabled=False,
                    enable_ks=getattr(self.cfg.validation, "enable_ks", True),
                    enable_morphometrics=getattr(self.cfg.validation, "enable_morphometrics", True),
                    enable_light_joint=getattr(self.cfg.validation, "enable_light_joint", True),
                    dc_k=getattr(self.cfg.validation, "dc_nearest_k", 5),
                    tmd_pca_ncomp=getattr(self.cfg.validation, "tmd_pca_ncomp", 32),
                )
            self._floor_cache[key] = floor
        return floor

    @th.no_grad()
    def evaluate(self, eval_graphs: list[nx.Graph], beta):
        """Evaluate model for given beta on given graphs."""
        model = self.ema_models[beta]

        # Shuffle prediction order to make size distribution more uniform
        pred_perm = self.rng.permutation(np.arange(len(eval_graphs)))

        # Select target number of nodes and split into batches
        target_size = np.array([len(g) for g in eval_graphs])[pred_perm]

        # Extract num_root_children per graph (degree of root node)
        nrc_all = np.array([
            g.degree[g.graph["root"]] if "root" in g.graph else 2
            for g in eval_graphs
        ])[pred_perm]

        tmd_hidden_dim = getattr(model, "tmd_hidden_dim", 0)
        tmds = None
        if tmd_hidden_dim > 0:
            tmds = np.stack(
                [compute_tmd_mixed(g) for g in eval_graphs],
                axis=0,
            )[pred_perm]
        bs = (
            self.cfg.validation.batch_size
            if self.cfg.validation.batch_size is not None
            else self.cfg.training.batch_size
        )
        batches = [target_size[i : i + bs] for i in range(0, len(target_size), bs)]
        nrc_batches = [nrc_all[i : i + bs] for i in range(0, len(nrc_all), bs)]

        results = {}

        # Free-running generation is the dominant validation cost; for variants whose topology
        # head is unsupervised (predict_positions_only) the generated trees are meaningless
        # anyway. skip_free_running bypasses generation (and the dist metrics / plots / metric
        # loop that consume it), leaving only the teacher-forced metrics below.
        skip_free_running = bool(getattr(self.cfg.validation, "skip_free_running", False))

        # Generate graphs
        _t0_gen = time()
        pred_graphs = []
        if not skip_free_running:
            cursor = 0
            for batch, nrc_batch in zip(batches, nrc_batches):
                tmd_batch = None
                if tmds is not None:
                    tmd_batch = th.from_numpy(tmds[cursor : cursor + len(batch)]).to(self.device)
                pred_graphs_batch = self.method.sample_graphs(
                    target_size=th.tensor(batch, device=self.device),
                    model=model,
                    tmd=tmd_batch,
                    num_root_children=th.tensor(nrc_batch, device=self.device),
                )  # returns list[nx.Graph] with geometric node attrs
                pred_graphs += pred_graphs_batch
                cursor += len(batch)
            # Reorder back to original eval_graphs order
            inv_perm = np.empty_like(pred_perm)
            inv_perm[pred_perm] = np.arange(len(pred_perm))
            results["pred_graphs"] = [pred_graphs[i] for i in inv_perm]

            # Rescale positions back to original coordinate space
            if self.pos_scale_factor is not None:
                for G in results["pred_graphs"]:
                    for n in G.nodes():
                        G.nodes[n]['pos'] = G.nodes[n]['pos'] * self.pos_scale_factor

            if self.device == "cuda":
                th.cuda.empty_cache()
        else:
            results["pred_graphs"] = []
        _t_generation = time() - _t0_gen

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

        # Generated graphs are unrooted; the root is materialized first and always
        # lands at local index 0 (roots get the smallest global indices). Bifurcation
        # angles, TMD and tree-edit distance all need G.graph["root"].
        for G in results["pred_graphs"]:
            if "root" not in G.graph or G.graph.get("root") not in G.nodes:
                G.graph["root"] = 0 if G.number_of_nodes() > 0 else None

        # Model SO(2) symmetry axis: extents/plots are measured relative to it
        # (never hardcoded z). Shared by the dist metrics and the 3D plots below.
        uhat_np = (
            model.uhat.detach().cpu().numpy().reshape(-1)
            if getattr(model, "uhat", None) is not None
            else np.array([0.0, 0.0, 1.0])
        )

        # Distribution-level comparison of generated vs GT statistics (Wasserstein-1
        # per stat + avg tree-edit distance). Logged as floats -> wandb scalars.
        if getattr(self.cfg.validation, "enable_dist_metrics", True) and not skip_free_running:
            _t0_dist = time()
            gt_cache = self._gt_cache_for(eval_graphs, uhat_np)
            results["dist"] = compute_distribution_metrics(
                results["pred_graphs"],
                eval_graphs,
                uhat=uhat_np,
                ged_enabled=getattr(self.cfg.validation, "ged_enabled", True),
                ged_timeout=getattr(self.cfg.validation, "ged_timeout", 5.0),
                enable_ks=getattr(self.cfg.validation, "enable_ks", True),
                enable_morphometrics=getattr(self.cfg.validation, "enable_morphometrics", True),
                enable_light_joint=getattr(self.cfg.validation, "enable_light_joint", True),
                gt_cache=gt_cache,
                embed_fn=self._eval_embed_fn(),
                dc_k=getattr(self.cfg.validation, "dc_nearest_k", 5),
                tmd_pca_ncomp=getattr(self.cfg.validation, "tmd_pca_ncomp", 32),
            )
            # Real-vs-real floor as reference lines + a single headline excess used
            # for checkpoint selection (gen MMD above the achievable real-vs-real floor).
            if getattr(self.cfg.validation, "enable_floor", True):
                floor = self._floor_for(eval_graphs, gt_cache, uhat_np)
                if floor:
                    results["floor"] = floor
                    gen_mmd = results["dist"].get("mmd_morpho", float("nan"))
                    floor_mmd = floor.get("mmd_morpho", float("nan"))
                    if np.isfinite(gen_mmd) and np.isfinite(floor_mmd):
                        results["dist"]["headline_excess_mmd_morpho"] = float(gen_mmd - floor_mmd)
            self.logger.info(
                "[evaluate beta=%s] dist_metrics=%.1fs", beta, time() - _t0_dist
            )

        # Teacher-forced metrics: replay GT reduction sequences through the full ODE sampler and
        # compare to GT (W1/KS distribution distances + per-axis final-sample position MSE). This
        # is the direct "is geometry being learned given topology" signal; ~one ODE sample per
        # small val batch. Logged under validation/ema_{beta}/teacher_forced/*.
        if getattr(self.cfg.validation, "enable_teacher_forced", False):
            from validation.teacher_forced_eval import evaluate_teacher_forced
            _t0_tf = time()
            tf_batches = self._tf_batches_for(eval_graphs)
            was_training = model.training
            model.eval()  # dropout off for stable metrics
            try:
                results["teacher_forced"] = evaluate_teacher_forced(
                    self.method, model, tf_batches, uhat_np, device=self.device,
                    level_min=int(getattr(self.cfg.validation, "tf_level_min", 1)),
                )
            finally:
                if was_training:
                    model.train()
            self.logger.info(
                "[evaluate beta=%s] teacher_forced=%.1fs", beta, time() - _t0_tf
            )

        # Metric computation gated
        enable_metrics = getattr(self.cfg.validation, 'enable_metrics', True)
        _t0_metrics = time()
        if enable_metrics and not skip_free_running:
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
        _t_metrics = time() - _t0_metrics

        self.logger.info(
            "[evaluate beta=%s n_graphs=%d] generation=%.1fs metrics=%.1fs",
            beta, len(eval_graphs), _t_generation, _t_metrics,
        )

        # Example plots: 3D multi-azimuth views that orbit the model's uhat axis.
        # Replaces the old 2D XY-projection grids. Keys are unchanged so
        # _strip_figures (pickling) and the wandb.Image logging path still apply.
        enable_plots = getattr(self.cfg.validation, 'enable_plots', True)

        if enable_plots and len(results["pred_graphs"]) > 0:
            # Azimuths orbit the *true* symmetry axis (shared uhat_np computed above).
            uhat = uhat_np
            angles = getattr(self.cfg.validation, "plot_angles", None) or DEFAULT_ANGLES
            angles = [tuple(a) for a in angles]
            max_examples = min(8, len(results["pred_graphs"]))
            eval_plots_dir = self.output_dir / 'eval_plots'
            stem = f"step_{self.step}_beta_{beta}"

            gen_fig, gen_path = plot_graph_grid_angles(
                results["pred_graphs"][:max_examples],
                out_dir=eval_plots_dir,
                stem=stem,
                file_tag="gen3d",
                angles=angles,
                uhat=uhat,
                title_prefix="Gen",
                max_graphs=max_examples,
            )
            results["examples"] = gen_fig
            results["examples_path"] = str(gen_path)

            # GT references at the same angles for qualitative eyeballing
            # (the distribution metrics are the quantitative signal).
            ref_fig, ref_path = plot_graph_grid_angles(
                eval_graphs[:max_examples],
                out_dir=eval_plots_dir,
                stem=stem,
                file_tag="ref3d",
                angles=angles,
                uhat=uhat,
                title_prefix="GT",
                node_color="#1f77b4",
                max_graphs=max_examples,
            )
            results["examples_compare"] = ref_fig
            results["examples_compare_path"] = str(ref_path)
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
                    self.wandb_run.log({f"{prefix}{key}": value, "train_step": self.step})
            elif isinstance(value, Figure):
                # Wandb logging for figures currently disabled or wandb import commented out.
                # Keeping placeholder for future reactivation.
                if getattr(self.cfg.wandb, 'logging', False) and self.wandb_run is not None and wandb is not None:
                    try:
                        self.wandb_run.log({f"{prefix}{key}": wandb.Image(value), "train_step": self.step})
                    except Exception as e:
                        print(f"[wandb logging skipped] {e}")
