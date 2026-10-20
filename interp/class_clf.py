"""Cell-class classifier on real morphology: the readout for E1 S-class (methodology §11b.2).

Features per tree: the per-tree scalars of ``validation.dist_metrics`` (size/extent, Strahler,
partition asymmetry, Sholl summaries), the morphometric vector (``MORPHO_KEYS``) and the
tall-arm statistics of ``interp.metrics``. Standardised multinomial logistic regression with
balanced class weights, fit on real train neurons.

``size_free=True`` drops the count/size features and divides the remaining lengths by the
tree's 3-D extent, so a steer that only makes neurons bigger cannot raise P(class) through
size alone.

Fit and evaluate::

    python -m interp.class_clf --config parity_neurons_uncond --out /.../class_clf
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from . import metrics as M

SIZE_KEYS = ("node_count", "leaf_count", "bifurcation_count", "sholl_peak", "sholl_auc", "n_nodes", "n_tips",
             "max_depth", "total_length", "basal_nodes", "apical_nodes")
LENGTH_KEYS = ("axial_extent", "radial_span", "mean_branch_length", "mean_radial_to_root", "apical_reach",
               "basal_reach_max")
LOG_KEYS = SIZE_KEYS + LENGTH_KEYS + ("total_extent", "strahler", "max_branch_order")
ARM_KEYS = ("n_arms", "n_tall", "dominance", "apical_reach", "basal_reach_max", "apical_nodes", "basal_nodes")


def tree_features(G, uhat) -> dict:
    from validation.dist_metrics import MORPHO_KEYS, _per_tree_scalars, assemble_morpho_vector

    f = dict(_per_tree_scalars(G, uhat))
    f.update(zip(MORPHO_KEYS, assemble_morpho_vector(G, uhat=uhat)))
    s = M.tree_row(G, uhat)
    f.update({k: s[k] for k in ARM_KEYS + ("n_nodes", "n_tips", "max_depth", "total_length")})
    return {k: float(v) for k, v in f.items()}


def feature_matrix(feats: list[dict], keys: list[str], size_free: bool) -> np.ndarray:
    X = np.array([[d.get(k, np.nan) for k in keys] for d in feats], dtype=np.float64)
    if size_free:
        ext = np.array([d.get("total_extent", np.nan) for d in feats])
        for j, k in enumerate(keys):
            if k in LENGTH_KEYS:
                X[:, j] = X[:, j] / np.where(ext > 0, ext, np.nan)
    else:
        for j, k in enumerate(keys):
            if k in LOG_KEYS:
                X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return X


class MorphoClassifier:
    def __init__(self, uhat, size_free: bool = False, C: float = 1.0):
        self.uhat, self.size_free, self.C = np.asarray(uhat, float), size_free, C

    def _keys(self, feats):
        keys = sorted(feats[0])
        if self.size_free:
            keys = [k for k in keys if k not in SIZE_KEYS and k != "total_extent"]
        return keys

    def fit_features(self, feats: list[dict], y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.keys = self._keys(feats)
        X = feature_matrix(feats, self.keys, self.size_free)
        self.fill = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, self.fill)
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000))
        self.model.fit(X, np.asarray(y, int))
        self.classes_ = self.model.classes_
        return self

    def proba_features(self, feats: list[dict]) -> np.ndarray:
        X = feature_matrix(feats, self.keys, self.size_free)
        X = np.where(np.isfinite(X), X, self.fill)
        return self.model.predict_proba(X)

    def features(self, graphs) -> list[dict]:
        return [tree_features(G, self.uhat) for G in graphs]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "MorphoClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


def main(argv=None):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    from .load import compose_cfg, load_split, uhat_of

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="parity_neurons_uncond")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = compose_cfg(a.config)
    uhat = uhat_of(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for sp in (a.fit_split, a.eval_split):
        gs = [g for g in load_split(cfg, sp, data_dir=a.data_dir) if g.graph.get("cell_class") is not None]
        data[sp] = ([tree_features(g, uhat) for g in gs], np.array([int(g.graph["cell_class"]) for g in gs]))
        print(f"[class_clf] {sp}: {len(gs)} neurons", flush=True)
    report = {}
    for sf in (False, True):
        clf = MorphoClassifier(uhat, size_free=sf).fit_features(*data[a.fit_split])
        fe, ye = data[a.eval_split]
        pred = clf.classes_[clf.proba_features(fe).argmax(1)]
        name = "size_free" if sf else "full"
        report[name] = dict(balanced_accuracy=float(balanced_accuracy_score(ye, pred)),
                            per_class_recall={int(c): float((pred[ye == c] == c).mean()) for c in np.unique(ye)},
                            confusion=confusion_matrix(ye, pred, labels=clf.classes_).tolist(),
                            classes=clf.classes_.tolist(), n_features=len(clf.keys))
        clf.save(out / f"class_clf_{name}.pkl")
        print(f"[class_clf] {name}: balanced accuracy {report[name]['balanced_accuracy']:.3f}", flush=True)
    (out / "class_clf_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
