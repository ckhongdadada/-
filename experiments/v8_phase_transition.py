"""
V8 Phase Transition Experiment
==============================
Extends the V8 AL+SSL framework to run the phase transition experiments
designed in section 6.13 of the report:

  B1/B1-ssl : CIFAR-10  ρ scan (AL-only + AL+FixMatch)
  B2/B2-ssl : AG News   ρ scan (AL-only + AL+Dropout Consistency)
  B3/B3-ssl : Covertype ρ scan (AL-only + AL+VAT)
  B4/B4-ssl : Dry Bean ρ scan (AL-only + AL+VAT)

Covertype (C=7, ~581K samples, tabular) replaces Adult (C=2) because
binary classification has an inherent ceiling on AL strategy differences.
Dry Bean (C=7, ~13K samples, 16 features, tabular) provides a smaller
tabular alternative to Covertype for faster experimentation.

This script imports and extends the existing v8 framework WITHOUT modifying
any original code files.  Run with:

    python experiments/v8_phase_transition.py --phase [b1|b2|b3|all]

Or run individual ρ values directly:

    python experiments/v8_phase_transition.py \
        --dataset covertype --imbalance-ratio 9 --use-ssl --ssl-method vat
    python experiments/v8_phase_transition.py \
        --dataset drybean --imbalance-ratio 9 --use-ssl --ssl-method vat
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# 0.  Bootstrap: import the V8 framework without triggering its main()
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = str(PROJECT_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import the v8 module from its file path.  Using importlib avoids creating
# a duplicate module object (the file lives in the same directory, so a bare
# `import v8_controlled_fast_al_ssl` would shadow the canonical entry).
import importlib.util
_v8_path = Path(__file__).resolve().parent / "v8_controlled_fast_al_ssl.py"
_spec = importlib.util.spec_from_file_location("v8_controlled_fast_al_ssl", _v8_path)
v8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v8)

logger = logging.getLogger("phase_transition")

# Expose key v8 symbols
CFG = v8.CFG
Config = v8.Config
Dataset = v8.Dataset  # noqa: F811 — torch Dataset
TextMLPClassifier = v8.TextMLPClassifier
DataLoader = v8.DataLoader
Subset = v8.Subset
torch = v8.torch
nn = v8.nn
F = v8.F
np_v8 = v8.np

# ---------------------------------------------------------------------------
# 1.  Covertype dataset
# ---------------------------------------------------------------------------

COVERTYPE_CLASS_NAMES = [
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
    "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz",
]


class CovertypeDataset(Dataset):
    """Thin wrapper matching AdultDataset's interface (features, labels)."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_covertype(max_train: int = 50_000):
    """Load UCI Covertype via sklearn, cache to disk.

    Returns (train_set, test_set, infer_set, None).
    Labels are 0-indexed (original 1-7 → 0-6).
    """
    from sklearn.preprocessing import StandardScaler

    print("Loading Covertype (forest cover type, tabular, C=7)...")
    data_root = str(PROJECT_ROOT / "data" / "covertype")
    os.makedirs(data_root, exist_ok=True)

    cache_path = os.path.join(data_root, "covertype_cache.npz")
    if os.path.exists(cache_path):
        print("  Using cached Covertype dataset...")
        cache = np.load(cache_path)
        train_features = cache["train_features"]
        train_labels = cache["train_labels"]
        test_features = cache["test_features"]
        test_labels = cache["test_labels"]
    else:
        X, y = None, None

        # Strategy 1: sklearn fetch_covtype (direct download)
        try:
            from sklearn.datasets import fetch_covtype
            print("  Downloading Covertype via sklearn.fetch_covtype ...")
            covtype = fetch_covtype()
            X = covtype.data.astype(np.float32)
            y = covtype.target.astype(np.int64) - 1
        except Exception as e:
            print(f"  fetch_covtype failed ({e}), trying OpenML ...")

        # Strategy 2: OpenML fallback
        if X is None:
            try:
                from sklearn.datasets import fetch_openml
                print("  Fetching Covertype from OpenML ...")
                covtype = fetch_openml(name="covertype", version=3,
                                       as_frame=False, parser="auto")
                X = covtype.data.astype(np.float32)
                y = covtype.target.astype(np.int64) - 1
            except Exception as e2:
                print(f"  OpenML also failed ({e2}), generating synthetic data ...")

        # Strategy 3: synthetic fallback
        if X is None:
            n_total = 100_000
            n_features = 54
            rng = np.random.RandomState(42)
            X = rng.randn(n_total, n_features).astype(np.float32)
            # 7 classes with mild imbalance
            y = rng.choice(7, size=n_total, p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05])
            y = y.astype(np.int64)
            # Make features class-correlated so AL has something to learn
            for c in range(7):
                mask = y == c
                X[mask] += rng.randn(n_features) * 2.0

        # Shuffle and split
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        n_test = min(50_000, len(y) // 5)
        test_features, test_labels = X[:n_test], y[:n_test]
        train_features, train_labels = X[n_test:], y[n_test:]

        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features).astype(np.float32)
        test_features = scaler.transform(test_features).astype(np.float32)

        np.savez(cache_path,
                 train_features=train_features, train_labels=train_labels,
                 test_features=test_features, test_labels=test_labels)
        print(f"  Cached to {cache_path}")

    # Sub-sample training set to keep run-time reasonable
    if len(train_labels) > max_train:
        rng = np.random.RandomState(123)
        idx = rng.choice(len(train_labels), max_train, replace=False)
        train_features = train_features[idx]
        train_labels = train_labels[idx]

    train_set = CovertypeDataset(train_features, train_labels)
    test_set = CovertypeDataset(test_features, test_labels)
    infer_set = CovertypeDataset(train_features, train_labels)

    print(f"  Covertype: train={len(train_set)}, test={len(test_set)}, "
          f"classes={len(COVERTYPE_CLASS_NAMES)}, features={train_features.shape[1]}")
    return train_set, test_set, infer_set, None


# ---------------------------------------------------------------------------
# 1b. Dry Bean dataset
# ---------------------------------------------------------------------------

DRYBEAN_CLASS_NAMES = [
    "SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON"
]


class DryBeanDataset(Dataset):
    """Thin wrapper matching AdultDataset's interface."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_drybean():
    """Load UCI Dry Bean via sklearn/fetch_openml, cache to disk.

    Returns (train_set, test_set, infer_set, None).
    Labels are 0-indexed.
    """
    from sklearn.preprocessing import StandardScaler

    print("Loading Dry Bean (tabular, C=7)...")
    data_root = str(PROJECT_ROOT / "data" / "drybean")
    os.makedirs(data_root, exist_ok=True)

    cache_path = os.path.join(data_root, "drybean_cache.npz")
    if os.path.exists(cache_path):
        print("  Using cached Dry Bean dataset...")
        cache = np.load(cache_path)
        train_features = cache["train_features"]
        train_labels = cache["train_labels"]
        test_features = cache["test_features"]
        test_labels = cache["test_labels"]
    else:
        X, y = None, None

        # Strategy 1: sklearn fetch_openml
        try:
            from sklearn.datasets import fetch_openml
            print("  Fetching Dry Bean from OpenML ...")
            drybean = fetch_openml(data_id=602, as_frame=False, parser="auto")
            X = drybean.data.astype(np.float32)
            y = drybean.target.astype(np.int64)
            # Map string labels to integers if needed
            if y.dtype == object or y.dtype.kind in ('U', 'S', 'O'):
                unique_labels = np.unique(y)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                y = np.array([label_map[label] for label in y], dtype=np.int64)
        except Exception as e:
            print(f"  OpenML failed ({e}), generating synthetic data ...")

        # Strategy 2: synthetic fallback
        if X is None:
            n_total = 13_611
            n_features = 16
            rng = np.random.RandomState(42)
            X = rng.randn(n_total, n_features).astype(np.float32)
            y = rng.choice(7, size=n_total, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07])
            y = y.astype(np.int64)
            for c in range(7):
                mask = y == c
                X[mask] += rng.randn(n_features) * 1.5

        # Shuffle and split
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        n_test = min(2_000, len(y) // 5)
        test_features, test_labels = X[:n_test], y[:n_test]
        train_features, train_labels = X[n_test:], y[n_test:]

        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features).astype(np.float32)
        test_features = scaler.transform(test_features).astype(np.float32)

        np.savez(cache_path,
                 train_features=train_features, train_labels=train_labels,
                 test_features=test_features, test_labels=test_labels)
        print(f"  Cached to {cache_path}")

    train_set = DryBeanDataset(train_features, train_labels)
    test_set = DryBeanDataset(test_features, test_labels)
    infer_set = DryBeanDataset(train_features, train_labels)

    print(f"  Dry Bean: train={len(train_set)}, test={len(test_set)}, "
          f"classes={len(DRYBEAN_CLASS_NAMES)}, features={train_features.shape[1]}")
    return train_set, test_set, infer_set, None


# ---------------------------------------------------------------------------
# 2.  Patch V8 framework to support Covertype and Dry Bean
# ---------------------------------------------------------------------------

def patch_v8_for_covertype():
    """Add 'covertype' and 'drybean' to all v8 registry dicts/sets so existing functions
    work transparently.  Must be called once before any v8 function touches
    these datasets."""

    # 2a.  Number of classes
    _orig_get_num_classes = v8.get_num_classes

    def get_num_classes_patched(dataset_name):
        if dataset_name == "covertype":
            return 7
        if dataset_name == "drybean":
            return 7
        return _orig_get_num_classes(dataset_name)

    v8.get_num_classes = get_num_classes_patched

    # 2b.  SSL dataset membership — tabular datasets use pseudo-label style
    v8.PSEUDO_LABEL_SSL_DATASETS.add("covertype")
    v8.PSEUDO_LABEL_SSL_DATASETS.add("drybean")
    v8.SSL_SUPPORTED_DATASETS = v8.IMAGE_SSL_DATASETS | v8.PSEUDO_LABEL_SSL_DATASETS

    # 2c.  SSL method auto-resolution
    _orig_resolve = v8.resolve_ssl_method

    def resolve_ssl_patched(dataset_name, requested="auto"):
        if requested != "auto":
            return requested
        if dataset_name in ("covertype", "drybean"):
            return "vat"
        return _orig_resolve(dataset_name, requested)

    v8.resolve_ssl_method = resolve_ssl_patched

    # 2d.  Budget-level presets (use Adult-like tabular settings)
    for level_key, level_dict in v8.BUDGET_LEVELS.items():
        ref = level_dict.get("adult", level_dict.get("cifar10"))
        level_dict["covertype"] = dict(ref)
        level_dict["drybean"] = dict(ref)

    # 2e.  Fast 4060 preset — treat tabular datasets like adult
    _orig_fast = v8.apply_fast_4060_preset

    def fast_patched(explicit_bald=False, explicit_pool=False):
        _orig_fast(explicit_bald, explicit_pool)
        if CFG.dataset in ("covertype", "drybean"):
            CFG.batch_size_train = max(CFG.batch_size_train, 512)
            CFG.batch_size_infer = max(CFG.batch_size_infer, 1024)

    v8.apply_fast_4060_preset = fast_patched

    # 2f.  create_model — add covertype and drybean branches
    _orig_create_model = v8.create_model

    def create_model_patched(dataset, device, model_type=None):
        if dataset == "covertype":
            input_dim = getattr(CFG, "covertype_input_dim", 54)
            return TextMLPClassifier(
                input_dim=input_dim, num_classes=7,
                hidden_dim=256, dropout=0.2
            ).to(device)
        if dataset == "drybean":
            input_dim = getattr(CFG, "drybean_input_dim", 16)
            return TextMLPClassifier(
                input_dim=input_dim, num_classes=7,
                hidden_dim=128, dropout=0.2
            ).to(device)
        return _orig_create_model(dataset, device, model_type)

    v8.create_model = create_model_patched

    # 2g.  Modality dispatch — the v8 functions check
    #      `CFG.dataset in ["agnews", "adult"]` to select the tabular
    #      data branch.  We wrap train_model / evaluate /
    #      get_probs_and_features with a proxy that temporarily sets
    #      CFG.dataset = "adult" during execution.
    #
    #      The proxy also stores the *real* dataset name on CFG as
    #      `_real_dataset` so that get_num_classes (called inside
    #      train_model for pseudo-label stats) can return the correct
    #      class count even though CFG.dataset is temporarily "adult".

    # Re-patch get_num_classes to check _real_dataset first.
    _orig_get_num_classes_2 = v8.get_num_classes

    def get_num_classes_with_real(dataset_name):
        real = getattr(CFG, "_real_dataset", None)
        if real is not None:
            return _orig_get_num_classes_2(real)
        return _orig_get_num_classes_2(dataset_name)

    v8.get_num_classes = get_num_classes_with_real

    def _make_tabular_proxy(orig_fn):
        """Wrap *orig_fn* so that tabular datasets are treated
        as `"adult"` inside the function (same tabular branch)."""
        import functools

        @functools.wraps(orig_fn)
        def wrapper(*args, **kwargs):
            old = CFG.dataset
            if old in ("covertype", "drybean"):
                CFG._real_dataset = old
                CFG.dataset = "adult"
            try:
                return orig_fn(*args, **kwargs)
            finally:
                CFG.dataset = old
                CFG._real_dataset = None

        return wrapper

    v8.train_model = _make_tabular_proxy(v8.train_model)
    v8.evaluate = _make_tabular_proxy(v8.evaluate)
    v8.get_probs_and_features = _make_tabular_proxy(v8.get_probs_and_features)

    # Note: run_single_strategy itself does NOT need proxying.
    # It calls create_model(CFG.dataset, device) BEFORE entering the
    # training loop.  Since CFG.dataset is still "covertype" at that point,
    # create_model_patched correctly creates a 7-class model.
    # The proxied train_model/evaluate/get_probs_and_features then swap
    # CFG.dataset to "adult" only during their execution, so the tabular
    # branch fires correctly.  The model dimensions are already set.

    # 2h.  Dropout Consistency — needs a feature dimension for covertype
    #      The v8 implementation uses model._feature_dim; TextMLPClassifier
    #      already exposes this, so no patching needed.

    print("[patch] V8 framework patched for Covertype (C=7, tabular)")


# ---------------------------------------------------------------------------
# 3.  AG News long-tail support
# ---------------------------------------------------------------------------

def make_longtail_dataset(dataset, targets, imbalance_ratio, seed=42):
    """Apply exponential long-tail filtering, returning a Subset.

    Unlike CIFAR-10 where v8's built-in filter works, AG News needs
    a dedicated path because v8 explicitly skips `dataset == "agnews"`.
    """
    from ssl_v7_utils import make_longtail_indices
    indices = list(range(len(dataset)))
    filtered = make_longtail_indices(
        targets, imbalance_ratio=imbalance_ratio,
        distribution="exp", seed=seed)
    return [indices[i] for i in filtered]


def gather_targets(dataset):
    """Return integer label list for a dataset."""
    targets = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        targets.append(int(label))
    return targets


# ---------------------------------------------------------------------------
# 4.  Run one (dataset, rho, ssl) experiment
# ---------------------------------------------------------------------------

def run_one_experiment(dataset_name, imbalance_ratio, use_ssl, ssl_method,
                       strategies, seeds, output_dir):
    """Configure CFG, load data, apply imbalance, run all strategies."""

    # --- Configure CFG ---
    CFG.dataset = dataset_name
    CFG.budget_level = "low"
    CFG.imbalance_ratio = imbalance_ratio
    CFG.use_ssl = use_ssl
    CFG.ssl_method = ssl_method if ssl_method else "auto"
    CFG.strategies = strategies
    CFG.seeds = seeds
    CFG.output_dir = output_dir
    CFG.checkpoint_dir = os.path.join(output_dir, "checkpoints")
    CFG.resume = False  # always fresh for phase transition
    CFG.enable_tsne = False
    CFG.enable_cost_tracking = True

    # Apply budget-level preset
    if dataset_name in v8.BUDGET_LEVELS.get(CFG.budget_level, {}):
        preset = v8.BUDGET_LEVELS[CFG.budget_level][dataset_name]
        for k, v in preset.items():
            setattr(CFG, k, v)

    # Fast 4060 preset
    if CFG.fast_4060_preset:
        v8.apply_fast_4060_preset()

    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    # --- Logging ---
    log_file = os.path.join(CFG.output_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    logger.info(f"=== Phase Transition: {dataset_name} rho={imbalance_ratio} "
                f"SSL={use_ssl} ===")
    logger.info(f"Strategies: {strategies}, Seeds: {seeds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load dataset ---
    if dataset_name == "cifar10":
        train_set, test_set, infer_set, raw_set = v8.load_cifar10()
    elif dataset_name == "agnews":
        train_set, test_set, infer_set, raw_set = v8.load_agnews()
    elif dataset_name == "covertype":
        train_set, test_set, infer_set, raw_set = load_covertype()
        CFG.covertype_input_dim = train_set.features.shape[1]
    elif dataset_name == "drybean":
        train_set, test_set, infer_set, raw_set = load_drybean()
        CFG.drybean_input_dim = train_set.features.shape[1]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_idx_all = list(range(len(train_set)))
    test_idx = list(range(len(test_set)))

    # --- Apply long-tail imbalance ---
    if imbalance_ratio > 0:
        if dataset_name == "agnews":
            # v8 skips AG News; we handle it ourselves
            targets = gather_targets(train_set)
            train_idx_all = make_longtail_dataset(
                train_set, targets, imbalance_ratio, seed=42)
            logger.info(f"AG News imbalance: {len(targets)} -> "
                        f"{len(train_idx_all)} samples (rho={imbalance_ratio})")
        elif dataset_name == "covertype":
            targets = gather_targets(train_set)
            train_idx_all = make_longtail_dataset(
                train_set, targets, imbalance_ratio, seed=42)
            logger.info(f"Covertype imbalance: {len(targets)} -> "
                        f"{len(train_idx_all)} samples (rho={imbalance_ratio})")
        elif dataset_name == "drybean":
            targets = gather_targets(train_set)
            train_idx_all = make_longtail_dataset(
                train_set, targets, imbalance_ratio, seed=42)
            logger.info(f"Dry Bean imbalance: {len(targets)} -> "
                        f"{len(train_idx_all)} samples (rho={imbalance_ratio})")
        else:
            # CIFAR-10 uses v8's built-in path
            targets = gather_targets(train_set)
            from ssl_v7_utils import make_longtail_indices
            filtered = make_longtail_indices(
                targets, imbalance_ratio=imbalance_ratio,
                distribution="exp", seed=42)
            train_idx_all = [train_idx_all[i] for i in filtered]
            logger.info(f"Imbalance filter: {len(targets)} -> "
                        f"{len(train_idx_all)} samples (rho={imbalance_ratio})")

    # --- Run strategies ---
    all_results = []
    for strategy in strategies:
        logger.info(f"\n--- Strategy: {strategy} ---")
        for seed in seeds:
            try:
                result = v8.run_single_strategy(
                    strategy, train_set, test_set, infer_set, device,
                    train_idx_all, test_idx, seed, raw_dataset=raw_set)
                all_results.append(result)
            except Exception as e:
                logger.error(f"  ERROR [{strategy}] seed={seed}: {e}")
                import traceback
                traceback.print_exc()

    # --- Save results ---
    results_path = os.path.join(CFG.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Summary ---
    summary = {}
    for r in all_results:
        strat = r["strategy"]
        if strat not in summary:
            summary[strat] = {"f1_scores": [], "final_f1": []}
        if r["f1_scores"]:
            summary[strat]["final_f1"].append(r["f1_scores"][-1])

    logger.info("\n=== Summary ===")
    for strat, info in summary.items():
        if info["final_f1"]:
            mean_f1 = np.mean(info["final_f1"])
            std_f1 = np.std(info["final_f1"])
            logger.info(f"  {strat}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")

    # Save config snapshot
    config_snapshot = {k: v for k, v in CFG.__dict__.items()
                       if not k.startswith("_")}
    config_path = os.path.join(CFG.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)

    # Clean up handler to avoid duplicate logs
    logger.removeHandler(file_handler)

    return all_results


# ---------------------------------------------------------------------------
# 5.  Phase experiment definitions
# ---------------------------------------------------------------------------

PHASE_B1 = {
    "name": "B1: CIFAR-10 rho scan (AL-only)",
    "dataset": "cifar10",
    "ratios": [1, 2, 5, 10, 20, 50, 100],
    "use_ssl": False,
    "ssl_method": None,
    "output_prefix": "v8_cifar10_ir",
    "output_suffix": "_low_al",
}

PHASE_B1_SSL = {
    "name": "B1-ssl: CIFAR-10 rho scan (AL+FixMatch)",
    "dataset": "cifar10",
    "ratios": [1, 10, 100],
    "use_ssl": True,
    "ssl_method": "fixmatch",
    "output_prefix": "v8_cifar10_ir",
    "output_suffix": "_low_fixmatch",
}

PHASE_B2 = {
    "name": "B2: AG News rho scan (AL-only)",
    "dataset": "agnews",
    "ratios": [1, 3, 9, 27],
    "use_ssl": False,
    "ssl_method": None,
    "output_prefix": "v8_agnews_ir",
    "output_suffix": "_low_al",
}

PHASE_B2_SSL = {
    "name": "B2-ssl: AG News rho scan (AL+Dropout)",
    "dataset": "agnews",
    "ratios": [1, 9, 27],
    "use_ssl": True,
    "ssl_method": "dropout_consistency",
    "output_prefix": "v8_agnews_ir",
    "output_suffix": "_low_dropout",
}

PHASE_B3 = {
    "name": "B3: Covertype rho scan (AL-only)",
    "dataset": "covertype",
    "ratios": [1, 3, 9, 27],
    "use_ssl": False,
    "ssl_method": None,
    "output_prefix": "v8_covertype_ir",
    "output_suffix": "_low_al",
}

PHASE_B3_SSL = {
    "name": "B3-ssl: Covertype rho scan (AL+VAT)",
    "dataset": "covertype",
    "ratios": [1, 9, 27],
    "use_ssl": True,
    "ssl_method": "vat",
    "output_prefix": "v8_covertype_ir",
    "output_suffix": "_low_vat",
}

PHASE_B4 = {
    "name": "B4: Dry Bean rho scan (AL-only)",
    "dataset": "drybean",
    "ratios": [1, 3, 9, 27],
    "use_ssl": False,
    "ssl_method": None,
    "output_prefix": "v8_drybean_ir",
    "output_suffix": "_low_al",
}

PHASE_B4_SSL = {
    "name": "B4-ssl: Dry Bean rho scan (AL+VAT)",
    "dataset": "drybean",
    "ratios": [1, 9, 27],
    "use_ssl": True,
    "ssl_method": "vat",
    "output_prefix": "v8_drybean_ir",
    "output_suffix": "_low_vat",
}

PHASES = {
    "b1": [PHASE_B1, PHASE_B1_SSL],
    "b2": [PHASE_B2, PHASE_B2_SSL],
    "b3": [PHASE_B3, PHASE_B3_SSL],
    "b4": [PHASE_B4, PHASE_B4_SSL],
    "all": [PHASE_B1, PHASE_B1_SSL, PHASE_B2, PHASE_B2_SSL, PHASE_B3, PHASE_B3_SSL, PHASE_B4, PHASE_B4_SSL],
}


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V8 Phase Transition Experiment (section 6.13)")
    parser.add_argument(
        "--phase", type=str, default=None,
        choices=list(PHASES.keys()),
        help="Which phase to run: b1 (CIFAR-10), b2 (AG News), "
             "b3 (Covertype), b4 (Dry Bean), all (everything)")
    # Direct single-run mode (same args as v8)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar10", "agnews", "covertype", "drybean"])
    parser.add_argument("--imbalance-ratio", type=float, default=0.0)
    parser.add_argument("--use-ssl", dest="use_ssl", action="store_true",
                        default=False)
    parser.add_argument("--no-use-ssl", dest="use_ssl", action="store_false")
    parser.add_argument("--ssl-method", type=str, default=None)
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["random", "entropy", "margin"])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (1 seed, 2 rounds)")

    args = parser.parse_args()

    # --- Patch v8 for Covertype ---
    patch_v8_for_covertype()

    # Console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if args.phase is not None:
        # --- Phase mode: run all experiments in the phase ---
        phases = PHASES[args.phase]
        for phase in phases:
            logger.info(f"\n{'='*60}")
            logger.info(f"  {phase['name']}")
            logger.info(f"{'='*60}")

            for ratio in phase["ratios"]:
                out_dir = os.path.join(
                    str(PROJECT_ROOT / "output"),
                    f"{phase['output_prefix']}{ratio}{phase['output_suffix']}")
                seeds = args.seeds
                if args.quick:
                    seeds = [42]

                run_one_experiment(
                    dataset_name=phase["dataset"],
                    imbalance_ratio=ratio,
                    use_ssl=phase["use_ssl"],
                    ssl_method=phase["ssl_method"],
                    strategies=args.strategies,
                    seeds=seeds,
                    output_dir=out_dir,
                )

    elif args.dataset is not None:
        # --- Direct single-run mode ---
        out_dir = args.output_dir
        if out_dir is None:
            tag = "ssl" if args.use_ssl else "al"
            out_dir = os.path.join(
                str(PROJECT_ROOT / "output"),
                f"v8_{args.dataset}_ir{int(args.imbalance_ratio)}_low_{tag}")

        seeds = args.seeds
        if args.quick:
            seeds = [42]

        run_one_experiment(
            dataset_name=args.dataset,
            imbalance_ratio=args.imbalance_ratio,
            use_ssl=args.use_ssl,
            ssl_method=args.ssl_method,
            strategies=args.strategies,
            seeds=seeds,
            output_dir=out_dir,
        )

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run all phase B3 (Covertype rho scan)")
        print("  python experiments/v8_phase_transition.py --phase b3")
        print()
        print("  # Run all phase B4 (Dry Bean rho scan)")
        print("  python experiments/v8_phase_transition.py --phase b4")
        print()
        print("  # Run single experiment")
        print("  python experiments/v8_phase_transition.py \\")
        print("      --dataset covertype --imbalance-ratio 9 --use-ssl --ssl-method vat")
        print()
        print("  # Quick smoke test")
        print("  python experiments/v8_phase_transition.py --phase b4 --quick")


if __name__ == "__main__":
    main()
