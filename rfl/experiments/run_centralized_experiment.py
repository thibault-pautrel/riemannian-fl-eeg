"""
Centralized baseline: one SPDNet or EEGNet trained on the pooled data of all subjects.

The data are split once into train, validation and test sets with the same
(val_size, test_size, data_split_seed) as the federated runs, so the centralized
test set is the union of the federated client test sets.

Defaults: SPDNet architecture of the dataset (rfl/data/registry.py), cosine schedule from
--learning-rate to --learning-rate / 100, last epoch evaluated, no early stopping,
EEGNet without batch normalisation. --freeze-trunk trains the classifier head only
(random-feature control).

Reads:
    a MOABB dataset (downloaded on first use), or a synthetic SPD .npz written by
    rfl/data/synthetic.py.

Writes, in <save-root>/<experiment-name>/<run-name> or --save-dir:
    config.json, environment.json, checkpoints/, the evaluator outputs and
    centralized_results.json (metrics and training histories).

Usage (from any folder once the repository is installed, see README):
    python -m rfl.experiments.run_centralized_experiment --dataset BNCI2014_001 \
        --model spdnet --learning-rate 1e-2 --batch-size 64 --epochs 200 --seed 7
    python -m rfl.experiments.run_centralized_experiment --synthetic-clients synth.npz \
        --model spdnet --epochs 20 --device cpu

Requirements: MOABB for real EEG. A GPU is used when available.
"""

import json
import os
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from rfl.data.data_utils import load_moabb_dataset, create_splits, filter_binary_task, resolve_resample
from rfl.data.reduction import reduction_config
from rfl.data.datasets import build_dataset
from rfl.data.registry import dataset_default, get_sampling_rate, is_known_dataset
from rfl.models.eegnet import create_eegnet
from rfl.models.spdnet import create_spdnet
from rfl.training.train_loop import train_model
from rfl.training.evaluation import Evaluator
from rfl.training.checkpointing import CheckpointManager
from rfl.utils.config import ExperimentConfig
from rfl.utils.reproducibility import set_seed, log_environment


def prepare_centralized_data(
    dataset_name: str,
    subjects: List[int],
    config: ExperimentConfig = None
) -> Tuple[Dict, Dict]:
    """Load the trials of all subjects and split them into train, val and test.

    Args:
        dataset_name: MOABB dataset name, e.g. 'BNCI2014_001'.
        subjects: Subject IDs to include.
        config: Supplies fmin, fmax, resample, task, split_type, val_size, test_size
            and data_split_seed.

    Returns:
        (split_data, info). split_data maps 'train', 'val' and 'test' to (X, y)
        tuples of raw trials. info holds n_channels, n_timepoints, n_classes,
        class_names, sampling_rate, dataset_name and subjects.

    Raises:
        ValueError: if the dataset is not in data.registry.
    """
    print(f"\n{'='*70}")
    print(f"Loading centralized data: {dataset_name}")
    print(f"Subjects: {subjects}")
    print(f"{'='*70}\n")
    
    if not is_known_dataset(dataset_name):
        raise ValueError(f"Unknown dataset: {dataset_name}.")
    
    native_sfreq = get_sampling_rate(dataset_name)
    target_sfreq = resolve_resample(config.resample, native_sfreq)
    if target_sfreq != native_sfreq:
        print(f"Resampling: {native_sfreq} Hz (native) -> {target_sfreq} Hz")

    X, y, meta = load_moabb_dataset(
        dataset_name=dataset_name,
        subjects=subjects,
        resample=target_sfreq,
        fmin=config.fmin if config else 8.0,
        fmax=config.fmax if config else 32.0,
        verbose=True
    )
    
    print(f"\nLoaded data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of subjects: {len(subjects)}")
    
    task = config.task if config else 'all_classes'
    X, y, meta = filter_binary_task(
        X=X,
        y=y,
        meta=meta,
        task=task,
        dataset_name=dataset_name,
        verbose=True
    )
    
    # 'random' is a stratified split over the pooled trials (75/10/15 by default).
    split_type = config.split_type if config else 'random'
    val_size = config.val_size if config else 0.1
    test_size = config.test_size if config else 0.15
    data_split_seed = config.data_split_seed if config else 42
    
    print(f"\n{'='*70}")
    print("CENTRALIZED TRAINING SPLIT STRATEGY")
    print(f"{'='*70}")
    print(f"Split type: '{split_type}'")
    print(f"Data split seed: {data_split_seed}")
    print(f"Target proportions: Train ~{(1-test_size-val_size)*100:.0f}%, Val ~{val_size*100:.0f}%, Test ~{test_size*100:.0f}%")
    if split_type == 'random':
        print("Note: Using random stratified split for centralized training.")
        print("      This maximizes training data and provides standard ML split.")
    print(f"{'='*70}\n")
    
    split_data = create_splits(
        X=X,
        y=y,
        meta=meta,
        split_type=split_type,
        val_size=val_size,
        test_size=test_size,
        random_state=data_split_seed,
        verbose=True
    )
    if config is not None and config.train_trials_per_class:
        # Fewer calibration trials per subject (rfl/data/reduction.py), on the indices of
        # the same split as create_splits and the federated runs.
        from sklearn.model_selection import train_test_split
        from rfl.data.reduction import reduce_indices
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size,
                                  random_state=data_split_seed, stratify=y)
        tr, va = train_test_split(tr, test_size=val_size / (1 - test_size),
                                  random_state=data_split_seed, stratify=y[tr])
        subject_of = np.asarray(meta.reset_index(drop=True)['subject'])
        tr, va, red = reduce_indices(tr.tolist(), va.tolist(), y, subject_of,
                                     list(subjects), config)
        split_data = {'train': (X[tr], y[tr]), 'val': (X[va], y[va]),
                      'test': (X[np.sort(te)], y[np.sort(te)])}
        print(f"[Data] {len(red)} subject(s) reduced to {config.train_trials_per_class} "
              f"calibration trials per class: train={len(tr)} val={len(va)}")
    
    n_channels = X.shape[1]
    n_timepoints = X.shape[2]
    n_classes = len(np.unique(y))
    
    if 'event' in meta.columns:
        class_names = sorted(meta['event'].unique())
    else:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    sampling_rate = target_sfreq
    
    info = {
        'n_channels': n_channels,
        'n_timepoints': n_timepoints,
        'n_classes': n_classes,
        'class_names': class_names,
        'sampling_rate': sampling_rate,
        'dataset_name': dataset_name,
        'subjects': subjects
    }
    
    print(f"\n{'='*70}")
    print("Combined Dataset Statistics:")
    print(f"{'='*70}")
    print(f"Train: {split_data['train'][0].shape[0]} samples")
    print(f"Val:   {split_data['val'][0].shape[0]} samples")
    print(f"Test:  {split_data['test'][0].shape[0]} samples")
    print(f"Total: {sum([d[0].shape[0] for d in split_data.values()])} samples")
    print(f"Channels: {n_channels}, Timepoints: {n_timepoints}")
    print(f"Classes: {n_classes}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"{'='*70}\n")
    
    return split_data, info


def _evaluate(model, loader, criterion, device):
    """(mean loss, accuracy) of `model` on `loader`, in evaluation mode."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += float(criterion(out, y)) * len(y)
            correct += int((out.argmax(1) == y).sum())
            total += len(y)
    return (loss_sum / total, correct / total) if total else (float("nan"), float("nan"))


def run_centralized_experiment(
    config: ExperimentConfig,
    dataset_name: str,
    subjects: List[int],
    save_dir: str,
    synthetic_dataset=None,
) -> Dict[str, Any]:
    """Train, select and test one centralized model, and save its artefacts.

    Args:
        config: Experiment configuration. Its subjects and dataset fields are
            overwritten with the arguments below before it is saved.
        dataset_name: MOABB dataset name, or 'synthetic'.
        subjects: Subject IDs (ignored for synthetic data).
        save_dir: Output directory, created if needed.
        synthetic_dataset: Path to a synthetic .npz, or an already loaded dataset
            dict. When given, all synthetic clients are pooled and MOABB is not used.

    Returns:
        The results dict, also written to <save_dir>/centralized_results.json.
    """
    os.makedirs(save_dir, exist_ok=True)
    set_seed(config.seed, deterministic=getattr(config, 'deterministic', True))
    log_environment(save_dir)
    
    config.subjects = subjects
    config.dataset = dataset_name
    config.save(save_dir)
    
    if synthetic_dataset is not None:
        from rfl.data.synthetic import load_dataset as _load_syn, pool_clients
        _ds = _load_syn(synthetic_dataset) if isinstance(synthetic_dataset, str) else synthetic_dataset
        from rfl.data.reduction import reduce_synthetic
        _ds = dict(_ds)
        _ds['clients'], _ = reduce_synthetic(list(_ds['clients']), config)
        split_data = pool_clients(_ds)
        _m = _ds['metadata']
        info = {
            'n_channels': _m['n_channels'],
            'n_timepoints': _m['n_channels'],
            'n_classes': _m['n_classes'],
            'class_names': [f'class_{k}' for k in range(_m['n_classes'])],
            'sampling_rate': None,
            'dataset_name': 'synthetic',
            'subjects': [],
        }
    else:
        split_data, info = prepare_centralized_data(
            dataset_name=dataset_name,
            subjects=subjects,
            config=config
        )
    
    # build_dataset turns the arrays into 'raw', 'covariance' or 'spd' datasets.
    is_spdnet = config.model_type.lower() == 'spdnet'
    kind = config.data_type.lower()
    ds_kwargs = dict(
        normalization_type=config.normalization_type,
        estimator=config.cov_estimator,
        shrinkage=config.cov_shrinkage,
        regularization=config.cov_regularization,
    )
    train_dataset = build_dataset(split_data['train'][0], split_data['train'][1], kind, **ds_kwargs)
    val_dataset   = build_dataset(split_data['val'][0],   split_data['val'][1],   kind, **ds_kwargs)
    test_dataset  = build_dataset(split_data['test'][0],  split_data['test'][1],  kind, **ds_kwargs)
    
    # With a batch norm (SPD or EEGNet), a last batch of a single record is dropped: it
    # has no batch statistics (SPD) or noisy ones (BatchNorm), as in the federated runs.
    single_last = (config.has_batch_norm()
                   and len(train_dataset) % config.batch_size == 1)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=single_last,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available()
    )
    
    if is_spdnet:
        spdnet_params = config.get_model_params()
        model = create_spdnet(
            n_classes=info['n_classes'],
            n_channels=info['n_channels'],
            seed=config.seed,
            **spdnet_params
        )
        chain = " -> ".join(str(d) for d in [info['n_channels']] + model.hidden_dims)
        print(f"[SPDNet] {model.depth} BiMap block(s): {chain}")
    else:
        model = create_eegnet(
            n_classes=info['n_classes'],
            n_channels=info['n_channels'],
            n_timepoints=info['n_timepoints'],
            sampling_rate=info['sampling_rate'],
            **config.get_model_params()
        )
    
    if bool(config.freeze_trunk):
        # Random-feature control: only the classifier head is trained.
        from rfl.federated.param_groups import freeze_trunk
        n_frozen = freeze_trunk(model)
        print(f"[freeze-trunk] {n_frozen} parameter tensors kept at initialisation")

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    device = config.torch_device()
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_name=config.run_name,
        config=config
    )
    
    evaluator = Evaluator(
        num_classes=info['n_classes'],
        class_names=info['class_names'],
        save_dir=save_dir
    )
    
    print(f"\n{'='*70}")
    print("Starting Centralized Training")
    print(f"{'='*70}\n")
    
    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device,
        checkpoint_manager=checkpoint_manager,
        evaluator=evaluator,
        is_spdnet=is_spdnet,
        test_loader=test_loader,
    )
    
    # The test split is used once, here. With report_rule='selected' the checkpoint
    # with the best selection metric is reloaded first.
    print(f"\n{'='*70}")
    print("Final Test Evaluation")
    print(f"{'='*70}\n")

    if str(config.report_rule) == 'last':
        print(f"Evaluating the LAST epoch ({training_results['final_epoch']})")
    else:
        print(f"Reloading the checkpoint selected on "
              f"{training_results.get('selection_metric', 'val_loss')} "
              f"(epoch {training_results.get('best_epoch')})")
        checkpoint_manager.load_best(model, device=device)
    val_before_recalibration = None
    if config.spdnet_bn_recalibrate:
        # Exact running statistics of the batch norms (rfl/models/bn_recalibration.py)
        # of the reported model, then its validation metrics. The reported validation
        # field (final_* with report 'last', best_* with 'selected') takes the new values.
        from rfl.models.bn_recalibration import recalibrate
        loss0, acc0 = _evaluate(model, val_loader, criterion, device)
        recalibrate(model, train_loader, device)
        loss, acc = _evaluate(model, val_loader, criterion, device)
        val_before_recalibration = {'val_acc': acc0, 'val_loss': loss0}
        prefix = 'final' if str(config.report_rule) == 'last' else 'best'
        training_results[f'{prefix}_val_acc'] = acc
        training_results[f'{prefix}_val_loss'] = loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_recalibrated.pt'))
        print(f"[BN] running statistics recalibrated on the training data: val acc "
              f"{acc0:.4f} -> {acc:.4f} (model saved to model_recalibrated.pt)")
    test_results = evaluator.evaluate_and_save(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        split_name='test',
        run_name=config.run_name
    )
    
    final_results = {
        'config': config.to_dict(),
        'dataset': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
        },
        'model': {
            'type': config.model_type,
            'n_parameters': sum(p.numel() for p in model.parameters()),
        },
        # The best_val_* values are all taken at the selected epoch.
        'training': {
            'selection_metric': training_results.get('selection_metric', 'val_loss'),
            'best_val_acc': training_results['best_val_acc'],
            'best_val_loss': training_results['best_val_loss'],
            'best_val_f1': training_results.get('best_val_f1'),
            'best_epoch': training_results.get('best_epoch'),
            'final_epoch': training_results['final_epoch'],
            'final_val_loss': training_results.get('final_val_loss'),
            'final_val_acc': training_results.get('final_val_acc'),
            'lr_schedule': training_results.get('lr_schedule'),
            'report_rule': training_results.get('report_rule'),
            'train_loss_history': training_results['history']['train_loss'],
            'test_acc_history': training_results['history'].get('test_acc'),
            'test_loss_history': training_results['history'].get('test_loss'),
            'lr_history': training_results['history']['learning_rate'],
            'epochs_budget': config.epochs,
            'stopped_early': training_results.get('stopped_early'),
            # True when the epoch budget ran out before early stopping fired.
            'hit_budget': bool(training_results['final_epoch'] >= config.epochs
                               and not training_results.get('stopped_early')),
            'val_acc_history': training_results['history']['val_acc'],
            'val_loss_history': training_results['history']['val_loss'],
            'bn_recalibrated': bool(config.spdnet_bn_recalibrate),
            'before_recalibration': val_before_recalibration,
        },
        'test_metrics': {
            'accuracy': test_results['accuracy'],
            'f1_macro': test_results['f1_macro'],
            'precision_macro': test_results['precision_macro'],
            'recall_macro': test_results['recall_macro'],
            'loss': test_results['loss'],
        },
        'per_class_metrics': {
            'f1': test_results['per_class_f1'],
            'precision': test_results['per_class_precision'],
            'recall': test_results['per_class_recall'],
            'support': test_results['per_class_support'],
        }
    }
    
    results_path = os.path.join(save_dir, 'centralized_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Centralized Experiment Complete!")
    print(f"{'='*70}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_results['f1_macro']:.4f}")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*70}\n")
    
    return final_results


def _build_arg_parser():
    import argparse
    from rfl.experiments.cli_checks import HelpFormatter
    from rfl.data.registry import DATASET_CONFIGS
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run a centralized SPDNet/EEGNet experiment (real EEG or synthetic SPD).",
        formatter_class=HelpFormatter)

    # Data source: exactly one of --dataset and --synthetic.
    p.add_argument("--dataset", type=str, default=None, choices=sorted(DATASET_CONFIGS),
                   help="MOABB dataset (real EEG).")
    p.add_argument("--synthetic-clients", "--synthetic", dest="synthetic", type=str,
                   default=None,
                   help="Path to a synthetic SPD .npz (from rfl/data/synthetic.py). "
                        "Mutually exclusive with --dataset.")
    p.add_argument("--subjects", type=int, nargs="+", default=None,
                   help="Subject IDs (real EEG). Default: all subjects of the dataset.")
    p.add_argument("--task", type=str, default="all_classes",
                   choices=["binary", "all_classes", "no_rest"], help="Classes kept.")
    p.add_argument("--resample", type=float, default=128.0,
                   help="Target sampling rate in Hz. 0 keeps the native rate. Never "
                        "upsamples. For EEGNet it sets the input length and the temporal "
                        "kernel.")
    p.add_argument("--fmin", type=float, default=8.0, help="Band-pass low cutoff (Hz).")
    p.add_argument("--fmax", type=float, default=32.0, help="Band-pass high cutoff (Hz).")
    p.add_argument("--split-type", type=str, default="random", choices=["random"],
                   help="Split of the trials: stratified random, the split of the "
                        "federated runs. Other strategies of rfl/data/data_utils.py "
                        "(session, subject) have no federated counterpart and are not "
                        "exposed.")

    # Model
    p.add_argument("--model", type=str, default="spdnet", choices=["spdnet", "eegnet"])
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="SPDNet BiMap output dim (<= n_channels). Default: the dataset "
                        "value of rfl/data/registry.py, n_channels // 2 otherwise.")
    p.add_argument("--epsilon", type=float, default=None,
                   help="SPDNet ReEig floor. Default: the dataset value of "
                        "rfl/data/registry.py, 1e-4 otherwise.")
    p.add_argument("--hidden-dims", type=str, default="",
                   help="Comma-separated BiMap output dimensions of a deep SPDNet, e.g. "
                        "'16,8'. Overrides --hidden-dim.")
    p.add_argument("--epsilons", type=str, default="",
                   help="One ReEig floor per block, strictly increasing. Required with "
                        "--hidden-dims.")
    p.add_argument("--reeig-percentile", type=float, default=None,
                   help="Set the ReEig floors to this percentile of the spectrum each "
                        "ReEig acts on, from the floors file of analyze_covariances "
                        "--architectures (needs --hidden-dims).")
    p.add_argument("--floors-file", default=None,
                   help="Floors file of --reeig-percentile (default "
                        "$RFL_RESULTS/reeig_floors/<dataset>.json).")
    p.add_argument("--stiefel", default="projection",
                   choices=["projection", "trivialization"],
                   help="BiMap weights: optimiser step then polar projection, or the "
                        "library's orthogonal parametrization.")
    _add_spd_bn_args(p)
    tn = p.add_mutually_exclusive_group()
    tn.add_argument("--trace-normalize", dest="trace_normalize", action="store_true")
    tn.add_argument("--no-trace-normalize", dest="trace_normalize", action="store_false")
    p.set_defaults(trace_normalize=True)

    # Covariance estimation (real EEG, SPDNet). --cov-shrinkage is only used by 'lwf'.
    p.add_argument("--cov-estimator", type=str, default="scm", choices=["scm", "lwf"])
    p.add_argument("--cov-shrinkage", type=float, default=0.0)
    p.add_argument("--cov-regularization", type=float, default=0.0)
    from rfl.data.reduction import add_reduction_args
    add_reduction_args(p)

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eegnet-norm-type", type=str, default="none",
                   choices=["none", "batch"],
                   help="EEGNet normalisation: 'none' (no normalisation layer) or "
                        "'batch' (original architecture). Ignored for spdnet.")
    p.add_argument("--eegnet-input-norm", default="per_trial",
                   choices=["per_trial", "per_trial_global"],
                   help="EEGNet input standardisation, per trial: each channel on its own "
                        "(the papers), or all channels together, which keeps the relative "
                        "channel powers (the motor-imagery feature SPDNet sees).")
    p.add_argument("--freeze-trunk", action="store_true",
                   help="Keep every parameter except the classifier head at its random "
                        "initialisation.")
    p.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    p.add_argument("--learning-rate", type=float, default=1e-3,
                   help="Adam learning rate (start of the schedule).")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"],
                   help="Optimiser of the training steps.")
    p.add_argument("--momentum", type=float, default=0.0,
                   help="Heavy-ball momentum of --optimizer sgd (0: plain SGD).")
    p.add_argument("--early-stopping-patience", "--early-stop-patience",
                   dest="early_stopping_patience", type=int, default=0,
                   help="Epochs without a strict improvement of the selection metric "
                        "before training stops. 0 runs the full budget.")
    p.add_argument("--selection-metric", type=str, default="val_acc",
                   choices=["val_loss", "val_acc"],
                   help="Validation metric that drives the plateau schedule, early "
                        "stopping and the checkpoint evaluated on the test set.")
    p.add_argument("--lr-plateau-patience", type=int, default=20,
                   help="Epochs without a strict improvement before the rate is "
                        "multiplied by --lr-plateau-factor.")
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-min", type=float, default=None,
                   help="Lower bound of the learning rate for the plateau and cosine "
                        "schedules. Default: --learning-rate / 100.")
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["cosine", "plateau", "step", "none"],
                   help="'cosine' decays from --learning-rate to --lr-min over --epochs.")
    p.add_argument("--report", type=str, default="last",
                   choices=["selected", "last"],
                   help="Model evaluated on the test set: the last epoch, or the one "
                        "selected on --selection-metric. 'last' requires "
                        "--early-stopping-patience 0.")
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.15)

    # Misc and output
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   help="cpu, cuda, cuda:N, or auto (cuda when available, else cpu)")
    p.add_argument("--experiment-name", type=str, default="centralized")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-root", type=str, default="./results")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Output directory. Default: <save-root>/<experiment-name>/<run-name>.")
    p.add_argument("--print-config", action="store_true",
                   help="Print the complete configuration of this command (JSON), with "
                        "every default resolved, and exit without running.")
    return p


def _add_spd_bn_args(p) -> None:
    """SPD batch-norm options, shared by both runners (rfl/models/spdnet.py)."""
    from rfl.models.spdnet import (BN_DEFAULTS, BN_MEAN_TYPES, BN_NORM_AND_BIAS_MODES,
                               BN_POSITIONS, BN_TYPES)
    p.add_argument("--use-batch-norm", action="store_true",
                   help="SPDNet: one SPD batch norm per block (yetanotherspdnet). Kept "
                        "on each client in federated runs (FedBN). Refused with "
                        "--dp-enabled.")
    p.add_argument("--bn-type", default=BN_DEFAULTS["bn_type"], choices=BN_TYPES,
                   help="'mean_only' (BatchNormSPDMean) or 'mean_var_scalar' "
                        "(BatchNormSPDMeanScalarVariance).")
    p.add_argument("--bn-mean-type", default=BN_DEFAULTS["bn_mean_type"],
                   choices=BN_MEAN_TYPES, help="Geometry of the batch mean.")
    p.add_argument("--bn-momentum", type=float, default=BN_DEFAULTS["bn_momentum"],
                   help="Update rate of the running statistics used at evaluation: the "
                        "running mean of the SPD batch norm, or the running mean and "
                        "variance of the EEGNet BatchNorm (--eegnet-norm-type batch).")
    p.add_argument("--bn-norm-and-bias-mode", default=BN_DEFAULTS["bn_norm_and_bias_mode"],
                   choices=BN_NORM_AND_BIAS_MODES,
                   help="Centring and bias by whitening or by parallel transport "
                        "(mean_only only).")
    p.add_argument("--bn-position", default=BN_DEFAULTS["bn_position"],
                   choices=BN_POSITIONS,
                   help="After the ReEig of each block (library SPDnet) or before it "
                        "(SPDNetBN). 'before_reeig' needs recalibrated ReEig floors.")
    p.add_argument("--bn-recalibrate", action="store_true",
                   help="After training, set the running statistics of every batch norm "
                        "(SPD batch norm, EEGNet BatchNorm) to the exact statistics of its "
                        "inputs on the training data, instead of the momentum average. "
                        "Shared batch norm: the clients' statistics are combined as in the "
                        "aggregation. See rfl/models/bn_recalibration.py.")


def _spd_bn_config(args) -> dict:
    """The SPD batch-norm fields of the configuration, from parsed arguments."""
    return dict(spdnet_bn_type=args.bn_type, spdnet_bn_mean_type=args.bn_mean_type,
                spdnet_bn_momentum=args.bn_momentum,
                spdnet_bn_norm_and_bias_mode=args.bn_norm_and_bias_mode,
                spdnet_bn_position=args.bn_position,
                spdnet_bn_recalibrate=args.bn_recalibrate)


def apply_defaults(args, src: str):
    """Fill the options left unset.

    SPDNet BiMap dimension and ReEig floor of dataset `src` (rfl/data/registry.py), and
    lr_min = learning_rate / 100. Synthetic data falls back to n_channels // 2 and 1e-4.
    A real dataset without calibrated values gets the same fallback, with a warning.
    """
    if args.model == "spdnet":
        warn_uncalibrated(src, args.hidden_dim, args.epsilon,
                          getattr(args, "hidden_dims", ""))
    if args.hidden_dim is None and args.model == "spdnet":
        args.hidden_dim = dataset_default(src, "spdnet_hidden_dim")
    if args.epsilon is None:
        args.epsilon = (dataset_default(src, "spdnet_reeig", 1e-4)
                        if args.model == "spdnet" else 1e-4)
    if args.lr_min is None:
        args.lr_min = args.learning_rate / 100.0
    return args


def warn_uncalibrated(src: str, hidden_dim, epsilon, hidden_dims="") -> None:
    """Warn when SPDNet falls back to generic values on a real dataset."""
    import warnings
    if src == "synthetic" or str(hidden_dims or "").strip():
        return
    missing = [k for k, v in (("spdnet_hidden_dim", hidden_dim), ("spdnet_reeig", epsilon))
               if v is None and dataset_default(src, k) is None]
    if missing:
        warnings.warn(
            f"{src}: no calibrated {', '.join(missing)} in rfl/data/registry.py "
            "(DATASET_DEFAULTS). Generic values are used (hidden dim n_channels // 2, "
            "ReEig floor 1e-4). Calibrate them with `python -m "
            f"rfl.experiments.analyze_covariances --datasets {src}` and add them to "
            "DATASET_DEFAULTS, or pass --hidden-dim and --epsilon.", stacklevel=2)


def prepare_args(argv=None):
    """Parse and complete a command line: (parser, args, data source).

    Raises:
        SystemExit: on an invalid command line (argparse error).
    """
    p = _build_arg_parser()
    args = p.parse_args(argv)
    if (args.dataset is None) == (args.synthetic is None):
        p.error("provide exactly one of --dataset or --synthetic")
    is_syn = args.synthetic is not None
    if is_syn and args.model == "eegnet":
        p.error("the synthetic data are SPD matrices, which only SPDNet reads: use "
                "--model spdnet (EEGNet needs raw EEG, --dataset)")
    if not is_syn and args.subjects is None:
        from rfl.data.registry import get_subjects
        args.subjects = get_subjects(args.dataset)
    from rfl.experiments.cli_checks import explicit_dests, inert_option_errors
    errors = inert_option_errors(p, args, explicit_dests(p, argv))
    if errors:
        p.error("options without effect in this run:\n  " + "\n  ".join(errors))
    src = "synthetic" if is_syn else args.dataset
    from rfl.experiments.cli_checks import resolve_reeig_percentile
    resolve_reeig_percentile(p, args, src)
    apply_defaults(args, src)
    from rfl.experiments.cli_checks import dataset_errors
    errors, warns = dataset_errors(args, src)
    for w in warns:
        warnings.warn(w, stacklevel=2)
    if errors:
        p.error("\n  ".join(["run impossible on this dataset:", *errors]))
    return p, args, src


def config_from_args(args, src: str) -> ExperimentConfig:
    """The ExperimentConfig of a completed command line (see `prepare_args`)."""
    is_syn = src == "synthetic"
    # EEGNet reads raw trials. SPDNet reads covariances (real EEG) or SPD matrices.
    data_type = "raw" if args.model == "eegnet" else ("spd" if is_syn else "covariance")
    config = ExperimentConfig(
        experiment_name=args.experiment_name, run_name=args.run_name or "run",
        dataset=src, subjects=[] if is_syn else args.subjects,
        data_path=os.path.abspath(args.synthetic) if is_syn else "",
        data_type=data_type, task=args.task, split_type=args.split_type,
        cov_estimator=args.cov_estimator, cov_shrinkage=args.cov_shrinkage,
        cov_regularization=args.cov_regularization,
        model_type=args.model, spdnet_hidden_dim=args.hidden_dim,
        eegnet_norm_type=args.eegnet_norm_type,
        normalization_type=args.eegnet_input_norm,
        spdnet_use_batch_norm=args.use_batch_norm, spdnet_epsilon=args.epsilon,
        **_spd_bn_config(args),
        spdnet_trace_normalize=args.trace_normalize,
        spdnet_hidden_dims=args.hidden_dims, spdnet_epsilons=args.epsilons,
        freeze_trunk=args.freeze_trunk,
        batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, early_stopping_patience=args.early_stopping_patience,
        optimizer=args.optimizer, sgd_momentum=args.momentum, spdnet_stiefel=args.stiefel,
        spdnet_reeig_percentile=args.reeig_percentile,
        selection_metric=args.selection_metric,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor, lr_min=args.lr_min,
        lr_schedule=args.lr_schedule, report_rule=args.report,
        val_size=args.val_size, test_size=args.test_size, resample=args.resample,
        **reduction_config(args),
        fmin=args.fmin, fmax=args.fmax,
        seed=args.seed, data_split_seed=args.seed, device=args.device,
        save_root=args.save_root,
    )
    if args.run_name is None:
        # The hash covers every setting except the seed and the output location, so two
        # commands that differ in any option never share a directory (or a checkpoint).
        from rfl.analysis.results import config_id
        config.run_name = (f"{args.model}_{src}{'_rnd' if args.freeze_trunk else ''}"
                           f"_seed{args.seed}_{config_id(config.to_dict())}")
        config.update()
    return config


def check_command(argv) -> None:
    """Build the configuration of a command line without running it.

    Raises:
        SystemExit: on an invalid command line.
        ValueError: on an invalid configuration.
    """
    _, args, src = prepare_args(argv)
    try:
        config_from_args(args, src)
    except ValueError as e:
        from rfl.experiments.cli_checks import with_flags
        raise ValueError(with_flags(str(e))) from None


def main(argv=None):
    from rfl.utils.logs import setup_logging
    setup_logging()
    p, args, src = prepare_args(argv)
    try:
        config = config_from_args(args, src)
    except ValueError as e:
        from rfl.experiments.cli_checks import with_flags
        p.error(with_flags(str(e)))
    if args.print_config:
        from rfl.experiments.run_federated_experiment import print_config
        print_config(config)
        return
    save_dir = args.save_dir or os.path.join(args.save_root, args.experiment_name,
                                             config.run_name)
    from rfl.training.divergence import DIVERGENCE_ERRORS, write_diverged
    try:
        run_centralized_experiment(config, src, config.subjects, save_dir,
                                   synthetic_dataset=args.synthetic)
    except DIVERGENCE_ERRORS as e:            # a result (rfl/training/divergence.py)
        write_diverged(save_dir, config, "centralized_results.json", e)


if __name__ == "__main__":
    main()