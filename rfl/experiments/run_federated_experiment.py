"""
Federated SPDNet or EEGNet experiment, with optional personalisation and optional
record-level differential privacy. The two options are independent.

Clients are groups of subjects of a MOABB dataset (--dataset), or the clients stored in
a synthetic SPD file (--synthetic-clients). Each round broadcasts the global model,
runs local training on every participating client, aggregates the uploads with
fedavg, projavg, rlavg or retavg (rfl/federated/aggregation.py) and evaluates the new
global model on every client.

Personalisation (rfl/federated/client.py, rfl/federated/personalization.py):
    --local-fc              personal classifier head (FedPer), the trunk is aggregated
    --local-head-mode       'train' (head trained during the rounds) or 'frozen' (head
                            pinned at its initialisation, then re-fitted)
    --head-refit-epochs     re-fit the head on each client after the last round
    --local-only            no broadcast, no aggregation (baseline)
    --freeze-trunk          random-feature control, only the head trains
Normalisation layers (EEGNet --eegnet-norm-type batch, SPDNet --use-batch-norm) stay
local by default (FedBN). --no-fedbn shares them.

Privacy (--dp-enabled, rfl/federated/privacy/). Each local step is the privatised step of
DP-RFedProj (rfl/federated/privacy/dp_grad.py)

    g~ = (1/B) sum_l clip_C( P_x grad f(x; z_l) ) + P_x xi,   xi ~ N(0, sigma^2 I),

on a minibatch drawn without replacement, over every parameter that trains during the
round (a personal head included). Client i runs local_epochs * floor(m_i / B) such
steps per round. The clip constant C is fixed a priori (--dp-clip-C). sigma is
calibrated once so that the composition of all local steps meets (eps, delta). See
rfl/federated/privacy/orchestration.py. Without DP, a round is local_epochs passes over a
shuffled DataLoader.

Writes, in <save-root>/<experiment-name>/<run-name> or --save-dir:
    config.json, environment.json, fl_history.json, global_model_final.pt (last and
    selected global states), client_states.pt (what stays on each client: personal
    heads, local normalisation layers, re-fitted heads) and
    fl_results.json (configuration, DP budget, reported metrics, personalised
    metrics, selection and per-round history).

Usage (from any folder once the repository is installed, see README):
    python -m rfl.experiments.run_federated_experiment --dataset BNCI2014_001 \
        --model spdnet --rounds 200 --batch-size 64 --learning-rate 0.01
    python -m rfl.experiments.run_federated_experiment --dataset BNCI2014_001 \
        --model spdnet --rounds 200 --batch-size 64 --learning-rate 0.01 \
        --local-fc --dp-enabled --dp-target-eps 10

Requirements: MOABB for real EEG. A GPU is used when available.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn


from rfl.data.reduction import reduction_config
from rfl.data.registry import DATASET_CONFIGS, dataset_default
from rfl.experiments.run_centralized_experiment import _spd_bn_config, warn_uncalibrated
from rfl.federated.fl_config import FLConfig
from rfl.federated.server import FLServer, create_fl_clients
from rfl.models.eegnet import create_eegnet
from rfl.models.spdnet import create_spdnet, fix_batch_norm_dtype
from rfl.utils.reproducibility import log_environment, set_seed


def _global_n_classes(clients: List[Any], verbose: bool = True) -> int:
    """Number of classes of the global label space, taken over all clients.

    A single client can miss a class, so the head width cannot be read from one
    client. Labels are 0..K-1 (filter_binary_task remaps them), so the width is the
    largest label seen plus one.

    Raises:
        ValueError: if no client has any label.
    """
    seen = set()
    for c in clients:
        seen.update(c.info.get("classes") or range(c.info["n_classes"]))
    if not seen:
        raise ValueError("No labels found on any client.")
    n_classes = max(seen) + 1
    if verbose and n_classes != clients[0].info["n_classes"]:
        print(f"[info] global n_classes={n_classes} (client 0 saw "
              f"{clients[0].info['n_classes']}); using the union across clients.")
    return int(n_classes)


def _build_global_model(config: FLConfig, data_info: Dict[str, Any],
                        n_classes: int, device: torch.device,
                        verbose: bool = True) -> nn.Module:
    """Build the initial global SPDNet or EEGNet from the config and data shapes."""
    if config.model_type.lower() == "spdnet":
        model = create_spdnet(
            n_classes=n_classes,
            n_channels=data_info["n_channels"],
            seed=config.seed,
            **config.get_model_params(),
        )
        model.to(device)
        fix_batch_norm_dtype(model, device)
        if verbose:
            chain = " -> ".join(str(d) for d in [data_info["n_channels"]] + model.hidden_dims)
            print(f"[SPDNet] {model.depth} BiMap block(s): {chain}   "
                  f"ReEig eps = {model.epsilons}")
    else:
        model = create_eegnet(
            n_classes=n_classes,
            n_channels=data_info["n_channels"],
            n_timepoints=data_info["n_timepoints"],
            sampling_rate=data_info["sampling_rate"],
            **config.get_model_params(),
        )
        model.to(device)
    return model


def run_federated_experiment(
    config: FLConfig,
    dataset_name: str,
    subjects: List[int],
    save_dir: str,
    verbose: bool = True,
    synthetic_clients: str | None = None,
) -> Dict[str, Any]:
    """Run one federated experiment end to end and write its artefacts.

    Args:
        config: Federated configuration. Under DP, setup_dp writes the clip constant
            and sigma onto it.
        dataset_name: MOABB dataset name, or 'synthetic'.
        subjects: Subject IDs (ignored with synthetic clients).
        save_dir: Output directory, created if needed.
        verbose: Print progress.
        synthetic_clients: Path to a synthetic SPD .npz. When given, its stored
            clients are used instead of MOABB data.

    Returns:
        The results dict, also written to <save_dir>/fl_results.json.

    Raises:
        ValueError: if no client is created, or if the training-set reductions or
            the report rule are inconsistent.
    """
    os.makedirs(save_dir, exist_ok=True)
    set_seed(config.seed, deterministic=config.deterministic)

    device = config.torch_device()

    if verbose:
        print(f"\n{'=' * 70}")
        print("FEDERATED LEARNING EXPERIMENT")
        print(f"{'=' * 70}")
        print(f"Dataset:             {dataset_name}")
        print(f"Model:               {config.model_type}")
        print(f"Aggregation:         {config.fl_aggregation}")
        print(f"Rounds:              {config.fl_rounds}")
        print(f"Local epochs:        {config.fl_local_epochs}")
        print(f"Subjects per client: {config.fl_subjects_per_client}")
        print(f"Client partition:    {config.fl_client_partition}")
        print(f"Personalisation:     {config.personalisation_name()}")
        print(f"DP:                  {'on' if config.dp_enabled else 'off'}")
        print(f"Device:              {device}")
        print(f"{'=' * 70}\n")

    config.save(save_dir)
    log_environment(save_dir)

    # ---- clients ------------------------------------------------------------
    if synthetic_clients is not None:
        from rfl.data.synthetic import load_clients
        from rfl.federated.server import create_fl_clients_synthetic
        clients = create_fl_clients_synthetic(
            load_clients(synthetic_clients), config, device, verbose=verbose)
    else:
        clients = create_fl_clients(
            dataset_name=dataset_name,
            all_subjects=subjects,
            subjects_per_client=config.fl_subjects_per_client,
            config=config,
            device=device,
            verbose=verbose,
        )
    if not clients:
        raise ValueError("No clients created. Check the subject list.")

    # Shapes and sampling rate are shared by all clients. The label set may not be.
    data_info = clients[0].info
    n_classes = _global_n_classes(clients, verbose=verbose)

    # ---- global model --------------------------------------------------------
    global_model = _build_global_model(config, data_info, n_classes, device, verbose)
    n_params = sum(p.numel() for p in global_model.parameters())
    if verbose:
        print(f"\nGlobal model: {global_model.__class__.__name__} "
              f"({n_params:,} parameters)\n")

    for client in clients:
        client.set_model(global_model)

    # The re-fit selection needs validation records on every client. Check before
    # training rather than after.
    if config.head_refit_enabled and config.fl_head_refit_selection != 'fixed':
        empty = [c.client_id for c in clients if len(c.val_loader.dataset) == 0]
        if empty:
            raise ValueError(f"clients {empty} have no validation records, so the head "
                             "re-fit cannot be selected. Use --head-refit-selection fixed.")

    # ---- DP setup ------------------------------------------------------------
    # Before any training: fix C, calibrate sigma and record the realised budget.
    # setup_dp writes C and sigma onto the config, where DPFLClient reads them.
    dp_budget = None
    if config.dp_enabled:
        from rfl.federated.privacy.orchestration import setup_dp
        dp_budget = setup_dp(config, clients, global_model,
                             nn.CrossEntropyLoss(), device, verbose=verbose)

    # ---- federated training --------------------------------------------------
    server = FLServer(global_model=global_model, config=config, device=device)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        print(f"\n{'=' * 70}")
        print("Starting federated training")
        print(f"{'=' * 70}")

    # config.selection_metric (pooled over clients) drives the plateau schedule, the
    # early stopping below and, with report_rule='selected', the reported round. This
    # mirrors the centralized protocol of rfl/training/train_loop.py.
    sel_key, sel_min = server.selection_rule()
    stop_patience = int(config.early_stop_patience or 0)
    report_rule = config.report_rule                                  # checked by FLConfig
    best_val = float('inf') if sel_min else -float('inf')
    stale = 0
    for round_num in range(1, config.fl_rounds + 1):
        server.run_round(round_num=round_num, clients=clients,
                         criterion=criterion, verbose=verbose)
        # In local-only mode each client has its own stopping rule (in the server).
        # The run ends when all clients have stopped.
        if config.fl_local_only:
            if server.all_clients_stopped:
                if verbose:
                    print(f"\n[early stop] every client has stopped; ending at round "
                          f"{round_num} of {config.fl_rounds}")
                break
            continue
        if stop_patience > 0:
            hist = server.round_history.get(sel_key) or []
            if hist:
                improved = (hist[-1] < best_val) if sel_min else (hist[-1] > best_val)
                if improved:
                    best_val, stale = hist[-1], 0
                else:
                    stale += 1
                    if stale >= stop_patience:
                        if verbose:
                            print(f"\n[early stop] no improvement of {sel_key} for "
                                  f"{stale} rounds; stopping at round {round_num} "
                                  f"of {config.fl_rounds}")
                        break

    # ---- results -------------------------------------------------------------
    # `final_metrics` holds the reported round: the selected round, or the last one
    # when report_rule='last'. `last_round_metrics` always holds the last round.
    last_round_metrics = server.get_final_metrics()
    sel_round, sel_metrics = server.get_selected_round(
        report_metrics=("global_test_acc", "global_test_f1"),
    )
    if report_rule == 'last':
        final_metrics = last_round_metrics
    else:
        final_metrics = server.get_metrics_at_round(sel_round) or last_round_metrics

    server.save_global_model(os.path.join(save_dir, "global_model_final.pt"))
    with open(os.path.join(save_dir, "fl_history.json"), "w") as f:
        json.dump(server.round_history, f, indent=2)

    # ---- the reported model on every client ------------------------------------
    # Global state of the reported round with each client's local keys of that round.
    server.load_reported_model(clients, report_rule)
    reported_state = (server.best_state_dict
                      if report_rule == 'selected' and server.best_state_dict
                      else server.global_state_dict)
    if config.fl_local_only and report_rule == 'selected':
        # Each client holds the model of its own best round: report what they hold,
        # not a pooled round that matches none of them.
        ev = server.evaluate_clients(clients, criterion, local_only=True)
        final_metrics = {**last_round_metrics, **_metrics_of(ev)}
        sel_round, sel_metrics = None, {"per_client": True}

    # ---- exact running means of the SPD batch norm (rfl/models/bn_recalibration.py) --
    metrics_before_recalibration = None
    if config.spdnet_bn_recalibrate:
        from rfl.models.bn_recalibration import recalibrate_clients
        shared = not (config.fl_fedbn or config.fl_local_only)
        weights = ([float(c.n_samples) for c in clients] if config.fl_weight_by_samples
                   else [1.0] * len(clients))
        recalibrate_clients(clients, shared, weights, device, [reported_state])
        # The clients hold the reported model: evaluate it as it is.
        ev = server.evaluate_clients(clients, criterion, local_only=True)
        metrics_before_recalibration = final_metrics
        final_metrics = {**final_metrics, **_metrics_of(ev), "bn_recalibrated": True}
        # The saved global model must be the reported one (shared batch norm: the
        # combined statistics live in the global state).
        server.save_global_model(os.path.join(save_dir, "global_model_final.pt"))
        if verbose:
            print(f"[BN] running statistics recalibrated on the training data "
                  f"({'combined across clients' if shared else 'per client'}): val acc "
                  f"{metrics_before_recalibration.get('global_val_acc', float('nan')):.4f}"
                  f" -> {final_metrics['global_val_acc']:.4f}")

    # What each client holds for the reported model, before any re-fit changes its head.
    reported_clients = _client_snapshot(clients, full=config.fl_local_only)

    # ---- ReEig activity per block (SPDNet), on the validation split -------------
    reeig = (_reeig_activity(clients)
             if config.model_type.lower() == "spdnet" else None)
    if verbose and reeig is not None:
        print("[SPDNet] ReEig activity per block on validation "
              "(||ReEig(X) - X|| / ||X||, 0 = identity): "
              + ", ".join(f"{a:.3g}" for a in reeig))

    # ---- head re-fit (personalisation after the last release) ------------------
    personalized = None
    if config.head_refit_enabled:
        from rfl.federated.personalization import (parse_epoch_grid, parse_weight_decays,
                                               refit_and_evaluate)
        personalized = refit_and_evaluate(
            clients, reported_state, criterion, device,
            lr=float(config.fl_head_refit_lr or config.learning_rate),
            epoch_grid=parse_epoch_grid(config.fl_head_refit_epochs),
            weight_decays=parse_weight_decays(config.fl_head_refit_weight_decays),
            selection=config.fl_head_refit_selection, verbose=verbose)

    # Everything that stays on the clients, so that the reported model of each client
    # can be rebuilt from the saved files (with global_model_final.pt).
    if (personalized is not None or config.fl_local_only
            or any(c.local_state_dict() for c in clients)):
        _save_client_states(clients, os.path.join(save_dir, "client_states.pt"),
                            reported_clients, refit=personalized is not None)

    results = {
        "config": config.to_dict(),
        "model": {
            "type": config.model_type,
            "n_classes": n_classes,
            "n_parameters": n_params,
        },
        "fl_info": {
            "n_clients": len(clients),
            "subjects_per_client": config.fl_subjects_per_client,
            "client_partition": config.fl_client_partition,
            "rounds": config.fl_rounds,
            "local_epochs": config.fl_local_epochs,
            "aggregation": config.fl_aggregation,
            "client_subjects": [c.subjects for c in clients],
            "client_train_samples": [c.n_samples for c in clients],
            "client_val_samples": [len(c.val_loader.dataset) for c in clients],
            "train_trials_per_class": config.train_trials_per_class,
            "reduced_clients": [c.client_id for c in clients if c.reduced],
        },
        "personalisation": config.personalisation_name(),
        "dp_budget": dp_budget,
        "report_rule": report_rule,
        "final_metrics": final_metrics,
        "final_metrics_before_recalibration": metrics_before_recalibration,
        "personalized_metrics": personalized,
        "reeig_activity": reeig,
        "last_round_metrics": last_round_metrics,
        "selection": {
            "metric": sel_key,
            "round": sel_round,
            "score": server.best_score,
            "n_rounds_run": (server.round_history.get("round") or [None])[-1],
            "rounds_budget": config.fl_rounds,
            # True when the round budget ran out before early stopping fired.
            "hit_budget": bool((server.round_history.get("round") or [0])[-1]
                               >= config.fl_rounds),
            **sel_metrics,
        },
        "round_history": server.round_history,
    }

    with open(os.path.join(save_dir, "fl_results.json"), "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)

    if verbose:
        print(f"\n{'=' * 70}")
        print("Experiment complete")
        print(f"{'=' * 70}")
        n_run = (server.round_history.get('round') or [0])[-1]
        if report_rule == 'last':
            print(f"Reported round: last ({n_run} of {config.fl_rounds})")
        elif config.fl_local_only:
            print(f"Reported models: each client's own best round ({sel_key})")
        else:
            print(f"Selected round ({sel_key}): {sel_round} of {n_run} run "
                  f"(budget {config.fl_rounds})")
        print(f"Test acc (pooled): {final_metrics.get('global_test_acc', float('nan')):.4f}")
        print(f"Test F1 (macro):   {final_metrics.get('global_test_f1', float('nan')):.4f}")
        print(f"[ref] last round: "
              f"acc={last_round_metrics.get('global_test_acc', float('nan')):.4f}, "
              f"F1={last_round_metrics.get('global_test_f1', float('nan')):.4f}")
        if personalized is not None:
            print(f"After head re-fit:  acc="
                  f"{personalized['personalized_test_acc']:.4f}, "
                  f"F1={personalized['personalized_test_f1']:.4f}")
        if dp_budget is not None:
            print(f"DP: C={dp_budget['C']:.4g}  sigma={dp_budget['sigma']:.4g}  "
                  f"mu0={dp_budget['mu0']:.4g}  "
                  f"(eps={dp_budget['realized_eps']:.4g}, "
                  f"delta={dp_budget['delta']:.0e})")
        print(f"Results: {save_dir}")
        print(f"{'=' * 70}\n")

    return results


@torch.no_grad()
def _reeig_activity(clients: List[Any]) -> List[float]:
    """Mean relative change made by each ReEig on the validation records.

    Weighted by the number of validation batches of each client. A value near 0 means
    that the ReEig of that block acts as the identity, so the block adds no
    non-linearity (rfl/models/spdnet.py). Each client uses the model it holds, which the
    runner sets to the reported model (FLServer.load_reported_model).
    """
    total, n_batches = None, 0
    for c in clients:
        m = c.local_model
        if not hasattr(m, "reeig_activity"):
            return None
        m.eval()
        m.track_clipping = True
        m.reset_clip_stats()
        for x, _ in c.val_loader:
            m(x.to(c.device))
        k = m._clip_calls
        if k:
            act = [a * k for a in m.reeig_activity()]
            total = act if total is None else [t + a for t, a in zip(total, act)]
            n_batches += k
        m.track_clipping = False
        m.reset_clip_stats()
    return [t / n_batches for t in total] if n_batches else None


def _metrics_of(ev: Dict[str, Any]) -> Dict[str, Any]:
    """The round metrics of `FLServer.evaluate_clients` output (pooled and per client)."""
    out = {k: v for k, v in ev.items() if k.startswith("global_")}
    va, te, tf = ev["val_accs"], ev["test_accs"], ev["test_f1s"]
    out.update(avg_val_acc=float(np.mean(va)), std_val_acc=float(np.std(va)),
               avg_test_acc=float(np.mean(te)), std_test_acc=float(np.std(te)),
               avg_test_f1=float(np.mean(tf)), std_test_f1=float(np.std(tf)),
               per_client_test_acc=[float(a) for a in te],
               per_client_test_f1=[float(f) for f in tf],
               per_client_val_acc=[float(a) for a in va])
    return out


def _client_snapshot(clients: List[Any], full: bool = False) -> Dict[int, Dict]:
    """Per client: its local keys, its classifier head and (with `full`) its model."""
    from rfl.federated.param_groups import identify_classifier_param_names
    snap = {}
    for c in clients:
        state = c.local_model.state_dict()
        names = identify_classifier_param_names(c.local_model)
        snap[int(c.client_id)] = {
            "local": {k: v.detach().cpu().clone() for k, v in c.local_state_dict().items()},
            "head": {k: state[k].detach().cpu().clone() for k in names},
            "full": ({k: v.detach().cpu().clone() for k, v in state.items()}
                     if full else None),
        }
    return snap


def _save_client_states(clients: List[Any], path: str, reported: Dict[int, Dict],
                        refit: bool = False) -> None:
    """Save what stays on each client, keyed by client id.

    `local_states` (local keys: personal head, local normalisation layers) and `heads`
    (the classifier head) are those of the reported model, before any re-fit: the
    global state of the reported round (global_model_final.pt: `best_state_dict` with
    report 'selected', `model_state_dict` with 'last'), then `local_states[cid]`,
    rebuilds the model behind `final_metrics`. `refit_heads` holds the re-fitted heads,
    behind `personalized_metrics`: load them on top. `full_states` holds whole models
    in local-only runs.
    """
    from rfl.federated.param_groups import identify_classifier_param_names
    out = {"local_states": {}, "heads": {}, "refit_heads": {}, "full_states": {},
           "subjects": {int(c.client_id): c.subjects for c in clients}}
    for c in clients:
        cid = int(c.client_id)
        snap = reported[cid]
        out["local_states"][cid] = snap["local"]
        out["heads"][cid] = snap["head"]
        if snap["full"] is not None:
            out["full_states"][cid] = snap["full"]
        if refit:
            state = c.local_model.state_dict()
            out["refit_heads"][cid] = {k: state[k].detach().cpu().clone()
                                       for k in identify_classifier_param_names(c.local_model)}
    torch.save(out, path)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser():
    import argparse

    from rfl.experiments.cli_checks import HelpFormatter

    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Federated SPDNet/EEGNet experiment (real EEG or synthetic SPD "
                    "clients), with optional personalisation and optional record-level "
                    "DP.",
        formatter_class=HelpFormatter)

    # -- data source: exactly one of --dataset and --synthetic-clients ---------
    g = p.add_argument_group("data")
    g.add_argument("--dataset", type=str, default=None, choices=sorted(DATASET_CONFIGS),
                   help="MOABB dataset (real EEG). Clients are groups of subjects.")
    g.add_argument("--synthetic-clients", type=str, default=None,
                   help="Path to a synthetic SPD .npz. Each stored client becomes an "
                        "FL client. Mutually exclusive with --dataset.")
    g.add_argument("--subjects", type=int, nargs="+", default=None,
                   help="Subject IDs (real EEG). Default: all subjects of the dataset.")
    g.add_argument("--task", type=str, default="all_classes",
                   choices=["binary", "all_classes", "no_rest"], help="Classes kept.")
    g.add_argument("--resample", type=float, default=128.0,
                   help="Target sampling rate in Hz. 0 keeps the native rate. Never "
                        "upsamples.")
    g.add_argument("--fmin", type=float, default=8.0, help="Band-pass low cutoff (Hz).")
    g.add_argument("--fmax", type=float, default=32.0, help="Band-pass high cutoff (Hz).")
    g.add_argument("--val-size", "--local-val-size", dest="val_size", type=float,
                   default=0.10, help="Validation fraction of the global split, shared "
                                      "with the centralised runs.")
    g.add_argument("--test-size", "--local-test-size", dest="test_size", type=float,
                   default=0.15, help="Test fraction of the global split.")
    # Covariance (real EEG, SPDNet). --cov-shrinkage is only used by 'lwf'.
    g.add_argument("--cov-estimator", type=str, default="scm", choices=["scm", "lwf"])
    g.add_argument("--cov-shrinkage", type=float, default=0.0)
    g.add_argument("--cov-regularization", type=float, default=0.0)
    from rfl.data.reduction import add_reduction_args
    add_reduction_args(p)

    # -- federation ------------------------------------------------------------
    g = p.add_argument_group("federation")
    from rfl.federated.aggregation import rule_names
    g.add_argument("--aggregation", type=str, default=None, choices=list(rule_names()),
                   help="Default: projavg for SPDNet, fedavg for EEGNet. Rules are "
                        "registered in rfl/federated/aggregation.py (AGGREGATION_RULES).")
    g.add_argument("--subjects-per-client", type=int, default=1,
                   help="Real EEG only. Number of consecutive subjects in one client.")
    g.add_argument("--client-partition", default="subject", choices=["subject", "iid"],
                   help="Real EEG only. 'subject': each client holds its subjects' "
                        "trials (non-IID). 'iid': the trials are pooled and dealt at "
                        "random to the same number of clients, stratified by class.")
    g.add_argument("--rounds", type=int, default=50, help="Communication rounds T.")
    g.add_argument("--local-epochs", type=int, default=1,
                   help="Local epochs per client and round.")
    g.add_argument("--client-fraction", type=float, default=1.0,
                   help="Fraction of clients sampled per round. Participation is visible "
                        "to the server, so it gives no privacy amplification.")
    g.add_argument("--weight-by-samples", action="store_true",
                   help="Weight aggregation by client sample count (default: uniform).")
    g.add_argument("--no-fedbn", dest="fedbn", action="store_false",
                   help="Share the normalisation layers (EEGNet BatchNorm, SPD batch "
                        "norm) instead of keeping them on each client (FedBN). Trainable "
                        "parameters are averaged. SPD running means are combined in the "
                        "geometry of their layer.")

    # -- personalisation -------------------------------------------------------
    g = p.add_argument_group("personalisation")
    g.add_argument("--local-fc", action="store_true",
                   help="Personal classifier head (FedPer): only the trunk is "
                        "aggregated.")
    g.add_argument("--local-head-mode", type=str, default="train",
                   choices=["train", "frozen"],
                   help="With --local-fc: train the head during the rounds (under DP it "
                        "is privatised with the trunk), or keep it at its random "
                        "initialisation and re-fit it after training.")
    g.add_argument("--head-refit-epochs", type=str, default="",
                   help="Comma-separated epoch grid of the post-training head re-fit, "
                        "e.g. '25,50,100,200'. Empty: no re-fit.")
    g.add_argument("--head-refit-weight-decays", type=str, default="0",
                   help="Comma-separated weight decays of the head re-fit.")
    g.add_argument("--head-refit-lr", type=float, default=None,
                   help="Learning rate of the head re-fit. Default: --learning-rate.")
    g.add_argument("--head-refit-selection", type=str, default="wd_and_epochs",
                   choices=["fixed", "wd_only", "wd_and_epochs"],
                   help="Per-client selection of the re-fit candidate on validation "
                        "accuracy.")
    g.add_argument("--local-only", action="store_true",
                   help="No broadcast and no aggregation. Every client trains and is "
                        "evaluated on its own model, with its own schedule and stopping "
                        "rule. With --dp-enabled, DP then protects each client's model.")
    g.add_argument("--freeze-trunk", action="store_true",
                   help="Keep every parameter except the classifier head at its random "
                        "initialisation.")

    # -- differential privacy --------------------------------------------------
    g = p.add_argument_group("differential privacy")
    g.add_argument("--dp-enabled", action="store_true",
                   help="Record-level DP: per-sample tangent clipping, tangent Gaussian "
                        "noise and minibatch subsampling without replacement.")
    g.add_argument("--dp-target-eps", type=float, default=10.0, help="Target epsilon.")
    g.add_argument("--dp-target-delta", type=float, default=1e-5)
    g.add_argument("--dp-clip-C", type=float, default=1.0,
                   help="Clip constant C, fixed a priori.")
    g.add_argument("--dp-accounting", type=str, default="ours",
                   choices=["ours", "prirfed"],
                   help="Accounting used to calibrate sigma. 'prirfed' uses Theorem 7 of "
                        "PriRFed (full participation), for the comparison baseline.")
    g.add_argument("--dp-compare-prirfed", action="store_true",
                   help="Also record the noise PriRFed's accounting would need for the "
                        "same target (reporting only). Doubles the calibration time.")
    g.add_argument("--dp-accountant-h", type=float, default=0.01,
                   help="Privacy-loss grid spacing of the accountant, refined internally "
                        "to min(h, mu0/40).")

    # -- model -----------------------------------------------------------------
    g = p.add_argument_group("model")
    g.add_argument("--model", type=str, default="spdnet", choices=["spdnet", "eegnet"])
    g.add_argument("--hidden-dim", type=int, default=None,
                   help="SPDNet BiMap output dimension (<= n_channels). Default: the "
                        "dataset value of rfl/data/registry.py, n_channels // 2 otherwise.")
    g.add_argument("--epsilon", type=float, default=None,
                   help="SPDNet ReEig eigenvalue floor. Default: the dataset value of "
                        "rfl/data/registry.py, 1e-4 otherwise.")
    g.add_argument("--hidden-dims", type=str, default="",
                   help="Comma-separated BiMap output dimensions of a deep SPDNet, e.g. "
                        "'16,8'. Overrides --hidden-dim.")
    g.add_argument("--epsilons", type=str, default="",
                   help="One ReEig floor per block, strictly increasing, e.g. "
                        "'1e-4,1e-2'. Required with --hidden-dims.")
    g.add_argument("--reeig-percentile", type=float, default=None,
                   help="Set the ReEig floors to this percentile of the spectrum each "
                        "ReEig acts on, from the floors file of analyze_covariances "
                        "--architectures (needs --hidden-dims).")
    g.add_argument("--floors-file", default=None,
                   help="Floors file of --reeig-percentile (default "
                        "$RFL_RESULTS/reeig_floors/<dataset>.json).")
    g.add_argument("--stiefel", default="projection",
                   choices=["projection", "trivialization"],
                   help="BiMap weights: optimiser step then polar projection, or the "
                        "library's orthogonal parametrization (no DP).")
    from rfl.experiments.run_centralized_experiment import _add_spd_bn_args
    _add_spd_bn_args(g)
    tn = g.add_mutually_exclusive_group()
    tn.add_argument("--trace-normalize", dest="trace_normalize", action="store_true")
    tn.add_argument("--no-trace-normalize", dest="trace_normalize", action="store_false")
    p.set_defaults(trace_normalize=True)
    g.add_argument("--eegnet-norm-type", type=str, default="none",
                   choices=["none", "batch"],
                   help="EEGNet normalisation: 'none' (no normalisation layer) or "
                        "'batch' (original architecture, kept local to each client as "
                        "in FedBN, refused with --dp-enabled). Ignored for spdnet.")
    g.add_argument("--eegnet-input-norm", default="per_trial",
                   choices=["per_trial", "per_trial_global"],
                   help="EEGNet input standardisation, per trial: each channel on its own "
                        "(the papers), or all channels together, which keeps the relative "
                        "channel powers (the motor-imagery feature SPDNet sees).")

    # -- training --------------------------------------------------------------
    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=64,
                   help="Local batch size B. Under DP it is the subsampled batch size, "
                        "with sensitivity 2C/B. It must not exceed the smallest client.")
    g.add_argument("--learning-rate", type=float, default=1e-3,
                   help="Adam learning rate of the local steps (start of the schedule).")
    g.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    g.add_argument("--optimizer", default="adam", choices=["adam", "sgd"],
                   help="Optimiser of the local steps and of the head re-fit.")
    g.add_argument("--momentum", type=float, default=0.0,
                   help="Heavy-ball momentum of --optimizer sgd (0: plain SGD).")
    g.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["none", "cosine", "step", "plateau"],
                   help="Learning rate per round: cosine to --lr-min, plateau, step or constant (none).")
    g.add_argument("--lr-min", type=float, default=None,
                   help="Floor of the cosine and plateau schedules. Default: "
                        "--learning-rate / 100.")
    g.add_argument("--lr-plateau-patience", type=int, default=20,
                   help="Rounds without a strict improvement of the selection metric "
                        "before the rate is multiplied by --lr-plateau-factor "
                        "(--lr-schedule plateau).")
    g.add_argument("--lr-plateau-factor", type=float, default=0.5,
                   help="Multiplier applied at each plateau.")
    g.add_argument("--early-stop-patience", "--early-stopping-patience",
                   dest="early_stop_patience", type=int, default=0,
                   help="Stop after this many rounds without a strict improvement of the "
                        "selection metric. 0 runs the full budget.")
    g.add_argument("--selection-metric", type=str, default="val_acc",
                   choices=["val_loss", "val_acc"],
                   help="Pooled validation metric that drives the plateau schedule, "
                        "early stopping and the selected round.")
    g.add_argument("--report", type=str, default="last", choices=["selected", "last"],
                   help="Round whose metrics are reported: the last one, or the one "
                        "selected on --selection-metric. 'last' requires "
                        "--early-stop-patience 0.")

    # -- misc and output -------------------------------------------------------
    g = p.add_argument_group("output")
    g.add_argument("--seed", type=int, default=42,
                   help="Seed of the model initialisation, the training randomness and "
                        "the data split.")
    g.add_argument("--device", type=str, default="auto",
                   help="cpu, cuda, cuda:N, or auto (cuda when available, else cpu)")
    g.add_argument("--experiment-name", type=str, default="federated")
    g.add_argument("--run-name", type=str, default=None)
    g.add_argument("--save-root", type=str, default="./results")
    g.add_argument("--save-dir", type=str, default=None,
                   help="Output directory. Default: <save-root>/<experiment-name>/<run-name>.")
    g.add_argument("--print-config", action="store_true",
                   help="Print the complete configuration of this command (JSON), with "
                        "every default resolved, and exit without running.")
    return p


def apply_defaults(args, src: str):
    """Fill the options left unset.

    Aggregation (projavg for SPDNet, fedavg for EEGNet), SPDNet BiMap dimension and
    ReEig floor of dataset `src` (rfl/data/registry.py), and lr_min = learning_rate / 100.
    Synthetic data falls back to n_channels // 2 and 1e-4.
    """
    if args.aggregation is None:
        args.aggregation = "fedavg" if args.model == "eegnet" else "projavg"
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


def config_from_args(args) -> FLConfig:
    """Build the FLConfig of a parsed command line (after `apply_defaults`)."""
    is_syn = args.synthetic_clients is not None
    src = "synthetic" if is_syn else args.dataset
    data_type = "raw" if args.model == "eegnet" else ("spd" if is_syn else "covariance")
    return FLConfig(
        experiment_name=args.experiment_name, run_name=args.run_name or "run",
        dataset=src, subjects=[] if is_syn else list(args.subjects),
        data_path=os.path.abspath(args.synthetic_clients) if is_syn else "",
        data_type=data_type, task=args.task,
        resample=args.resample, fmin=args.fmin, fmax=args.fmax,
        val_size=args.val_size, test_size=args.test_size,
        **reduction_config(args),
        cov_estimator=args.cov_estimator, cov_shrinkage=args.cov_shrinkage,
        cov_regularization=args.cov_regularization,
        # federation
        fl_aggregation=args.aggregation,
        fl_subjects_per_client=args.subjects_per_client,
        fl_client_partition=args.client_partition,
        fl_rounds=args.rounds, fl_local_epochs=args.local_epochs,
        fl_client_fraction=args.client_fraction,
        fl_weight_by_samples=args.weight_by_samples,
        fl_fedbn=args.fedbn,
        # personalisation
        fl_local_fc=args.local_fc,
        fl_local_head_mode=args.local_head_mode,
        fl_head_refit_epochs=args.head_refit_epochs,
        fl_head_refit_weight_decays=args.head_refit_weight_decays,
        fl_head_refit_lr=args.head_refit_lr,
        fl_head_refit_selection=args.head_refit_selection,
        fl_local_only=args.local_only,
        freeze_trunk=args.freeze_trunk,
        # privacy
        dp_enabled=args.dp_enabled,
        dp_target_eps=args.dp_target_eps,
        dp_target_delta=args.dp_target_delta,
        dp_clip_C=args.dp_clip_C,
        dp_accounting=args.dp_accounting,
        dp_compare_prirfed=args.dp_compare_prirfed,
        dp_accountant_h=args.dp_accountant_h,
        # model
        model_type=args.model,
        spdnet_hidden_dim=args.hidden_dim,
        spdnet_epsilon=args.epsilon,
        spdnet_hidden_dims=args.hidden_dims,
        spdnet_epsilons=args.epsilons,
        spdnet_use_batch_norm=args.use_batch_norm,
        **_spd_bn_config(args),
        spdnet_trace_normalize=args.trace_normalize,
        eegnet_norm_type=args.eegnet_norm_type,
        normalization_type=args.eegnet_input_norm,
        # training
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer, sgd_momentum=args.momentum,
        spdnet_stiefel=args.stiefel, spdnet_reeig_percentile=args.reeig_percentile,
        lr_schedule=args.lr_schedule, lr_min=args.lr_min,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        early_stop_patience=args.early_stop_patience,
        selection_metric=args.selection_metric,
        report_rule=args.report,
        # misc
        seed=args.seed, data_split_seed=args.seed,
        device=args.device, save_root=args.save_root,
    )


def default_run_name(config: FLConfig) -> str:
    """'<model>_<aggregation>_<source>_<personalisation>_<nodp|eps..>_seed<seed>_<hash>'.

    The hash covers every setting except the seed and the output location, so two
    commands that differ in any option never share a directory.
    """
    privacy = f"eps{config.dp_target_eps:g}" if config.dp_enabled else "nodp"
    pers = config.personalisation_name().replace("+", "-")
    from rfl.analysis.results import config_id
    return (f"{config.model_type}_{config.fl_aggregation}_{config.dataset}_{pers}_"
            f"{privacy}_seed{config.seed}_{config_id(config.to_dict())}")


def prepare_args(argv=None):
    """Parse and complete a command line: (parser, args, data source).

    Raises:
        SystemExit: on an invalid command line (argparse error).
    """
    p = _build_arg_parser()
    args = p.parse_args(argv)
    if (args.dataset is None) == (args.synthetic_clients is None):
        p.error("provide exactly one of --dataset or --synthetic-clients")
    is_syn = args.synthetic_clients is not None
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


def build_config(args) -> FLConfig:
    """The FLConfig of a completed command line, with its default run name."""
    config = config_from_args(args)
    if args.run_name is None:
        config.run_name = default_run_name(config)
        config.update()                    # refresh experiment_dir and hash
    return config


def print_config(config) -> None:
    """The configuration as indented JSON, as saved in config.json by a run."""
    import json
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True, default=str))


def check_command(argv) -> None:
    """Build the configuration of a command line without running it.

    Raises:
        SystemExit: on an invalid command line.
        ValueError: on an invalid configuration.
    """
    _, args, _ = prepare_args(argv)
    try:
        build_config(args)
    except ValueError as e:
        from rfl.experiments.cli_checks import with_flags
        raise ValueError(with_flags(str(e))) from None


def main(argv=None):
    from rfl.utils.logs import setup_logging
    setup_logging()
    p, args, src = prepare_args(argv)
    try:
        config = build_config(args)
    except ValueError as e:
        from rfl.experiments.cli_checks import with_flags
        p.error(with_flags(str(e)))
    if args.print_config:
        print_config(config)
        return
    save_dir = args.save_dir or os.path.join(
        args.save_root, args.experiment_name, config.run_name)

    from rfl.training.divergence import DIVERGENCE_ERRORS, write_diverged
    try:
        run_federated_experiment(config, src, config.subjects, save_dir,
                                 synthetic_clients=args.synthetic_clients)
    except DIVERGENCE_ERRORS as e:            # a result (rfl/training/divergence.py)
        write_diverged(save_dir, config, "fl_results.json", e)


if __name__ == "__main__":
    main()
