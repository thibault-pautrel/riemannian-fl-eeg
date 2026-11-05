"""Options without effect: refused, so that a run never silently differs from its command.

An option is *set* when it appears on the command line, even with its default value. Each rule below
names options that only act under a condition. If such an option is set while the
condition is false, the runner stops with a message, before any data is loaded. For
example `--dp-target-eps 5` without `--dp-enabled` would otherwise run without DP.

Both runners call `inert_option_errors` in `prepare_args`. A rule whose options a
runner does not have is skipped.
"""

from __future__ import annotations

import argparse
from typing import Callable, List, Sequence, Tuple

Rule = Tuple[Sequence[str], Callable[[object], bool], str]


def _bn_layer(a) -> bool:
    return ((a.model == "eegnet" and a.eegnet_norm_type == "batch")
            or (a.model == "spdnet" and a.use_batch_norm))


def _real(a) -> bool:
    return getattr(a, "dataset", None) is not None


def _refit_on(a) -> bool:
    from rfl.federated.personalization import parse_epoch_grid
    return bool(parse_epoch_grid(a.head_refit_epochs))


def _val_driven(a) -> bool:
    patience = getattr(a, "early_stop_patience", None)
    if patience is None:
        patience = getattr(a, "early_stopping_patience", 0)
    return a.report == "selected" or a.lr_schedule == "plateau" or patience > 0


DP = ("dp_target_eps", "dp_target_delta", "dp_clip_C", "dp_accounting",
      "dp_compare_prirfed", "dp_accountant_h")
BN = ("bn_type", "bn_mean_type", "bn_momentum", "bn_norm_and_bias_mode", "bn_position")
SPD = ("use_batch_norm", "hidden_dim", "epsilon", "hidden_dims", "epsilons",
       "trace_normalize", "stiefel", "reeig_percentile")
COV = ("cov_estimator", "cov_shrinkage", "cov_regularization")
PREPROC = ("subjects", "task", "resample", "fmin", "fmax", "subjects_per_client",
           "client_partition")
REFIT = ("head_refit_weight_decays", "head_refit_lr", "head_refit_selection")
FEDERATION = ("aggregation", "client_fraction", "weight_by_samples", "fedbn")

RULES: List[Rule] = [
    (DP, lambda a: a.dp_enabled, "only acts with --dp-enabled"),
    (SPD, lambda a: a.model == "spdnet", "only acts with --model spdnet"),
    (("momentum",), lambda a: a.optimizer == "sgd", "only acts with --optimizer sgd"),
    (("floors_file",), lambda a: a.reeig_percentile is not None,
     "only acts with --reeig-percentile"),
    (BN, lambda a: a.model == "spdnet" and a.use_batch_norm,
     "only acts with --model spdnet --use-batch-norm"),
    (("eegnet_norm_type", "eegnet_input_norm"), lambda a: a.model == "eegnet",
     "only acts with --model eegnet"),
    (("bn_recalibrate",), _bn_layer,
     "only acts with a batch norm (--use-batch-norm, or --eegnet-norm-type batch)"),
    (COV, lambda a: a.model == "spdnet" and _real(a),
     "only acts with --model spdnet on a real dataset (--dataset)"),
    (("cov_shrinkage",), lambda a: a.cov_estimator == "lwf",
     "only acts with --cov-estimator lwf"),
    (PREPROC, _real, "only acts on a real dataset (--dataset), not on synthetic data"),
    (("hidden_dim", "epsilon"), lambda a: not str(a.hidden_dims).strip(),
     "is replaced by --hidden-dims/--epsilons: give the values per block there"),
    (("epsilons",), lambda a: bool(str(a.hidden_dims).strip()),
     "needs --hidden-dims (one floor per block)"),
    (("lr_plateau_patience", "lr_plateau_factor"), lambda a: a.lr_schedule == "plateau",
     "only acts with --lr-schedule plateau"),
    (("lr_min",), lambda a: a.lr_schedule in ("cosine", "plateau"),
     "only acts with --lr-schedule cosine or plateau"),
    (("selection_metric",), _val_driven,
     "only acts with --report selected, --lr-schedule plateau or early stopping"),
    (REFIT, lambda a: _refit_on(a), "only acts with --head-refit-epochs (a non-zero grid)"),
    (FEDERATION, lambda a: not a.local_only,
     "has no effect with --local-only (nothing is aggregated)"),
    (("local_head_mode",), lambda a: a.local_fc, "only acts with --local-fc"),
    (("calibration_val_fraction", "calibration_order", "reduced_subjects"),
     lambda a: a.train_trials_per_class is not None,
     "only acts with --train-trials-per-class"),
    (("fedbn",), _bn_layer,
     "needs a normalisation layer: --use-batch-norm (SPDNet) or "
     "--eegnet-norm-type batch (EEGNet)"),
]


def dataset_errors(args, src: str):
    """(errors, warnings) of a completed command line against the dataset statistics.

    Checks the BiMap dimensions against the number of EEG channels, and under DP the
    batch size against the approximate size of the smallest client training set
    (rfl/data/stats.py). Skipped for synthetic data, and for the client sizes when the
    statistics file of the dataset does not exist yet.
    """
    from rfl.data.stats import client_trial_counts, eeg_channel_count, stats_path
    errors, warns = [], []
    # only the centralised runner has --bn-position
    if args.model == "spdnet" and getattr(args, "bn_position", "after_reeig") == "before_reeig":
        warns.append("--bn-position before_reeig: every ReEig block then reads batch-"
                     "normalised matrices (spectrum near the identity). The registry floors "
                     "were set for trace-normalised covariances. Recalibrate "
                     "--epsilon/--epsilons (README, SPD batch norm).")
    if src == "synthetic":
        return errors, warns
    from rfl.data.registry import get_events
    events = get_events(src)
    if args.task == "binary":
        if not {"left_hand", "right_hand"} <= set(events):
            errors.append(f"--task binary keeps left_hand and right_hand, which {src} "
                          f"does not both have ({events})")
        elif len(events) == 2:
            errors.append(f"--task binary has no effect: {src} has only {events} "
                          "(use the default all_classes)")
    if args.task == "no_rest" and "rest" not in events:
        errors.append(f"--task no_rest has no effect: {src} has no rest class "
                      "(use the default all_classes)")
    if args.model == "spdnet":
        from rfl.models.spdnet import parse_int_list
        dims = (parse_int_list(args.hidden_dims, []) if str(args.hidden_dims).strip()
                else ([args.hidden_dim] if args.hidden_dim else []))
        n_ch = eeg_channel_count(src)
        chain = ([n_ch] if n_ch else []) + dims
        if any(b > a for a, b in zip(chain, chain[1:])):
            errors.append(f"BiMap dimensions {' -> '.join(map(str, chain))}: each "
                          "dimension must not exceed the previous one "
                          f"({src} has {n_ch} EEG channels)" if n_ch else
                          f"BiMap dimensions {dims} must not increase")
    if (src in INTERPOLATED and args.model == "spdnet"
            and not getattr(args, "cov_regularization", 0)):
        warns.append(f"{src} interpolates its bad channels, so its covariances are "
                     "rank-deficient: their smallest eigenvalues are zero up to round-off. "
                     "--cov-regularization adds a diagonal floor (in the units of the "
                     "covariance, uV^2).")
    if getattr(args, "dp_enabled", False):
        counts = client_trial_counts(src, list(args.subjects), args.subjects_per_client,
                                     args.task)
        if counts and args.client_partition == "iid":    # the same share for every client
            counts = [sum(counts) / len(counts)] * len(counts)
        if not counts:
            warns.append(f"no trial counts for {src} ({stats_path(src)}): the DP batch "
                         "size is checked against the smallest client only when the run "
                         "starts. `bash condor/fetch_data.sh "
                         f"{src}` (or python -m rfl.experiments.fetch_dataset) writes them.")
        else:
            # A client keeps a binomial share of its trials for training (the split is
            # stratified over all trials): expected f n, spread sqrt(f (1 - f) n).
            f = 1.0 - args.val_size - args.test_size
            expected = min(f * n for n in counts)
            low = min(f * n - 3.0 * (f * (1 - f) * n) ** 0.5 for n in counts)
            B = args.batch_size
            if B > expected:
                errors.append(f"--batch-size {B} exceeds the smallest client training "
                              f"set (about {expected:.0f} records, {stats_path(src)}): "
                              "a DP step draws B records of one client")
            elif B > low:
                warns.append(f"--batch-size {B} is close to the smallest client "
                             f"training set (about {expected:.0f} records, as few as "
                             f"{max(low, 0):.0f} for some seeds): a run may stop at the "
                             "exact check")
    return errors, warns


# Datasets whose loader interpolates bad channels (MOABB 1.7.2).
INTERPOLATED = ("Stieger2021",)


def _flag(parser, dest: str) -> str:
    for action in parser._actions:
        if action.dest == dest and action.option_strings:
            return "/".join(action.option_strings)
    return dest


def explicit_dests(parser, argv) -> set:
    """Destinations of the options that appear in `argv` (None: sys.argv[1:])."""
    import argparse
    import copy
    import sys
    probe = copy.deepcopy(parser)
    for action in probe._actions:
        action.default = argparse.SUPPRESS
        action.required = False
    probe.set_defaults(**{k: argparse.SUPPRESS for k in probe._defaults})
    probe._defaults = {}
    ns, _ = probe.parse_known_args(sys.argv[1:] if argv is None else list(argv))
    return set(vars(ns))


def inert_option_errors(parser, args, explicit=None) -> List[str]:
    """Messages for the options given on the command line that have no effect.

    `explicit` is the set of destinations given on the command line
    (`explicit_dests`). Without it, an option counts as given when its value differs
    from the parser default.
    """
    have = vars(args)
    errors = []
    for dests, active, why in RULES:
        dests = [d for d in dests if d in have]
        if not dests:
            continue
        try:
            on = active(args)
        except AttributeError:          # the runner lacks an option of the condition
            continue
        if on:
            continue
        for d in dests:
            given = (d in explicit) if explicit is not None else (
                have[d] != parser.get_default(d))
            if given:
                errors.append(f"{_flag(parser, d)} {why}")
    return errors


# Configuration fields named in validation messages, and their command-line options.
FIELD_FLAGS = {
    "fl_aggregation": "--aggregation", "fl_local_fc": "--local-fc",
    "fl_client_partition": "--client-partition", "spdnet_stiefel": "--stiefel",
    "spdnet_reeig_percentile": "--reeig-percentile",
    "optimizer": "--optimizer", "sgd_momentum": "--momentum",
    "fl_local_head_mode": "--local-head-mode", "fl_local_only": "--local-only",
    "fl_head_refit_epochs": "--head-refit-epochs", "fl_head_refit_lr": "--head-refit-lr",
    "fl_head_refit_selection": "--head-refit-selection", "fl_fedbn": "--no-fedbn",
    "fl_client_fraction": "--client-fraction", "fl_rounds": "--rounds",
    "fl_local_epochs": "--local-epochs", "freeze_trunk": "--freeze-trunk",
    "report_rule": "--report", "early_stop_patience": "--early-stop-patience",
    "early_stopping_patience": "--early-stopping-patience",
    "selection_metric": "--selection-metric", "lr_schedule": "--lr-schedule",
    "lr_min": "--lr-min", "lr_plateau_patience": "--lr-plateau-patience",
    "lr_plateau_factor": "--lr-plateau-factor", "dp_target_eps": "--dp-target-eps",
    "dp_target_delta": "--dp-target-delta", "dp_clip_C": "--dp-clip-C",
    "dp_enabled": "--dp-enabled", "spdnet_use_batch_norm": "--use-batch-norm",
    "spdnet_bn_type": "--bn-type", "spdnet_bn_mean_type": "--bn-mean-type",
    "spdnet_bn_position": "--bn-position", "spdnet_epsilons": "--epsilons",
    "spdnet_hidden_dims": "--hidden-dims", "eegnet_norm_type": "--eegnet-norm-type",
    "cov_estimator": "--cov-estimator", "cov_shrinkage": "--cov-shrinkage",
    "batch_size": "--batch-size", "weight_decay": "--weight-decay",
    "val_size": "--val-size", "test_size": "--test-size", "subjects": "--subjects",
    "train_trials_per_class": "--train-trials-per-class",
    "calibration_val_fraction": "--calibration-val-fraction",
    "calibration_order": "--calibration-order", "reduced_subjects": "--reduced-subjects",
    "spdnet_bn_recalibrate": "--bn-recalibrate", "normalization_type": "--eegnet-input-norm", "spdnet_bn_momentum": "--bn-momentum",
}


def resolve_reeig_percentile(p, args, src: str) -> None:
    """--reeig-percentile Q: set --epsilons from the floors file (analyze_covariances
    --architectures), for the architecture --hidden-dims, with or without batch norm.

    Calls p.error on a combination without floors (no file, no entry, explicit floors).
    """
    q = args.reeig_percentile
    if q is None:
        return
    if args.model != "spdnet" or src == "synthetic":
        p.error("--reeig-percentile needs --model spdnet on a real dataset (--dataset)")
    if not str(args.hidden_dims).strip():
        p.error("--reeig-percentile needs --hidden-dims: the floors depend on the "
                "architecture")
    if str(args.epsilons).strip() or args.epsilon is not None:
        p.error("--reeig-percentile sets the ReEig floors: drop --epsilon/--epsilons")
    if args.use_batch_norm and getattr(args, "bn_position", "after_reeig") != "after_reeig":
        p.error("--reeig-percentile: the floors are computed with the batch norm after "
                "ReEig (--bn-position after_reeig)")
    from rfl.experiments.analyze_covariances import floors_for
    try:
        args.epsilons = floors_for(src, args.hidden_dims, args.use_batch_norm, q,
                                   args.floors_file)
    except (FileNotFoundError, KeyError) as e:
        p.error(str(e).strip("'\""))


def with_flags(message: str) -> str:
    """`message`, followed by the command-line options of the fields it names."""
    import re
    named = [f for f in FIELD_FLAGS if re.search(rf"\b{f}\b", message)]
    if not named:
        return message
    return message + "\n  (" + ", ".join(f"{f} is {FIELD_FLAGS[f]}" for f in named) + ")"


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Shows the defaults, except for on/off flags, whose default is just 'off'."""

    def _get_help_string(self, action):
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return action.help or ""
        return super()._get_help_string(action)
