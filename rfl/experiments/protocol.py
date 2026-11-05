"""The recommended protocol as a reusable object: sweep, select, final runs.

A `Protocol` describes one study on one dataset: which arms to compare, which values to
sweep for each, how to search them, and which seeds to use. `stages()` returns the
`STAGES` of a plan. A new dataset, or a new search space, is then a plan of a few lines:

    # rfl/experiments/plans/cho2017.py
    from rfl.experiments.protocol import Protocol

    P = Protocol(dataset="Cho2017", eps=10,
                 space={"nodp": {"lr": [1e-3, 3e-3, 1e-2], "batch_size": [32],
                                 "horizon": [100, 200]},
                        "dp": {"lr": [1e-2, 3e-2, 1e-1], "batch_size": [64],
                               "horizon": [100, 200], "--dp-clip-C": [0.5, 1, 2]}},
                 model_space={"spdnet": {"--hidden-dim": [8, 16]}})
    NAME, STAGES = P.name, P.stages()

Arms (`kinds` selects a subset), for each model m and each aggregation rule a of
`aggregations[m]` (SPDNet: ProjAvg, RLAvg and RetAvg by default):

    central_<m>             centralised reference
    fed_<m>_<a>_shared      trunk and head shared (final stage: also a head re-fit)
    fed_<m>_<a>_fedper      personal head (FedPer)
    fed_<m>_local           local only (no aggregation, so one arm per model)
    dp<eps>_<m>_<a>_shared  shared, record-level DP at eps (final stage: also a re-fit)
    dp<eps>_<m>_<a>_fedper  personal head, DP

Variants (`variants`). Each arm above can come in several variants, for example with
and without a batch norm, or with subject and IID clients. A variant adds command-line
arguments, optionally a search space, and applies to some arm kinds only. Its name is
appended to the arm name (`fed_spdnet_projavg_fedper_bn`). The default is one unnamed
variant without arguments, which leaves the arms as listed above:

    variants={"nobn": Variant(),
              "bn": Variant(args=("--use-batch-norm",),
                            space={"--bn-momentum": [0.1, 0.3]}),
              "bnshared": Variant(args=("--use-batch-norm", "--no-fedbn"),
                                  kinds=("fed_shared", "fed_fedper"))}

Search space. `space["nodp"]` applies to the non-private arms and `space["dp"]` to the
private ones (under DP the batch size sets the sensitivity 2C/B, so the two spaces
differ). `model_space[m]` adds values swept for model m only, the space of a variant
adds values swept for its arms, and `arm_space[arm]` replaces the space of one arm. Keys: `lr`, `batch_size`, `horizon` (rounds, or epochs
for the centralised arm), or any runner option written as on the command line.

Search (`search`):

    "grid"     every combination, in one stage `sweep`
    "random"   `n_samples` combinations drawn at random from the grid, stage `sweep`
    "staged"   one stage per group of `stage_keys`, `sweep1`, `sweep2`, ... Stage k
               sweeps the keys of group k. The keys of earlier groups keep the values
               selected at stage k - 1, those of later groups their first listed value.
               For example stage_keys=[["lr", "horizon"], ["--dp-clip-C"]] tunes the
               learning rate and horizon at C = first value, then C at the chosen
               learning rate and horizon. The cost is the sum of the group sizes, not
               their product.

Then `final` runs the values selected at the last sweep stage on `final_seeds`
(disjoint from the sweep seeds). Results go under $RFL_RESULTS/<name>/, with
name = mi_<dataset> by default, so two datasets never share a folder or a selection.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rfl.experiments.grid import (RESULT_FILES, Job, grid, load_extensions, load_selection,
                                  results_root, selected, selection_path)

KINDS = ("central", "fed_shared", "fed_fedper", "fed_local", "dp_shared", "dp_fedper")
SEARCHES = ("grid", "random", "staged")

DEFAULT_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "nodp": {"lr": [1e-3, 3e-3, 1e-2, 3e-2], "batch_size": [64], "horizon": [100, 200]},
    "dp": {"lr": [1e-2, 3e-2, 1e-1], "batch_size": [128], "horizon": [100, 200]},
}

DEFAULT_AGGREGATIONS: Dict[str, Sequence[str]] = {
    "spdnet": ("projavg", "rlavg", "retavg"),
    "eegnet": ("fedavg",),
}

DEFAULT_REFIT = ("--head-refit-epochs", "25,50,100,200",
                 "--head-refit-weight-decays", "0,1e-3,1e-1")


@dataclass(frozen=True)
class Variant:
    """One variant of the arms of a Protocol (module docstring, Variants).

    Args:
        args: command-line arguments added to the arms of the variant.
        kinds: arm kinds the variant applies to (KINDS). None: every kind.
        space: values swept for the arms of the variant, on top of the space of their
            privacy level and model (`arm_space` still replaces everything).
    """
    args: Sequence[str] = ()
    kinds: Optional[Sequence[str]] = None
    space: Dict[str, List[Any]] = field(default_factory=dict)


# Options with no effect without aggregation (refused with --local-only).
LOCAL_ONLY_INERT = ("--aggregation", "--client-fraction", "--weight-by-samples",
                    "--no-fedbn")


def _central_options() -> set:
    """Option strings of the centralised runner."""
    from rfl.experiments.run_centralized_experiment import _build_arg_parser
    return {o for a in _build_arg_parser()._actions for o in a.option_strings}


def _jobs(arm: str, entry: str, args: Sequence[str], seeds: Sequence[int],
          space: Dict[str, List[Any]], sample: Optional[int] = None,
          sample_seed: int = 0) -> List[Job]:
    sp = dict(space)
    return grid(arm, entry, args, seeds, lrs=sp.pop("lr", None),
                batch_sizes=sp.pop("batch_size", None), horizons=sp.pop("horizon", None),
                space=sp, sample=sample, sample_seed=sample_seed)


@dataclass
class Protocol:
    """One study on one dataset (module docstring).

    Args:
        dataset: name in rfl/data/registry.py.
        eps: privacy budget of the DP arms (delta: runner default).
        models: models compared.
        aggregations: aggregation rules per model (DEFAULT_AGGREGATIONS).
        kinds: arm kinds included (KINDS).
        space, model_space, arm_space: search space (module docstring).
        search: 'grid', 'random' or 'staged' (module docstring).
        n_samples: combinations per arm with search='random'.
        stage_keys: groups of keys, one stage each, with search='staged'.
        sample_seed: seed of the random search.
        sweep_seeds, final_seeds: seeds of the sweep and final stages. Disjoint.
        common_args: arguments of every federated arm.
        model_args: extra arguments per model (for example a fixed --hidden-dim).
        variants: variants of the arms, by name (Variant, module docstring).
        refit_args: added to the shared federated arms in the final stage (head re-fit
            on each client, evaluated alongside the shared head). Empty to disable.
        name: results folder (default mi_<dataset>).
        data_args: the data arguments of every arm (default ['--dataset', dataset]),
            for example ['--synthetic-clients', 'synth.npz'].
    """
    dataset: str
    eps: float = 10.0
    models: Sequence[str] = ("spdnet", "eegnet")
    aggregations: Dict[str, Sequence[str]] = field(
        default_factory=lambda: dict(DEFAULT_AGGREGATIONS))
    kinds: Sequence[str] = KINDS
    space: Dict[str, Dict[str, List[Any]]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_SPACE.items()})
    model_space: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    arm_space: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    search: str = "grid"
    n_samples: Optional[int] = None
    stage_keys: Sequence[Sequence[str]] = ()
    sample_seed: int = 0
    sweep_seeds: Sequence[int] = (1, 2, 3)
    final_seeds: Sequence[int] = tuple(range(101, 111))
    common_args: Sequence[str] = ("--subjects-per-client", "1")
    model_args: Dict[str, Sequence[str]] = field(default_factory=dict)
    variants: Dict[str, Variant] = field(default_factory=lambda: {"": Variant()})
    data_args: Optional[Sequence[str]] = None
    refit_args: Sequence[str] = DEFAULT_REFIT
    name: Optional[str] = None

    def __post_init__(self):
        unknown = set(self.kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown arm kinds {sorted(unknown)}; known: {KINDS}")
        if set(self.sweep_seeds) & set(self.final_seeds):
            raise ValueError("sweep_seeds and final_seeds must be disjoint: the final "
                             "runs must not reuse the seeds used for selection")
        if self.search not in SEARCHES:
            raise ValueError(f"search must be one of {SEARCHES}, got {self.search!r}")
        if self.search == "random" and not self.n_samples:
            raise ValueError("search='random' needs n_samples")
        if self.search != "random" and self.n_samples:
            raise ValueError("n_samples only acts with search='random'")
        if self.search == "staged" and not self.stage_keys:
            raise ValueError("search='staged' needs stage_keys, for example "
                             "[['lr', 'horizon'], ['--dp-clip-C']]")
        if self.search != "staged" and self.stage_keys:
            raise ValueError("stage_keys only acts with search='staged'")
        flat = [k for g in self.stage_keys for k in g]
        if len(flat) != len(set(flat)):
            raise ValueError("a key appears in two groups of stage_keys")
        for m in self.models:
            if not self.aggregations.get(m):
                raise ValueError(f"no aggregation rule for model {m!r}")
        if not self.variants:
            raise ValueError("variants is empty: use {'': Variant()} for no variant")
        for vname, v in self.variants.items():
            if not re.fullmatch(r"[A-Za-z0-9]*(_[A-Za-z0-9]+)*", vname):
                raise ValueError(f"variant name {vname!r}: letters, digits and single "
                                 "underscores only (it becomes part of a folder name)")
            bad = set(v.kinds or ()) - set(KINDS)
            if bad:
                raise ValueError(f"variant {vname!r}: unknown kinds {sorted(bad)}")
            if not any(r["variant"] == vname for r in self._table()):
                raise ValueError(f"variant {vname!r} applies to no arm of this protocol "
                                 f"(kinds {self.kinds})")
        names = [r["arm"] for r in self._table()]
        dup = sorted({a for a in names if names.count(a) > 1})
        if dup:
            raise ValueError(f"two arms have the same name: {dup}")
        if self.name is None:
            self.name = f"mi_{self.dataset}"
        if self.search == "staged":         # every multi-valued key must be in a group
            seen_keys = set()
            for arm, _, args, priv in self.arms():
                sp = self.space_of(arm, self._model(args), priv)
                seen_keys |= set(sp)
                loose = [k for k, v in sp.items() if len(v) > 1 and k not in flat]
                if loose:
                    raise ValueError(f"arm {arm}: swept keys {loose} are in no group of "
                                     "stage_keys")
            unknown = [k for k in flat if k not in seen_keys]
            if unknown:
                raise ValueError(f"stage_keys {unknown} are in no search space (typo?); "
                                 f"keys of the spaces: {sorted(seen_keys)}")

    # ------------------------------------------------------------------ #
    def arms(self) -> List[tuple]:
        """(arm, entry, args, privacy) of every arm."""
        return [(r["arm"], r["entry"], r["args"], r["privacy"]) for r in self._table()]

    def _table(self) -> List[Dict[str, Any]]:
        """Every arm with its kind and variant (arms() without the variants)."""
        out = []
        for arm, entry, args, priv, kind in self._base_arms():
            for vname, v in self.variants.items():
                if v.kinds is not None and kind not in v.kinds:
                    continue
                out.append({"arm": f"{arm}_{vname}" if vname else arm, "entry": entry,
                            "args": list(args) + list(v.args), "privacy": priv,
                            "kind": kind, "variant": vname, "space": dict(v.space)})
        return out

    def _base_arms(self) -> List[tuple]:
        """(arm, entry, args, privacy, kind) of every arm kind, before the variants."""
        out = []
        ds = (list(self.data_args) if self.data_args is not None
              else ["--dataset", self.dataset])
        dp = ["--dp-enabled", "--dp-target-eps", f"{self.eps:g}"]
        tag = f"dp{self.eps:g}"
        for m in self.models:
            margs = ["--model", m, *self.model_args.get(m, ())]
            fed = ds + list(self.common_args) + margs
            if "central" in self.kinds:
                out.append((f"central_{m}", "central", ds + margs, "nodp", "central"))
            for agg in self.aggregations[m]:
                fa = fed + ["--aggregation", agg]
                table = {
                    "fed_shared": (f"fed_{m}_{agg}_shared", "fed", fa, "nodp"),
                    "fed_fedper": (f"fed_{m}_{agg}_fedper", "fed", fa + ["--local-fc"],
                                   "nodp"),
                    "dp_shared": (f"{tag}_{m}_{agg}_shared", "fed", fa + dp, "dp"),
                    "dp_fedper": (f"{tag}_{m}_{agg}_fedper", "fed",
                                  fa + dp + ["--local-fc"], "dp"),
                }
                out += [table[k] + (k,) for k in KINDS if k in table and k in self.kinds]
            if "fed_local" in self.kinds:
                out.append((f"fed_{m}_local", "fed", fed + ["--local-only"], "nodp",
                            "fed_local"))
        return out

    def space_of(self, arm: str, model: str, privacy: str) -> Dict[str, List[Any]]:
        """Search space of one arm.

        The central arm drops the options its runner does not have (DP, federation,
        personalisation), and the local-only arm the options that act on aggregation.
        """
        row = {r["arm"]: r for r in self._table()}[arm]
        if arm in self.arm_space:
            sp = dict(self.arm_space[arm])
        else:
            sp = dict(self.space[privacy])
            sp.update(self.model_space.get(model, {}))
            sp.update(row["space"])
        if row["kind"] == "central":
            # Options of the federated runner only (DP, federation, personalisation).
            known = _central_options()
            sp = {k: v for k, v in sp.items() if not k.startswith("--") or k in known}
        elif row["kind"] == "fed_local":
            sp = {k: v for k, v in sp.items() if k not in LOCAL_ONLY_INERT}
        return {k: list(v) for k, v in sp.items()}

    @staticmethod
    def _model(args: Sequence[str]) -> str:
        return list(args)[list(args).index("--model") + 1]

    def sweep_stages(self) -> List[str]:
        if self.search == "staged":
            return [f"sweep{k + 1}" for k in range(len(self.stage_keys))]
        return ["sweep"]

    # ------------------------------------------------------------------ #
    def _check_chain(self, arm: str, upto: int) -> None:
        """Staged search: the selections of stages 1..upto agree on the earlier keys.

        Stage k + 1 was built from the selection of stage k. If stage k was selected
        again afterwards, the later stages no longer follow from it.

        Raises:
            SystemExit: naming the stage to run and select again.
        """
        own = [i for i in self._stages_of(arm) if i < upto]
        for i, j in zip(own, own[1:]):
            prev = load_selection(self.name, arm, f"sweep{i + 1}")["hp"]
            cur = load_selection(self.name, arm, f"sweep{j + 1}")["hp"]
            for key in (k for g in self.stage_keys[:j] for k in g):
                if key in prev and cur.get(key) != prev[key]:
                    raise SystemExit(
                        f"arm {arm}: sweep{i + 1} selects {key}={prev[key]}, but "
                        f"sweep{j + 1} ran with {key}={cur.get(key)}. sweep{i + 1} was "
                        f"selected again after sweep{j + 1} ran: submit sweep{j + 1} again, "
                        "then select it.")

    def _stages_of(self, arm: str) -> List[int]:
        """Sweep stages (0-based) where `arm` runs.

        Every arm runs the first stage. With search='staged', an arm skips a later
        stage when none of the keys of that stage has two values or more in its space:
        its choice then stays the one of its previous stage (no rerun of the same cell).
        """
        if self.search != "staged":
            return [0]
        row = {r["arm"]: r for r in self._table()}[arm]
        sp = self.space_of(arm, self._model(row["args"]), row["privacy"])
        return [0] + [k for k in range(1, len(self.stage_keys))
                      if any(len(sp.get(key, ())) > 1 for key in self.stage_keys[k])]

    def _last_stage(self, arm: str) -> str:
        """Name of the last sweep stage `arm` runs, which holds its final choice."""
        return f"sweep{self._stages_of(arm)[-1] + 1}" if self.search == "staged" \
            else "sweep"

    def _failed(self, arm: str, stage: str) -> bool:
        """`arm` has no choice at `stage`: every cell lost a seed (diverged, abandoned).

        Such an arm is dropped from the later stages, with a warning, so that one arm
        never blocks the others.
        """
        path = selection_path(self.name, stage)
        if not path.exists():
            return False
        import json
        entry = json.loads(path.read_text()).get(arm, {})
        if "error" in entry:
            print(f"warning: arm {arm} dropped after {stage}: {entry['error']}",
                  file=sys.stderr)
            return True
        return False

    def sweep(self, k: int = 0) -> List[Job]:
        """Jobs of sweep stage k (0-based; only k = 0 unless search='staged')."""
        jobs = []
        for arm, entry, args, priv in self.arms():
            if self.search == "staged":
                own = self._stages_of(arm)
                if k not in own:
                    continue                  # nothing to sweep here (_stages_of)
                if any(self._failed(arm, f"sweep{i + 1}") for i in own if i < k):
                    continue
                if k > 1:
                    self._check_chain(arm, k)
            sp = self.space_of(arm, self._model(args), priv)
            if self.search == "staged":
                before = [i for i in own if i < k]
                prev = (load_selection(self.name, arm, f"sweep{before[-1] + 1}")["hp"]
                        if before else {})
                later = {key for g in self.stage_keys[k + 1:] for key in g}
                group = set(self.stage_keys[k])
                for key, values in sp.items():
                    if key in group:
                        continue
                    if key in later or key not in prev:
                        sp[key] = values[:1]             # first listed value
                    else:
                        sp[key] = [prev[key]]            # selected at the previous stage
            # Values added beyond the edge of the grid (rfl/experiments/pipeline.py).
            group = (set(self.stage_keys[k]) if self.search == "staged" else set(sp))
            for key, extra in load_extensions(self.name, self.sweep_stages()[k]).get(
                    arm, {}).items():
                if key in group and key in sp:
                    sp[key] = sp[key] + [v for v in extra if v not in sp[key]]
            sample = self.n_samples if self.search == "random" else None
            jobs += _jobs(arm, entry, args, self.sweep_seeds, sp, sample, self.sample_seed)
        return jobs

    def _check_selection_is_current(self, last: str) -> None:
        """The selection of the sweep stage `last` covers its current grid and results.

        Raises:
            SystemExit: when sweep jobs are unfinished (the grid was extended after the
                selection) or results are newer than the selection file.
        """
        stage_jobs = self.sweep(self.sweep_stages().index(last))
        for j in stage_jobs:
            j.stage, j.plan = last, self.name
        open_jobs = [j for j in stage_jobs if j.state() in ("pending", "queued", "running?")]
        if open_jobs and len(open_jobs) == len(stage_jobs):
            raise SystemExit(f"{last} has not run yet: run it and select it before the "
                             "final stage.")
        if open_jobs:
            raise SystemExit(
                f"{len(open_jobs)} job(s) of {last} are not finished (the grid may have "
                f"been extended after the selection): finish them, then run "
                f"`--stage {last} --select` again before the final stage.")
        sel_file = selection_path(self.name, last)
        newest = max(((j.save_dir / RESULT_FILES[j.entry]).stat().st_mtime
                      for j in stage_jobs if j.state() == "done"), default=0.0)
        if sel_file.stat().st_mtime < newest:
            raise SystemExit(f"results of {last} are newer than {sel_file}: run "
                             f"`--stage {last} --select` again before the final stage.")

    def final(self) -> List[Job]:
        stages = self.sweep_stages()
        last = stages[-1]
        failed = {arm for arm, *_ in self.arms()
                  if any(self._failed(arm, st) for st in
                         ([stages[i] for i in self._stages_of(arm)]
                          if self.search == "staged" else stages))}
        if self.search == "staged":
            for arm, *_ in self.arms():
                if arm not in failed:
                    self._check_chain(arm, len(stages))
        for st in sorted({self._last_stage(arm) for arm, *_ in self.arms()}):
            self._check_selection_is_current(st)
        jobs, edges = [], []
        for row in self._table():
            arm, entry, args = row["arm"], row["entry"], row["args"]
            if arm in failed:
                continue
            last = self._last_stage(arm)
            s = load_selection(self.name, arm, last)
            for k in self._stages_of(arm):        # edges of the keys tuned at each stage
                st = stages[k]
                sk = s if st == last else load_selection(self.name, arm, st)
                own = (set(self.stage_keys[k]) if self.search == "staged"
                       else set(sk.get("hp", {})))
                hit = [e for e in sk.get("on_grid_edge", [])
                       if e in own and e not in sk.get("at_bound", [])]
                if hit:
                    edges.append(f"{arm} ({st}): {', '.join(hit)}")
            if row["kind"] in ("fed_shared", "dp_shared"):
                args = list(args) + list(self.refit_args)
            jobs += selected(arm, entry, args, self.final_seeds, self.name, stage=last)
        if edges:
            print("warning: selections on the edge of their grid (extend the grid and "
                  "rerun the sweep?):\n  " + "\n  ".join(edges), file=sys.stderr)
        # Final runs of an earlier selection would be mixed with these in the reports.
        current = {(j.arm, j.hp_name) for j in jobs}
        root = results_root() / self.name / "final"
        stale = sorted(str(d) for a in (root.iterdir() if root.exists() else [])
                       if a.is_dir() for d in a.iterdir()
                       if d.is_dir() and (a.name, d.name) not in current)
        if stale:
            print(f"warning: {len(stale)} final folder(s) come from an earlier selection "
                  "and would be mixed with the new runs in the reports. Move them out "
                  "of the final stage:\n  " + "\n  ".join(stale[:5]), file=sys.stderr)
        return jobs

    def selection(self, arm: str) -> Dict[str, Any]:
        """The final choice of `arm` ('hp', 'mean', 'std', ...), after the checks of the
        final stage (finished, current, consistent across stages). For a plan that
        builds on the choices of another one (mi_spdnet_bn on mi_spdnet_bn_calib)."""
        if self.search == "staged":
            self._check_chain(arm, len(self.sweep_stages()))
        last = self._last_stage(arm)
        self._check_selection_is_current(last)
        return dict(load_selection(self.name, arm, last))

    def selected_hp(self, arm: str) -> Dict[str, Any]:
        """The values selected for `arm` (see `selection`)."""
        return dict(self.selection(arm)["hp"])

    def stages(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, name in enumerate(self.sweep_stages()):
            out[name] = (lambda k=k: self.sweep(k))
        out["final"] = self.final
        return out
