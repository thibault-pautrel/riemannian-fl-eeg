"""Defaults and option checks of the two runners (rfl/experiments/cli_checks.py)."""
import io
import contextlib
import warnings

import pytest

import rfl.experiments.run_centralized_experiment as C
import rfl.experiments.run_federated_experiment as F
from rfl.data.registry import get_subjects
from rfl.federated.fl_config import FLConfig
from rfl.utils.config import ExperimentConfig

BASE = ["--dataset", "BNCI2014_001", "--save-dir", "x"]
NAMES = ("run_name", "experiment_name")          # set by the runner


@pytest.mark.parametrize("runner", ["fed", "central"])
def test_runner_defaults_are_the_config_defaults(runner):
    """One source of truth: a bare command line gives the dataclass defaults."""
    if runner == "fed":
        _, a, _ = F.prepare_args(BASE)
        cli, cls = F.build_config(a).to_dict(), FLConfig
    else:
        _, a, src = C.prepare_args(BASE)
        cli, cls = C.config_from_args(a, src).to_dict(), ExperimentConfig
    ref = cls(dataset="BNCI2014_001", subjects=get_subjects("BNCI2014_001"),
              run_name=cli["run_name"]).to_dict()
    diff = {k: (cli[k], ref.get(k)) for k in cli if k not in NAMES and cli[k] != ref.get(k)}
    assert diff == {}


def _check(mod, extra):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mod.check_command(BASE + extra)
        except SystemExit:
            return err.getvalue()
        except ValueError as e:
            return str(e)
    return None


@pytest.mark.parametrize("extra,needle", [
    (["--dp-target-eps", "5"], "--dp-enabled"),
    (["--dp-clip-C", "2"], "--dp-enabled"),
    (["--bn-mean-type", "harmonic"], "--use-batch-norm"),
    (["--model", "eegnet", "--use-batch-norm"], "--model spdnet"),
    (["--model", "eegnet", "--hidden-dim", "8"], "--model spdnet"),
    (["--eegnet-norm-type", "batch"], "--model eegnet"),
    (["--head-refit-weight-decays", "0,1e-3"], "--head-refit-epochs"),
    (["--lr-plateau-patience", "5"], "plateau"),
    (["--lr-schedule", "none", "--lr-min", "1e-5"], "cosine or plateau"),
    (["--selection-metric", "val_loss"], "--report selected"),
    (["--cov-shrinkage", "0.2"], "lwf"),
    (["--cov-estimator", "lwf"], "shrinkage"),
    (["--hidden-dims", "16,8", "--epsilons", "1e-4,1e-2", "--hidden-dim", "12"],
     "--hidden-dims"),
    (["--epsilons", "1e-4,1e-2"], "--hidden-dims"),
    (["--local-only", "--aggregation", "rlavg"], "--local-only"),
    (["--local-only", "--client-fraction", "0.5"], "--local-only"),
    (["--no-fedbn"], "normalisation layer"),
    (["--fmin", "30", "--fmax", "20"], "fmin < fmax"),
    (["--fmax", "80"], "Nyquist"),
    (["--weight-decay", "-1"], "non-negative"),
    (["--early-stop-patience", "5"], "report_rule='last'"),
])
def test_options_without_effect_or_impossible_values_are_refused(extra, needle):
    msg = _check(F, extra)
    assert msg is not None and needle in msg, msg


@pytest.mark.parametrize("extra", [
    ["--dp-enabled", "--dp-target-eps", "5", "--dp-clip-C", "2"],
    ["--use-batch-norm", "--bn-mean-type", "harmonic", "--no-fedbn"],
    ["--model", "eegnet", "--eegnet-norm-type", "batch", "--no-fedbn"],
    ["--hidden-dims", "16,8", "--epsilons", "1e-4,1e-2"],
    ["--cov-estimator", "lwf", "--cov-shrinkage", "0.1"],
    ["--lr-schedule", "plateau", "--lr-plateau-patience", "5", "--selection-metric",
     "val_loss"],
    ["--local-fc", "--head-refit-epochs", "10,20", "--head-refit-weight-decays", "0,1e-3"],
    ["--report", "selected", "--early-stopping-patience", "10"],   # alias of the flag
])
def test_meaningful_combinations_are_accepted(extra):
    assert _check(F, extra) is None


def test_centralised_runner_has_the_same_checks_and_names():
    assert "--use-batch-norm" in _check(C, ["--bn-mean-type", "harmonic"])
    assert _check(C, ["--report", "selected", "--early-stop-patience", "5"]) is None
    assert _check(C, ["--lr-schedule", "step"]) is None


def test_validation_driven_choices_warn_under_dp():
    with pytest.warns(UserWarning, match="validation records"):
        F.check_command(BASE + ["--dp-enabled", "--report", "selected"])


# ---- checks against the dataset ------------------------------------------------
@pytest.fixture
def stats_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    return tmp_path


def _write_stats(per_subject):
    import pandas as pd

    from rfl.data.stats import update_stats
    rows = [{"subject": s, "event": e} for s, n in per_subject.items()
            for e in ("left_hand", "right_hand", "feet", "tongue") for _ in range(n)]
    update_stats("BNCI2014_001", pd.DataFrame(rows), 22)


def test_stats_file_merges_subjects(stats_dir):
    from rfl.data.stats import client_train_sizes, load_stats
    _write_stats({1: 10, 2: 20})
    _write_stats({3: 5})
    st = load_stats("BNCI2014_001")
    assert st["n_channels"] == 22 and set(st["trials"]) == {"1", "2", "3"}
    assert st["trials"]["2"]["feet"] == 20
    assert client_train_sizes("BNCI2014_001", [1, 2, 3], 1, "all_classes", 0.75) == \
        [30, 60, 15]
    assert client_train_sizes("BNCI2014_001", [1, 2, 3], 2, "binary", 0.5) == [30, 5]
    assert client_train_sizes("BNCI2014_001", [1, 9], 1, "binary", 0.5) is None


def test_bimap_dimensions_are_checked_against_the_channels(stats_dir):
    pytest.importorskip("moabb")
    assert "22 EEG channels" in _check(F, ["--hidden-dim", "40"])
    assert "must not exceed" in _check(F, ["--hidden-dims", "16,18",
                                           "--epsilons", "1e-4,1e-3"])
    assert _check(F, ["--hidden-dim", "22"]) is None


def test_dp_batch_size_is_checked_against_the_smallest_client(stats_dir):
    _write_stats({s: 100 for s in range(1, 10)} | {4: 40})   # subject 4: 160 trials
    # train fraction 0.75 -> about 120 records for subject 4
    assert "smallest client" in _check(F, ["--dp-enabled", "--batch-size", "200"])
    assert _check(F, ["--dp-enabled", "--batch-size", "64"]) is None
    assert _check(F, ["--batch-size", "200"]) is None                # no DP: no limit
    with pytest.warns(UserWarning, match="close to the smallest"):
        F.check_command(BASE + ["--dp-enabled", "--batch-size", "120"])


def test_mne_channel_counts_of_the_registry():
    pytest.importorskip("moabb")
    from rfl.data.stats import eeg_channel_count
    expected = {"BNCI2014_001": 22, "Weibo2014": 60, "Cho2017": 64,
                "Schirrmeister2017": 128, "PhysionetMI": 64, "Stieger2021": 62,
                "Lee2019_MI": 62}
    assert {k: eeg_channel_count(k) for k in expected} == expected


@pytest.mark.parametrize("mod", [F, C])
def test_print_config_shows_the_resolved_configuration(mod, capsys, tmp_path):
    import json
    mod.main(["--dataset", "BNCI2014_001", "--print-config", "--save-dir", str(tmp_path)])
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["spdnet_hidden_dim"] == 16 and cfg["spdnet_epsilon"] == 8e-5
    assert cfg["lr_min"] == pytest.approx(cfg["learning_rate"] / 100)
    assert not any(tmp_path.iterdir())                    # nothing was run


@pytest.mark.parametrize("extra,needle", [
    (["--dp-target-eps", "10", "--dp-clip-C", "1.0"], "--dp-enabled"),   # default values
    (["--head-refit-epochs", "0", "--head-refit-lr", "0.1"], "--head-refit-epochs"),
    (["--model", "eegnet", "--aggregation", "rlavg"], "Stiefel"),
    (["--aggregation", "fedavg"], "projavg, rlavg, retavg"),
    (["--subjects", "1", "1", "2"], "duplicates"),
    (["--task", "no_rest"], "no rest class"),
])
def test_more_runs_without_meaning_are_refused(extra, needle):
    msg = _check(F, extra)
    assert msg is not None and needle in msg, msg


def test_binary_task_on_a_two_class_dataset_is_refused():
    msg = _check(F, ["--task", "binary"])
    assert msg is None                                      # BNCI2014_001: 4 classes
    out = io.StringIO()
    with contextlib.redirect_stderr(out), pytest.raises(SystemExit):
        F.check_command(["--dataset", "Cho2017", "--task", "binary", "--save-dir", "x"])
    assert "no effect" in out.getvalue()


def test_synthetic_file_is_part_of_the_configuration(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    cfgs = []
    for f in (a, b):
        _, args, _ = F.prepare_args(["--synthetic-clients", str(f), "--save-dir", "x"])
        cfgs.append(F.build_config(args))
    assert cfgs[0].data_path == str(a) and cfgs[0].config_hash != cfgs[1].config_hash
    assert cfgs[0].run_name != cfgs[1].run_name


def test_dp_batch_check_allows_for_the_spread_of_the_split(stats_dir):
    """14 subjects of 100 trials: about 75 training records each, fewer for some seeds."""
    import pandas as pd

    from rfl.data.stats import update_stats
    rows = [{"subject": s, "event": e} for s in range(1, 10)
            for e in ("left_hand", "right_hand", "feet", "tongue") for _ in range(25)]
    update_stats("BNCI2014_001", pd.DataFrame(rows), 22)
    assert "smallest client" in _check(F, ["--dp-enabled", "--batch-size", "78"])
    with pytest.warns(UserWarning, match="as few as"):
        F.check_command(BASE + ["--dp-enabled", "--batch-size", "70"])
    assert _check(F, ["--dp-enabled", "--batch-size", "48"]) is None


def test_stats_file_is_found_without_mne_data(tmp_path, monkeypatch):
    from rfl.data.stats import stats_path
    monkeypatch.delenv("MNE_DATA", raising=False)
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert stats_path("Cho2017") == (tmp_path / "mne_data" / "rfl-stats" / "Cho2017.json").resolve()


@pytest.mark.parametrize("extra,needle", [
    (["--local-head-mode", "train"], "--local-fc"),
])
def test_head_mode_needs_a_personal_head(extra, needle):
    assert needle in _check(F, extra)


def test_synthetic_data_need_spdnet(tmp_path):
    out = io.StringIO()
    with contextlib.redirect_stderr(out), pytest.raises(SystemExit):
        F.prepare_args(["--synthetic-clients", str(tmp_path / "s.npz"), "--model", "eegnet"])
    assert "only SPDNet" in out.getvalue()


def test_device_is_recorded_as_used():
    import torch
    _, args, _ = F.prepare_args(BASE)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert F.build_config(args).device == expected


def test_config_errors_name_the_command_line_options():
    from rfl.experiments.cli_checks import with_flags
    msg = with_flags("fl_local_only needs fl_client_fraction = 1")
    assert "fl_local_only is --local-only" in msg and "--client-fraction" in msg
    assert with_flags("nothing to name") == "nothing to name"


def test_missing_trial_counts_warn_under_dp(stats_dir):
    from rfl.experiments.cli_checks import dataset_errors
    _, args, src = F.prepare_args(BASE + ["--dp-enabled"])
    errors, warns = dataset_errors(args, src)
    assert not errors and any("no trial counts" in w for w in warns)
    _, args, src = F.prepare_args(BASE)
    assert not any("no trial counts" in w for w in dataset_errors(args, src)[1])


def test_batch_norm_before_reeig_warns_about_the_floors():
    from rfl.experiments.cli_checks import dataset_errors
    _, args, src = C.prepare_args(BASE + ["--use-batch-norm", "--bn-position", "before_reeig"])
    assert any("Recalibrate" in w for w in dataset_errors(args, src)[1])
    _, args, src = C.prepare_args(BASE + ["--use-batch-norm"])
    assert not any("Recalibrate" in w for w in dataset_errors(args, src)[1])


@pytest.mark.parametrize("mod", [F, C])
def test_help_shows_no_default_for_on_off_flags(mod):
    text = " ".join(mod._build_arg_parser().format_help().split())
    assert "(default: False)" not in text and "(default: True)" not in text
    assert "(default: 0.001)" in text
