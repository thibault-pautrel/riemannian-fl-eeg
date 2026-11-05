"""Smoke test: every major option on a small synthetic federation, a few minutes each.

Use it to check an installation (laptop, MUST user interface or worker nodes) before
launching real experiments. It needs no EEG download.

    python -m rfl.experiments.grid smoke --run --jobs 4      # here
    bash condor/submit.sh smoke                          # on HTCondor
    python -m rfl.experiments.grid smoke --status
    python -m rfl.analysis.report $RFL_RESULTS/smoke --by arch aggregation scope participation eps

Every job should finish. The accuracies are not meaningful (one seed, no tuning).
"""

import subprocess
import sys

from rfl.experiments.grid import REPO, grid, results_root


def _data():
    return results_root() / "smoke" / "synth.npz"


def prepare():
    """Generate the synthetic federation once (6 clients, 12 channels, 2 classes)."""
    DATA = _data()
    if DATA.exists():
        return
    DATA.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [sys.executable, "-m", "rfl.data.synthetic", "--n-clients", "6",
         "--samples-per-client", "300", "--n-channels", "12", "--n-classes", "2",
         "--mode", "noniid", "--signal-strength", "0.8", "--seed", "0",
         "--output", str(DATA)], cwd=str(REPO), stdout=subprocess.DEVNULL)


SPD = ["--model", "spdnet", "--device", "cpu"]
DIM = ["--hidden-dim", "6"]               # one block. The BN arms use --hidden-dims.
DP = ["--dp-enabled", "--dp-target-eps", "10"]
REFIT = ["--head-refit-epochs", "10,25", "--head-refit-weight-decays", "0,1e-3"]
BN = ["--hidden-dims", "8,6", "--epsilons", "0.01,0.01", "--use-batch-norm"]


def _arms():
    DATA = _data()
    SRC = ["--synthetic-clients", str(DATA)] + SPD
    FED = SRC + DIM
    return {
        # name: (entry, extra arguments)
        "central": ("central", ["--synthetic-clients", str(DATA)] + SPD + DIM),
        "central_rnd": ("central", ["--synthetic-clients", str(DATA), "--freeze-trunk"] + SPD + DIM),
        "shared": ("fed", FED),
        "shared_refit": ("fed", FED + REFIT),
        "fedper": ("fed", FED + ["--local-fc"]),
        "fedper_frozen_refit": ("fed", FED + ["--local-fc", "--local-head-mode", "frozen"]
                                + REFIT),
        "fedper_rnd": ("fed", FED + ["--local-fc", "--freeze-trunk"]),
        "local": ("fed", FED + ["--local-only"]),
        "rlavg": ("fed", FED + ["--aggregation", "rlavg"]),
        "half_participation": ("fed", FED + ["--client-fraction", "0.5"]),
        "dp_shared": ("fed", FED + DP),
        "dp_fedper": ("fed", FED + DP + ["--local-fc"]),
        "dp_shared_refit": ("fed", FED + DP + REFIT),
        "dp_retavg": ("fed", FED + DP + ["--aggregation", "retavg"]),
        "bn_fedbn": ("fed", SRC + BN),
        "bn_shared": ("fed", SRC + BN + ["--no-fedbn"]),
        "bn_fedbn_fedper": ("fed", SRC + BN + ["--local-fc"]),
    }


def _all():
    jobs = []
    for arm, (entry, args) in _arms().items():
        lr = 5e-2 if arm.startswith("dp_") else 1e-2
        jobs += grid(arm, entry, args, seeds=[1], lrs=[lr], batch_sizes=[64],
                     horizons=[20])
    return jobs


STAGES = {"all": _all}
