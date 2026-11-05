# riemannian-fl-eeg

Federated learning of SPDNet and EEGNet for motor-imagery EEG decoding (MOABB datasets).
Personalisation and record-level differential privacy are independent options.

* **Models.** SPDNet (BiMap weights on the Stiefel manifold, optional SPD batch norm)
  and EEGNet (optional BatchNorm).
* **Aggregation.** SPDNet: ProjAvg, RLAvg or RetAvg. EEGNet: FedAvg.
* **Personalisation.** Personal head (FedPer), head re-fit after training, batch norm
  kept on each client (FedBN), local-only training.
* **Privacy.** DP-RFedProj: per-sample tangent clipping, Gaussian noise and an f-DP
  accountant.
* **Baseline.** Tangent space + logistic regression (TS+LR), per client or pooled.

The code merges the code of two papers, which keep their frozen repositories:
*Projection-Based Federated Learning on Compact Submanifolds with Geometry-Independent
Differential Privacy* (arXiv:2610.33712, repository `dp-rfedproj`) and *How much
representation learning does federated EEG decoding need? The case of SPDNet*
(arXiv:2609.29037, repository `icassp-spdnet-fl`).

## Installation

Python 3.11 or 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch                       # see pytorch.org for a CUDA build
pip install -r requirements-dev.txt     # includes yetanotherspdnet at a pinned commit
pip install -e . --no-deps
```

## Quick start (CPU, minutes, no download)

```bash
python -m rfl.data.synthetic --n-clients 4 --samples-per-client 200 --n-channels 12 \
    --n-classes 2 --mode iid --signal-strength 1.0 --output results/synth.npz
COMMON="--synthetic-clients results/synth.npz --model spdnet --hidden-dim 6 \
    --rounds 10 --batch-size 32 --learning-rate 1e-2 --experiment-name demo"
python -m rfl.experiments.run_federated_experiment $COMMON                     # shared model
python -m rfl.experiments.run_federated_experiment $COMMON --local-fc          # personal head
python -m rfl.experiments.run_federated_experiment $COMMON --dp-enabled        # private
python -m rfl.analysis.report results/demo --by scope eps
```

## On MUST (HTCondor)

Everything lives on the shared filesystem: code, environment, data, results.

**Once.** In a VS Code terminal connected to `lappusmb.in2p3.fr`:

```bash
cd /uds_data/<lab>/<login>
git clone https://github.com/thibault-pautrel/riemannian-fl-eeg.git
cd riemannian-fl-eeg
bash condor/setup_must.sh               # .venv, dependencies, fast tests
```

**In every terminal.** `source condor/env.sh` activates the environment and sets
`RFL_RESULTS` (results), `MNE_DATA` (data, `../mne_data`) and the other paths.

**Data.** Downloaded on the user interface, since the jobs have no network. The same
command writes the preprocessed trials to a cache that the jobs read
([docs/data.md](docs/data.md#trial-cache)):

```bash
nohup bash condor/fetch_data.sh BNCI2014_001 > fetch.log 2>&1 &
```

**Check the cluster.** Synthetic runs, a small DAG, then five runs on the downloaded
data. They use the flash class of MUST: at most one hour per job, on reserved slots of
one core and 2GB.

```bash
export RFL_DATASET=BNCI2014_001
R="--flash --cpus 1 --memory 2GB"
bash condor/submit.sh smoke $R
bash condor/submit_dag.sh smoke_auto $R
bash condor/submit.sh smoke_real $R
python -m rfl.experiments.grid smoke_real --status
```

**One dataset, both studies, one DAG.** The DAG chains every step, selection and grid
extension. Nothing is left to do by hand until the analysis.

```bash
export RFL_DATASET=BNCI2014_001
bash condor/submit_dag.sh mi_study --cpus 1 --memory 4GB
python -m rfl.experiments.pipeline status mi_study      # needs RFL_DATASET in the shell
condor_q -dag
```

Submitting the same command again resumes the study. Finished runs are kept. To check
each step before the next, `--step N` submits one step only
([docs/cluster.md](docs/cluster.md#step-by-step)). Memory to
request, to check with `condor_history -af MemoryUsage RequestMemory` (MUST stops a job
above 1.5 times its request):

| Dataset | Subjects | Channels | `--memory` |
|---|---|---|---|
| BNCI2014_001 | 9 | 22 | 4GB |
| Weibo2014 | 10 | 60 | 8GB |
| PhysionetMI | 106 | 64 | 8GB |
| Cho2017 | 52 | 64 | 12GB |
| Schirrmeister2017 | 14 | 128 | 32GB |

Do not edit or `git pull` the code while a DAG runs: the jobs read it from this folder.

## The two studies

**SPD batch norm** (`mi_spdnet_bn`). SPDNet without batch norm, with FedBN or with a
shared batch norm, with a shared or personal head, on subject and IID clients, plus a
centralised reference: 18 arms. Protocol, the same for every arm:

1. ReEig floors from the covariance spectrum, then a centralised calibration of the
   architecture, floor percentile and running-mean momentum (`mi_spdnet_bn_calib`).
2. Learning rate x batch size, then the horizon (shortest within one standard error of
   the best), on 3 seeds. A choice on the edge of its grid extends the grid by itself.
3. Final runs on 10 other seeds, paired by seed across arms.

**Data quantity** (`mi_datasize`, `_eegnet`, `_tslr`). Each subject keeps N
calibration trials per class. In `U<N>` every subject has N, in `H<N>` a third of the
subjects has N and the others all their trials. Every pair of head and normalisation
is an arm, for SPDNet and EEGNet (both plain SGD), tuned on its own and selected on the
subjects with N trials:

| Head | Option | | Normalisation | SPDNet | EEGNet |
|---|---|---|---|---|---|
| shared (+ re-fit) | (none) | | none | (none) | `--eegnet-norm-type none` |
| personal | `--local-fc` | | FedBN | `--use-batch-norm` | `--eegnet-norm-type batch` |
| local | `--local-only` | | shared | `... --no-fedbn` | `... --no-fedbn` |

The batch-norm arms also set their running statistics exactly after training
(`--bn-recalibrate`). TS+LR runs on the same subjects and trials.

**Analysis.**

```bash
python -m rfl.analysis.report $RFL_RESULTS/mi_BNCI2014_001_bn/final \
    --by clients scope arch --metric client_mean_acc
python -m rfl.analysis.datasize $RFL_RESULTS/mi_BNCI2014_001_datasize*/final \
    --out datasize_BNCI2014_001 --pairs
```

The second command writes a table of both models and one figure per model: accuracy
against N, one curve per arm, colour for the head, line style for the normalisation.

## Documentation

| File | Content |
|---|---|
| [docs/cluster.md](docs/cluster.md) | MUST in detail: scripts, plans, monitoring and fixes, writing a plan |
| [docs/options.md](docs/options.md) | Every option: personalisation, privacy, SPD batch norm, data quantity, checks, training protocol, depth |
| [docs/data.md](docs/data.md) | Datasets and preprocessing, adding a dataset, output files, analysis tools |
| [docs/design.md](docs/design.md) | How the code is organised, equivalence with the paper repositories, limitations |

## Repository structure

| Path | Content |
|---|---|
| `rfl/models/` | SPDNet, EEGNet, batch-norm recalibration |
| `rfl/data/` | MOABB loading, registry, data reduction, synthetic data |
| `rfl/federated/` | Client, server, aggregation, personalisation, DP (`privacy/`) |
| `rfl/experiments/` | Runners, TS+LR, plans and their machinery (`grid.py`, `protocol.py`, `pipeline.py`) |
| `rfl/analysis/` | Result tables, paired tests, data-quantity figures |
| `condor/` | HTCondor scripts for MUST |
| `tests/` | Tests (pytest) |

## Tests

```bash
python -m pytest -q -m "not slow"     # a few minutes, no download
python -m pytest -q                   # full suite, about seven minutes
```

GitHub Actions runs the fast tests on every push.

## Citation and license

See `CITATION.cff`. MIT license (`LICENSE`). SPD layers from
[yetanotherspdnet](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet),
datasets from [MOABB](https://moabb.neurotechx.com/) and [MNE-Python](https://mne.tools/),
EEGNet from Lawhern et al. (2018).
