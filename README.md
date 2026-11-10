# riemannian-fl-eeg

Federated learning of SPDNet and EEGNet for motor-imagery EEG (MOABB datasets), with
personalisation and record-level differential privacy as independent options.

* **Models.** SPDNet (BiMap weights on the Stiefel manifold, optional SPD batch norm),
  EEGNet (optional BatchNorm).
* **Aggregation.** SPDNet: ProjAvg, RLAvg, RetAvg. EEGNet: FedAvg.
* **Personalisation.** Personal head (FedPer), head re-fit, FedBN, local-only training.
* **Privacy.** DP-RFedProj: tangent clipping, Gaussian noise, f-DP accountant.
* **Baseline.** Tangent space + logistic regression (TS+LR).

It merges the code of two papers, whose repositories stay frozen: *Projection-Based
Federated Learning on Compact Submanifolds with Geometry-Independent Differential
Privacy* (arXiv:2610.33712, `dp-rfedproj`) and *How much representation learning does
federated EEG decoding need? The case of SPDNet* (arXiv:2609.29037, `icassp-spdnet-fl`).

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

Code, environment, data and results all sit on the shared filesystem. Details in
[docs/cluster.md](docs/cluster.md).

```bash
# Once, in a VS Code terminal on lappusmb.in2p3.fr
cd /uds_data/<lab>/<login>
git clone https://github.com/thibault-pautrel/riemannian-fl-eeg.git && cd riemannian-fl-eeg
bash condor/setup_must.sh                # .venv, dependencies, fast tests

# In every terminal
source condor/env.sh                     # sets RFL_RESULTS, MNE_DATA, ...

# Data, on the user interface (the jobs have no network). Also fills the trial cache.
nohup bash condor/fetch_data.sh BNCI2014_001 > fetch.log 2>&1 &

# Check the cluster (flash class: 1 h, 1 core, 2 GB)
export RFL_DATASET=BNCI2014_001
R="--flash --cpus 1 --memory 2GB"
bash condor/submit.sh smoke $R
bash condor/submit_dag.sh smoke_auto $R
bash condor/submit.sh smoke_real $R
python -m rfl.experiments.grid smoke_real --status

# Both studies of one dataset, one DAG (resubmitting resumes it)
bash condor/submit_dag.sh mi_study --cpus 1 --memory 4GB
bash condor/submit_dag.sh mi_study --step 0 --flash      # or step N of `status` only
python -m rfl.experiments.pipeline status mi_study
condor_q -dag
```

| Dataset | Subjects | Channels | `--memory` |
|---|---|---|---|
| BNCI2014_001 | 9 | 22 | 4GB |
| Weibo2014 | 10 | 60 | 8GB |
| PhysionetMI | 106 | 64 | 8GB |
| Cho2017 | 52 | 64 | 12GB |
| Schirrmeister2017 | 14 | 128 | 32GB |

Do not edit or `git pull` the code while a DAG runs: the jobs read it from this folder.

## The studies

| Plans | Question | Arms |
|---|---|---|
| `mi_spdnet_bn_calib`, `mi_spdnet_bn` | Does the SPD batch norm help federated SPDNet? | none, FedBN, shared BN x shared head, FedPer, local x subject or IID clients, + centralised: 18 |
| `mi_eegnet_bn_calib`, `mi_eegnet_bn` | The same for EEGNet and its BatchNorm (FedAvg) | 18 |
| `mi_datasize`, `_eegnet`, `_tslr` | From how many trials per class does federation beat local training? | 8 per model and scenario, + TS+LR |

Every arm is tuned alike: learning rate x batch size, then the horizon, on 3 seeds, with
automatic grid extension. Final runs use 10 other seeds. Protocols in
[docs/options.md](docs/options.md).

```bash
python -m rfl.analysis.report $RFL_RESULTS/mi_BNCI2014_001_bn/final \
    --by clients scope arch --metric client_mean_acc
python -m rfl.analysis.stats $RFL_RESULTS/mi_BNCI2014_001_bn/final \
    --factor arch --by kind clients scope
python -m rfl.analysis.datasize $RFL_RESULTS/mi_BNCI2014_001_datasize*/final \
    --out datasize_BNCI2014_001 --pairs
```

## Documentation

| File | Content |
|---|---|
| [docs/cluster.md](docs/cluster.md) | MUST: scripts, plans, DAGs, monitoring, writing a plan |
| [docs/options.md](docs/options.md) | Options and study protocols: personalisation, privacy, batch norm, data quantity, training |
| [docs/data.md](docs/data.md) | Datasets, trial cache, adding a dataset, outputs, analysis |
| [docs/design.md](docs/design.md) | Code organisation, adding a Stiefel mean, changes from the papers, limitations |

## Repository structure

| Path | Content |
|---|---|
| `rfl/models/` | SPDNet, EEGNet, batch-norm recalibration |
| `rfl/data/` | MOABB loading, registry, trial cache, data reduction, synthetic data |
| `rfl/federated/` | Client, server, aggregation, personalisation, DP (`privacy/`) |
| `rfl/geometry/` | Stiefel and Grassmann tools |
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
