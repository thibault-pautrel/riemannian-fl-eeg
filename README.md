# riemannian-fl-eeg

Federated learning of SPDNet and EEGNet for motor-imagery EEG decoding on
[MOABB](https://moabb.neurotechx.com/) datasets, with personalisation and differential
privacy.

Code of two papers, whose own repositories stay frozen: *Projection-Based Federated
Learning on Compact Submanifolds with Geometry-Independent Differential Privacy*
(arXiv:2610.33712) and *How much representation learning does federated EEG decoding
need? The case of SPDNet* (arXiv:2609.29037).

## Installation

Python 3.11 or 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

## Quick start (synthetic data, a few minutes on CPU)

```bash
python -m rfl.data.synthetic --n-clients 4 --samples-per-client 200 --n-channels 12 \
    --n-classes 2 --mode iid --signal-strength 1.0 --output results/synth.npz
COMMON="--synthetic-clients results/synth.npz --model spdnet --hidden-dim 6 \
    --rounds 10 --batch-size 32 --learning-rate 1e-2 --experiment-name demo"
python -m rfl.experiments.run_federated_experiment $COMMON                 # shared model
python -m rfl.experiments.run_federated_experiment $COMMON --local-fc      # personal head
python -m rfl.experiments.run_federated_experiment $COMMON --dp-enabled    # private
python -m rfl.analysis.report results/demo --by scope eps
```

## Real EEG

```bash
python -m rfl.experiments.fetch_dataset --dataset BNCI2014_001     # download once
python -m rfl.experiments.run_federated_experiment --dataset BNCI2014_001 --model spdnet \
    --rounds 200 --batch-size 32 --learning-rate 0.01 --local-fc
python -m rfl.experiments.run_federated_experiment --dataset BNCI2014_001 --model eegnet \
    --eegnet-norm-type batch --rounds 200 --batch-size 32 --learning-rate 0.01
python -m rfl.experiments.run_centralized_experiment --dataset BNCI2014_001 --model spdnet
python -m rfl.experiments.run_tslr --dataset BNCI2014_001 --mode local     # TS+LR baseline
```

Datasets: BNCI2014_001, Weibo2014, Cho2017, Schirrmeister2017, PhysionetMI, Stieger2021,
Lee2019_MI. One client per subject by default. Data go to `$MNE_DATA` (default
`~/mne_data`). `--help` lists every option of a runner.

## Main options

| Option | Effect |
|---|---|
| `--model spdnet` / `eegnet` | SPDNet (BiMap weights on the Stiefel manifold) or EEGNet |
| `--aggregation projavg` / `rlavg` / `retavg` | SPDNet aggregation (EEGNet: FedAvg) |
| `--local-fc` | personal classifier head (FedPer) |
| `--local-only` | each client trains alone |
| `--head-refit-epochs 25,50,100` | each client re-fits its head after training |
| `--use-batch-norm` | SPD batch norm (EEGNet: `--eegnet-norm-type batch`), kept on each client unless `--no-fedbn` |
| `--client-partition iid` | IID clients instead of one subject per client |
| `--train-trials-per-class N` | keep N training trials per class and subject |
| `--dp-enabled --dp-target-eps 10` | record-level differential privacy |

Details in [docs/options.md](docs/options.md).

## Reproducing the studies

A study is a plan in `rfl/experiments/plans/`. Its docstring gives the protocol. The
pipeline tunes every arm (learning rate x batch size, then horizon, on 3 seeds, grids
extended at their edges) and runs the final arms on 10 seeds.

| Plan | Question |
|---|---|
| `mi_spdnet_bn` | Does the SPD batch norm help federated SPDNet? |
| `mi_eegnet_bn` | The same for EEGNet |
| `mi_datasize` (`_eegnet`, `_tslr`) | From how many trials per class does federation beat local training? |
| `mi_study` | All of them, for one dataset |

```bash
export RFL_DATASET=BNCI2014_001
python -m rfl.experiments.pipeline run mi_study --jobs 8      # thousands of runs: use a cluster
python -m rfl.experiments.pipeline status mi_study
python -m rfl.analysis.report results/mi_BNCI2014_001_bn/final \
    --by clients scope arch --metric client_mean_acc
python -m rfl.analysis.stats results/mi_BNCI2014_001_bn/final --factor arch --by kind clients scope
python -m rfl.analysis.datasize results/mi_BNCI2014_001_datasize*/final --out datasize --pairs
```

On an HTCondor cluster, `bash condor/submit_dag.sh mi_study` runs the same pipeline as
one DAG (scripts and options in the headers of `condor/*.sh`).

## Tests

```bash
python -m pytest -q -m "not slow"     # a few minutes, no download
```

## Citation and license

See `CITATION.cff`. MIT license. SPD layers from
[yetanotherspdnet](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet),
EEGNet from Lawhern et al. (2018).
