# riemannian-fl-eeg

Federated learning of SPDNet and EEGNet for motor-imagery EEG decoding. Privacy and
personalisation are two independent options. Each one can be switched on or off, and
every combination the code accepts keeps the privacy guarantee.

* **Models.** SPDNet (BiMap weights on the Stiefel manifold, one or more BiRe blocks)
  and EEGNet (with or without BatchNorm).
* **Aggregation.** For SPDNet, ProjAvg (polar projection of the average), RLAvg
  (lifting to the tangent space) and RetAvg (inverse retraction, the PriRFed baseline).
  For EEGNet, FedAvg. No other pairing is accepted.
* **Personalisation.** Personal classifier head (FedPer), trained during the rounds or
  frozen then re-fitted, post-training head re-fit (fine-tuning), local-only baseline,
  frozen random trunk, FedBN for BatchNorm layers.
* **Privacy.** Record-level differential privacy (DP-RFedProj): per-sample tangent
  clipping, Gaussian noise, minibatch subsampling, and an f-DP accountant that calibrates
  the noise. The PriRFed accountant is included as a baseline.

This repository merges the code of two papers into one base for further work.

| Paper | Frozen code of the paper |
|---|---|
| *Projection-Based Federated Learning on Compact Submanifolds with Geometry-Independent Differential Privacy* (TMLR, under review, arXiv:2610.33712) | `dp-rfedproj` |
| *How much representation learning does federated EEG decoding need? The case of SPDNet* (ICASSP 2027, submitted, arXiv:2609.29037) | `icassp-spdnet-fl` |

With the library version of the papers, this code gives the same numbers as the paper
repositories, bit for bit, on the settings of both papers. The newer library pinned
here, and a frozen SPDNet trunk, change them by float32 rounding only. A few corrections
made after the merge change some runs (see
[Equivalence with the paper repositories](#equivalence-with-the-paper-repositories)).

## Contents

1. [How the code is organised, in five ideas](#how-the-code-is-organised-in-five-ideas)
2. [Installation](#installation) and [quick start](#quick-start-cpu-a-few-minutes-no-download)
3. [Running on MUST (HTCondor)](#running-on-must-htcondor)
4. [Options](#options): personalisation, privacy, SPD batch norm, training protocol, depth
5. [Data](#data), [outputs](#outputs), [analysis](#analysis)
6. [Tests](#tests), [equivalence with the paper repositories](#equivalence-with-the-paper-repositories), [limitations](#limitations)

## How the code is organised, in five ideas

**1. A client is a group of subjects.** The trials of a MOABB dataset are split once into
train, validation and test (75/10/15, stratified, seeded by `--seed`). Each client keeps
the trials of its subjects (`--subjects-per-client`, 1 by default). A centralised run
at the same seed uses exactly the same trials, so the two are directly comparable.

**2. A round is broadcast, local training, aggregation, evaluation.** The server sends
the global model, each sampled client trains for `--local-epochs` on its records, the
server aggregates the uploads (`rfl/federated/aggregation.py`), and every client evaluates
the new model on its validation and test records. The main metrics pool the
predictions of all clients, like a centralised run.

**3. SPDNet weights live on a manifold.** A BiMap weight W is semi-orthogonal
(W^T W = I, a point of the Stiefel manifold). The plain average of such points leaves
the manifold, so SPDNet never uses FedAvg. ProjAvg projects the average back with the
polar factor (the default for SPDNet). RLAvg and RetAvg average in a tangent space
instead. EEGNet is Euclidean and uses FedAvg. On each client, W stays on the manifold
by a polar projection after each step (`--stiefel projection`, the default and the only
choice under DP) or by the library's orthogonal parametrization (`--stiefel
trivialization`, as in Gallet et al.). The aggregation sees W in both cases.

**4. Personalisation decides what stays on the client.** A personal head (`--local-fc`)
is never uploaded. Normalisation layers stay local by default (FedBN). `--local-only`
keeps everything local. A head re-fit after training adapts the head to each client.

**5. Privacy changes the local step, nothing else.** With `--dp-enabled`, each local step
clips the per-sample gradients, averages them and adds Gaussian noise, with sigma
calibrated before training so that the whole run meets (eps, delta). Every parameter
that trains goes through this step, a personal head included, so everything a client
sends is covered.

The code is one package, `rfl`. `rfl/experiments/` holds the entry points, run with
`python -m rfl.experiments.<name>`, and the other subpackages the library code (see
[Repository structure](#repository-structure)). Installed with `pip install -e .`, it
runs from any folder.

## Installation

Python 3.11 or 3.12 (yetanotherspdnet needs Python 3.11 or later). A GPU is optional.
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 1. PyTorch for your platform, see https://pytorch.org. For example:
pip install torch                                                              # CPU, or Apple Silicon
# pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121  # Linux with CUDA 12.1

# 2. Everything else, including yetanotherspdnet at a pinned commit, and pytest
pip install -r requirements-dev.txt

# 3. This repository, editable (an edit of the code applies at once)
pip install -e . --no-deps
```

The commands of this README run from any folder once step 3 is done. Results go to
`./results` of the current folder by default.

`requirements.txt` pins yetanotherspdnet to commit `5c08ebc` (21 October 2026), which
brings the refactored SPD batch norm (`norm_and_bias_mode`) and a symmetrised eigh
gradient. `environment/requirements.lock.linux-py312-cpu.txt` is the environment of the
tests of this repository. The two other lock files are the environments of the paper
runs (Linux GPU cluster and Intel Mac), with the earlier library commit `5b25c30`.

## Quick start (CPU, a few minutes, no download)

A small synthetic SPD federation, then the same run with each option.

```bash
python -m rfl.data.synthetic --n-clients 4 --samples-per-client 200 --n-channels 12 \
    --n-classes 2 --mode iid --signal-strength 1.0 --output results/synth.npz

COMMON="--synthetic-clients results/synth.npz --model spdnet --hidden-dim 6 \
    --rounds 10 --batch-size 32 --learning-rate 1e-2 --device cpu --experiment-name demo"

python -m rfl.experiments.run_federated_experiment $COMMON                                  # shared model
python -m rfl.experiments.run_federated_experiment $COMMON --local-fc                       # personal head
python -m rfl.experiments.run_federated_experiment $COMMON --dp-enabled --dp-target-eps 10  # private
python -m rfl.experiments.run_federated_experiment $COMMON --local-fc --dp-enabled          # both

python -m rfl.analysis.report results/demo --by scope eps
```

This demo checks the pipeline. The private runs stay near chance here because they use
the learning rate of the non-private ones: add `--learning-rate 1e-1` to them and they
reach 0.94 to 0.98 test accuracy, against 0.98 to 0.99 without DP. Each setting needs its
own learning rate, which is why the study protocol (section
[4](#4-run-a-study-sweep-select-final)) tunes it per arm.

Each run goes to `results/<experiment-name>/<run-name>`. The default run name encodes
the model, the aggregation, the data, the personalisation, the privacy budget, the seed
and an identifier of the other settings, for example
`spdnet_projavg_synthetic_fedper_eps10_seed42_d0a30e2e`. Two identifiers appear in a
run: `config_id` (in the run name and the analysis tables) covers every setting except
the seed, so the seeds of one setting share it. `config_hash` (in `config.json` and
the checkpoints) also covers the seed and identifies one run.

## Running on MUST (HTCondor)

[MUST](https://doc.must-datacentre.fr/) is the LAPP / Université Savoie Mont Blanc
computing centre. Jobs run on worker nodes under HTCondor. Everything they need (the
repository, the Python environment, the EEG data and the results) sits on the shared
filesystem, so nothing is copied to the nodes. The folder `condor/` holds the scripts:

| File | Role |
|---|---|
| `condor/env.sh` | Paths of the repository, the environment, the data and the results. Edit once. |
| `condor/setup_must.sh` | Creates the Python environment and runs the fast tests. Once. |
| `condor/fetch_data.sh` | Downloads MOABB datasets. Once per dataset, on the user interface. |
| `condor/submit.sh` | Submits the pending runs of a plan, one HTCondor job per run. |
| `condor/run_one.sh` | What each job executes: one run, skipped if already done. |
| `condor/rfl.sub` | The submit description used by `submit.sh`. |
| `condor/submit_dag.sh` | Runs a whole study as one DAG: every stage, selection and grid extension (`rfl/experiments/pipeline.py`). |
| `condor/dag_step.sh`, `condor/run_cmd.sh` | The PRE/POST scripts of the DAG nodes, and the job of a command step. |

A **plan** (`rfl/experiments/plans/*.py`) lists the runs of a study. `rfl/experiments/grid.py`
turns a plan into jobs, checks them, runs or monitors them, and selects
hyperparameters. The plans provided: `smoke` (the main options on synthetic data,
minutes), `smoke_auto` (the automatic pipeline on synthetic data, minutes), `mi_protocol` (a motor-imagery study with the recommended protocol),
`mi_spdnet_bn_calib` then `mi_spdnet_bn` (the SPD batch norm in SPDNet: none, FedBN or shared, with a shared or
personal head, on subject and IID clients), `synth_bn_calib` then `synth_bn` (the same
study on a small synthetic federation, about 2 h on 2 cores), `mi_datasize` with
`mi_datasize_eegnet` and `mi_datasize_tslr` (federated against local training as the
number of calibration trials falls, see [Data quantity](#data-quantity)),
`synth_datasize` with `synth_datasize_tslr` (the same on synthetic subjects), `mi_study`
(both studies of one dataset in one DAG) and `template_new_dataset` (a study with its own search space, to copy).

### 1. Connect with VS Code

Install the *Remote - SSH* extension, then add to `~/.ssh/config` on your laptop:

```
Host must
    HostName lappusmb.in2p3.fr       # lappui.in2p3.fr for LAPP accounts
    User <your_login>
```

In VS Code, run *Remote-SSH: Connect to Host...* and choose `must`. The terminal of the
VS Code window is then a shell on the user interface. The user interface is for
editing, setting up and submitting, not for long computations: those go to HTCondor.
VS Code installs its server in your home directory. If your home quota is small, the
setting `remote.SSH.serverInstallPath` can point it to the shared filesystem.

### 2. Set up once

```bash
cd /uds_data/<lab>/<login>                 # the shared filesystem (or /lapp_data/...)
git clone https://github.com/thibault-pautrel/riemannian-fl-eeg.git    # public repository
cd riemannian-fl-eeg
nano condor/env.sh                          # check RFL_VENV, RFL_RESULTS, MNE_DATA
bash condor/setup_must.sh                   # Python environment + fast tests
# The datasets you need. Long (hours for several datasets): in the background, so that
# closing VS Code does not stop it. Rerun until no subject is missing.
nohup bash condor/fetch_data.sh BNCI2014_001 Cho2017 > fetch.log 2>&1 &
tail -f fetch.log
```

By default `condor/env.sh` puts the environment in `riemannian-fl-eeg/.venv`, the
results in `riemannian-fl-eeg/results` and the data in `../mne_data`, all on the
shared filesystem. `setup_must.sh` needs Python 3.11 or later. If the user interface
only has an older Python, its header shows how to get one with Miniforge.

Every new terminal loads the environment with `source condor/env.sh`. It activates
the Python environment and sets `$RFL_RESULTS`, which the commands below use.

### 3. Check the cluster with the smoke plan

```bash
bash condor/submit.sh smoke                  # 17 jobs, a few minutes each
condor_q                                     # your jobs: idle, running, held
python -m rfl.experiments.grid smoke --status    # done / failed / pending
python -m rfl.analysis.report $RFL_RESULTS/smoke --by arch aggregation scope participation eps
```

Every job should end as `done`. The accuracies of the smoke plan mean nothing (one
seed, no tuning). It only checks that the main options run on the worker nodes. It has
no EEGNet, no real data, no TS+LR and no data reduction.

### 4. Run a study: sweep, select, final

`mi_protocol` follows the protocol recommended in [Training protocol](#training-protocol):
the hyperparameters are chosen on validation runs with 3 sweep seeds, then the chosen
values are run on 10 other seeds, which give the reported numbers. It compares 20 arms:
SPDNet with ProjAvg, RLAvg and RetAvg, and EEGNet with FedAvg, each shared and FedPer,
with and without DP, plus a centralised and a local-only reference per model.

```bash
export RFL_DATASET=BNCI2014_001   # the default. Keep it set for every command below.

# 0. Every run of the stage is a valid command line (seconds, nothing runs)
python -m rfl.experiments.grid mi_protocol --stage sweep --check

# 1. Sweep: 20 arms x (lr, T) grid x 3 sweep seeds = 432 runs
bash condor/submit.sh mi_protocol sweep --memory 16GB
python -m rfl.experiments.grid mi_protocol --stage sweep --status

# 2. Select, when every run is done: the best values of each arm on the mean
#    validation accuracy over the sweep seeds
python -m rfl.experiments.grid mi_protocol --stage sweep --select
# -> $RFL_RESULTS/mi_BNCI2014_001/sweep/selection.json. "EDGE" (grids of 3 values or
#    more) marks a value on the edge of its
#    grid: extend the grid and submit the sweep again (finished runs are skipped).

# 3. Final: the selected values of each arm on 10 other seeds = 200 runs
bash condor/submit.sh mi_protocol final --memory 16GB
python -m rfl.analysis.report $RFL_RESULTS/mi_BNCI2014_001/final --by kind arch aggregation scope eps
python -m rfl.analysis.report $RFL_RESULTS/mi_BNCI2014_001/final --metric refit_test_acc
python -m rfl.analysis.stats $RFL_RESULTS/mi_BNCI2014_001/final --factor scope --by arch eps
```

`RFL_DATASET` changes the dataset and `RFL_EPS` the privacy budget of the DP arms.
Each dataset has its own results folder (`$RFL_RESULTS/mi_<dataset>/`), and the DP arms carry
their budget in their name (`dp10_spdnet_rlavg_shared`), so studies never overwrite
each other. To sweep other hyperparameters (clip C, BiMap dimension, weight decay, ...),
write a plan (section 6).

#### The batch-norm study, automatically

One HTCondor DAG per dataset runs the whole study: ReEig floors, calibration, sweeps,
selections, grid extensions and final runs (`rfl/experiments/pipeline.py`, protocol in
[SPD batch norm](#spd-batch-norm), Study). Nothing is left to do by hand until the
reports. The floors go to `$RFL_RESULTS/reeig_floors/<dataset>.json`, the DAG and its
logs to `$RFL_RESULTS/condor/dag_<NAME>_<time>/`. `bash condor/submit_dag.sh <plan>
--dry-run` writes the DAG without submitting it. The study covers BNCI2014_001,
Cho2017, Weibo2014, PhysionetMI and Schirrmeister2017 (`STUDY` in
`mi_spdnet_bn_calib.py`). Download the data first (section 2): the jobs have no network.

```bash
export RFL_DATASET=Cho2017                  # recorded in the DAG; `status` reads your shell's
bash condor/submit_dag.sh mi_spdnet_bn --memory 12GB
python -m rfl.experiments.pipeline status mi_spdnet_bn     # step by step
condor_q -dag                                               # the DAG and its jobs
# when the last step is done:
python -m rfl.analysis.report $RFL_RESULTS/mi_Cho2017_bn/final --by clients scope arch
python -m rfl.analysis.report $RFL_RESULTS/mi_Cho2017_bn/final --by clients scope arch --metric client_mean_acc
python -m rfl.analysis.stats $RFL_RESULTS/mi_Cho2017_bn/final --factor arch --by kind clients scope
```

Each node of the DAG is one step. Its PRE script lists the jobs still to run, or skips
the node when there are none. Its job runs them. Its POST script selects the stage and,
when a choice lies on the edge of its grid, adds one value beyond it (the plan's
`EXTEND` rules, within their bounds). Failed jobs and new grid values run again, up to
`--retries` passes per node (20). A run whose training diverges (a learning rate too
large: non-finite loss, then a failed eigendecomposition) counts as a result: it
writes `"diverged": true` with NaN metrics, its cell is left out of the selection, and
it is not run again. A run that fails three times (`RFL_MAX_ATTEMPTS`) is abandoned:
its cell is left out too, and `grid --status` lists it. A run killed by HTCondor (memory
rule, eviction, a hold of more than an hour) writes no exit code: the POST script counts
it as failed, and gives it up after six starts. In a DAG every job exits 0 and the POST
script reads its `exit_code`, so one failing job never removes the others of its node.
An arm with no complete cell (every cell lost a seed) is dropped from the later stages
with a warning, and the other arms go on. Submitting the same DAG again is safe once the
first one has ended: finished steps are skipped and finished jobs are not rerun. One DAG
per dataset at a time, each with its own `RFL_DATASET`: they run side by side.
`mi_spdnet_bn` and `mi_datasize` share the calibration nodes: submit the second one once
the calibration of the first is finished, or submit `mi_study`, which runs both. `bash condor/submit_dag.sh smoke_auto --short`
checks the mechanism on the cluster in minutes, `bash condor/submit_dag.sh synth_bn
--short` the whole batch-norm protocol on synthetic data, and `python -m
rfl.experiments.pipeline run <plan> --jobs N` runs the same steps on a laptop.

#### Both studies of one dataset, in one DAG

`mi_study` chains the batch-norm study and the data-quantity study of one dataset. The
calibration runs once for both. EEGNet and TS+LR run side by side with the SPDNet chain,
since they need no calibration.

```bash
source condor/env.sh
export RFL_DATASET=BNCI2014_001
bash condor/submit_dag.sh mi_study --cpus 2 --memory 8GB
python -m rfl.experiments.pipeline status mi_study
```

Then the next dataset, in the same way, once its data are downloaded. The `--memory`
of each dataset is in the table of [Data](#data).

### 5. Monitor and fix

| Need | Command |
|---|---|
| Your jobs | `condor_q`, `condor_q -nobatch` |
| Why a job is held or was removed | `condor_q -hold`, `condor_history -l <job>` (MUST terminates a job whose memory exceeds 1.5 times its request: raise `--memory`) |
| Remove jobs | `condor_rm <cluster>` or `condor_rm <login>` |
| The log of one run | `$RFL_RESULTS/<NAME>/<stage>/<arm>/<hp>/seed<s>/run.log` |
| HTCondor logs | `$RFL_RESULTS/condor/<NAME>_<stage>/logs/` (NAME: the results folder of the plan, `<NAME>` alone for a plan run without a stage) |
| Resubmit the failed and interrupted runs (killed, evicted, removed), once `condor_q` shows no job of the plan | `bash condor/submit.sh <plan> <stage> --retry-failed` |
| Run one job by hand | copy a line of `$RFL_RESULTS/condor/<NAME>_<stage>/jobs_*.txt` after `bash condor/run_one.sh` |
| Rerun a finished run | `FORCE=1 bash condor/run_one.sh <line>` |

`submit.sh` only submits the runs that are neither done nor already queued, so
resubmitting a plan is safe. It first checks every command line and submits nothing if
one is invalid. `--status` counts the jobs by state:

| State | Meaning |
|---|---|
| `pending` | never submitted |
| `queued` | submitted by `submit.sh`, not started |
| `running?` | started, no exit code yet: running, or killed by HTCondor (memory above 1.5 times the request, eviction, `condor_rm`). The DAG scripts count it as failed, and as abandoned after 6 starts |
| `failed` | finished with a non-zero exit code, or without a result file |
| `abandoned` | failed 3 times (`RFL_MAX_ATTEMPTS`): not run again, left out of the selection (`FORCE=1` reruns it) |
| `done` | result file present, exit code 0 (a diverged run is done, with NaN metrics) |
| `conflict` | the folder holds a run of another command: the plan changed after the run. Give the plan a new `NAME`, or delete the folder |

`--select` refuses to run while jobs of the stage are not finished (`--allow-incomplete`
overrides), and selects on a validation metric present in the results. It adds its
arms to the selection file of the stage, so plans run under several `RFL_EPS` values
keep every selection. The final stage refuses a selection older than the sweep
results, or a sweep with unfinished jobs (a grid extended after the selection). It
warns about final folders left by an earlier selection, which reports would mix with the
new ones. In a staged search, it also checks that each stage was run with the current
selection of the stage before it.

`submit.sh` takes the plan, then the stage (or `--stage NAME`), then these options:

| Option | Default | Meaning |
|---|---|---|
| `--cpus N` | 4 | cores per job (PyTorch uses this many threads) |
| `--memory SIZE` | 8GB | memory per job. Long private runs need more (the accountant composes many steps) |
| `--gpu [TYPE]` | none | one GPU, of a MUST type (`t4`, `v100`, `a100`, ...). SPDNet runs well on CPU |
| `--short`, `--long` | 4 days (8 days for GPU jobs) | MUST duration classes: 2 days, 8 days |
| `--disk SIZE` | 1GB | local disk per job |
| `--group NAME` | default group | accounting group, e.g. `group_usmb.<group>` (`cat /etc/condor/mapfile \| grep $(id -un)`) |
| `--retry-failed` | | also resubmit failed and interrupted jobs (once `condor_q` shows none of the plan) |
| `--dry-run` | | write the jobs file and print the `condor_submit` command only |

### 6. Write your own plan

Most studies only change the dataset, the arms or the search space. Copy
`rfl/experiments/plans/template_new_dataset.py` and edit the `Protocol` it builds
(`rfl/experiments/protocol.py`):

```python
from rfl.experiments.protocol import Protocol

P = Protocol(
    dataset="Cho2017", eps=10, models=("spdnet",),
    kinds=("central", "fed_shared", "fed_fedper", "dp_shared", "dp_fedper"),
    space={"nodp": {"lr": [1e-3, 1e-2], "batch_size": [32], "horizon": [100, 200]},
           "dp": {"lr": [1e-2, 1e-1], "batch_size": [64], "horizon": [100, 200],
                  "--dp-clip-C": [0.5, 1.0]}},
    model_space={"spdnet": {"--hidden-dim": [8, 16]}},
    name="cho2017_spdnet")
NAME, STAGES = P.name, P.stages()
```

* `space["nodp"]` is the search space of the non-private arms, `space["dp"]` that of
  the private arms. `model_space` adds values for one model, `arm_space` replaces the
  space of one arm.
* A key is `lr`, `batch_size`, `horizon` (rounds, or epochs for the centralised arm)
  or any option of the runner, written as on the command line (`--dp-clip-C`,
  `--hidden-dim`, `--weight-decay`, `--local-epochs`, ...).
* Every combination runs on every sweep seed. The example above gives 456 runs
  (13 arms: one central arm, and four federated arms for each of the three
  aggregation rules). Grow the grids one
  dimension at a time, or use the staged search below.
* `--select` picks, for each arm, the combination with the best mean validation
  accuracy, whatever the swept keys. The final stage reruns it on the final seeds.
* `common_args` (default `--subjects-per-client 1`) and `model_args` hold options fixed
  for every federated arm or for one model. A key swept in the space may not appear
  there too.
* Relative paths in a plan (a synthetic data file, for example) resolve against the
  folder where you run `grid` or `submit.sh`: the jobs run there.
* `aggregations` sets the rules compared per model (default: ProjAvg, RLAvg and RetAvg
  for SPDNet, FedAvg for EEGNet). Each rule gives its own federated arms.
* `variants` gives each arm kind several variants: extra arguments, an extra search
  space, and the arm kinds they apply to. The variant name is appended to the arm name.
  `mi_spdnet_bn.py` uses six (`nobn`, `bn`, `bnshared`, and the same three with IID
  clients):

  ```python
  from rfl.experiments.protocol import Variant
  variants={"nobn": Variant(),
            "bn": Variant(args=("--use-batch-norm",), space={"--bn-momentum": [0.1, 0.3]}),
            "bnshared": Variant(args=("--use-batch-norm", "--no-fedbn"),
                                kinds=("fed_shared", "fed_fedper"))}
  ```

Three searches are available (`search=`):

| Search | Stages | Runs per arm and seed |
|---|---|---|
| `"grid"` (default) | `sweep`, `final` | the product of the list sizes |
| `"random"`, `n_samples=N` | `sweep`, `final` | N combinations drawn from the grid |
| `"staged"`, `stage_keys=[[...], [...]]` | `sweep1`, `sweep2`, ..., `final` | the sum over stages of the product of the group sizes |

The staged search tunes the groups of keys one after the other: stage k sweeps group k,
with the earlier groups at the values selected at stage k - 1 and the later groups at
their first listed value. An arm with no key of stage k in its space (say no batch norm,
when stage k tunes the momentum) skips that stage and keeps its earlier choice. Run
`--select` after each stage, before submitting the next one. `template_new_dataset.py` uses it: learning rate and horizon first, then the clip
constant and the BiMap dimension. Each stage writes its selection to
`$RFL_RESULTS/<NAME>/<stage>/selection.json`.

For anything else, a plan is a module with a `STAGES` dict that maps each stage name to
a function returning a list of jobs. `grid(arm, entry, args, seeds, lrs, batch_sizes,
horizons, space)` builds every combination for one arm, and `selected(arm, entry,
args, seeds, plan)` the jobs with the values selected for that arm:

```python
from rfl.experiments.grid import grid

def sweep():
    args = ["--dataset", "Cho2017", "--model", "spdnet", "--local-fc"]
    return grid("fedper", "fed", args, seeds=[1, 2, 3], lrs=[1e-3, 1e-2],
                batch_sizes=[32], horizons=[100, 200],
                space={"--weight-decay": [0, 1e-4]})

STAGES = {"sweep": sweep}
```

`entry` is `"fed"` or `"central"`. `args` holds the options that stay fixed. A swept
option may not also appear in `args`. Then `python -m rfl.experiments.grid myplan --check`
validates the runs and `bash condor/submit.sh myplan` submits them. A plan works the
same on a laptop with `--run --jobs N`.

## Options

`python -m rfl.experiments.run_federated_experiment --help` lists every option, by group.

### Personalisation

| Option | Effect |
|---|---|
| (none) | Trunk and head are shared and aggregated. |
| `--local-fc` | Personal classifier head (FedPer). Only the trunk is aggregated. The head is never uploaded. |
| `--local-head-mode train` | Default. The personal head trains during the rounds, with the trunk. |
| `--local-head-mode frozen` | The personal head stays at its initialisation (the same on every client) during the rounds. Use it with a re-fit. |
| `--head-refit-epochs 25,50,100,200` | After the last round, each client fits its head on its own training data, the rest of the model frozen. Works with a shared or a personal head. The candidate (epochs, weight decay) is chosen per client on its validation accuracy (`--head-refit-weight-decays`, `--head-refit-selection`, `--head-refit-lr`). |
| `--local-only` | No broadcast and no aggregation. Each client trains alone, with its own schedule and stopping rule. A baseline. |
| `--freeze-trunk` | Every parameter except the head stays exactly at its random initialisation (no step, no max-norm, no re-projection). The random-feature control. Also in the centralised runner. |
| `--eegnet-norm-type batch` | EEGNet with BatchNorm. Non-private runs only. |
| `--use-batch-norm` | SPDNet with one SPD batch norm per block (see [SPD batch norm](#spd-batch-norm)). Non-private runs only. |
| `--client-partition iid` | IID clients: the trials of each split are pooled and dealt at random to the clients, stratified by class. Same number of clients and same trials as the default `subject` partition, in which each client holds its own subjects. The IID control of a study on heterogeneity. Real EEG only. |
| `--no-fedbn` | By default the normalisation layers stay on each client (FedBN): affine parameters, SPD bias and running statistics are never aggregated. With this flag they are shared: trainable parameters are averaged, EEGNet running statistics too, and SPD running means are combined in the geometry of their layer. |

The result tables call the scheme `scope`: `shared`, `fedper`, `fedper_frozen` or
`local`, with `_rnd` when the trunk is frozen and `+refit` with a head re-fit.

### Privacy

| Option | Effect |
|---|---|
| `--dp-enabled` | Record-level DP of everything the clients send. |
| `--dp-target-eps`, `--dp-target-delta` | Target budget (default eps = 10, delta = 1e-5). sigma is calibrated once, before training. |
| `--dp-clip-C` | Clip constant C, fixed a priori (default 1). |
| `--dp-accounting prirfed` | Calibrate sigma with the PriRFed accountant (baseline). |
| `--dp-compare-prirfed` | Also record the sigma PriRFed would need (reporting only). |

One private local step on the parameters x that train during the round is

```
g~ = (1/B) sum_l clip_C( P_x grad f(x; z_l) ) + P_x xi,    xi ~ N(0, sigma^2 I)
```

on a minibatch of B records drawn without replacement. P_x is the tangent projection
for BiMap weights and the identity elsewhere. For BiMap weights the clip and the noise
act in orthonormal coordinates of the tangent space at uf(x), computed in float64
(`rfl/geometry/stiefel.py:StiefelChart`). The step is then exactly a Gaussian mechanism with
sensitivity 2C/B, also when x is orthonormal only up to float32 round-off. A per-sample gradient that is not finite
contributes zero. The noise and the minibatch draws use two independent streams per
client, derived from the run seed: the noise is reproducible, as suits a simulation.
Client i runs
`local_epochs * floor(m_i / B)` steps per round. The accountant composes
`T * local_epochs * ceil(m_max / B)` steps at subsampling rate `B / m_min`, an upper
bound for every client. `fl_results.json` records the realised budget (`dp_budget`).

What the guarantee covers, with personalisation:

* Every parameter that trains during the rounds is inside the privatised vector,
  shared or personal. A personal head in `train` mode is therefore clipped jointly with
  the trunk and noised, even though it is never uploaded. A head trained on raw
  gradients would leak the records into the later trunk updates.
* A frozen head or trunk does not depend on the data. It stays out of the vector, so
  the whole clip budget goes to the parameters that train.
* The head re-fit happens after the last release and sends nothing. The transcript
  guarantee is unchanged. The re-fitted head itself is computed from the raw records,
  so it is not protected if it is published.
* With `--local-only` nothing is sent. DP then protects each client's own model
  (private local training), a baseline for private federated training.
* Client participation is visible to the server, so `--client-fraction` gives no
  amplification.
* The guarantee is stated for a given train/validation/test split. The split is
  stratified over all trials, so it depends on every label, and so do the client
  training-set sizes and the number of classes of the head. Once the split is drawn,
  these are fixed and the guarantee covers the training records.
* The training loss and accuracy that each client reports per round
  (`avg_train_loss`, `avg_train_acc` in the history) are computed on its private
  minibatches. They are monitoring diagnostics of the simulation and are not covered
  by the guarantee. A deployment would not send them. The validation and test
  metrics use the held-out records, outside the guarantee too.
* A record on which the model fails (a non-finite matrix, for example) contributes
  zero to the step, like a non-finite gradient, so the run never fails because of
  one record.

Every personalisation option works with and without `--dp-enabled`. The code refuses,
with an explicit message, the settings that would break the guarantee: BatchNorm under
DP (EEGNet `batch`, SPDNet `--use-batch-norm`) and global normalisation of the trials
under DP. It warns when DP protects less than the transcript (local only, frozen trunk
with a personal head).

With `--report selected`, the server also keeps each client's local keys (personal
head, local normalisation) of the selected round. The re-fit, the ReEig activity and
`client_states.pt` then use exactly the model that was evaluated at that round.

### SPD batch norm

The layer follows Gallet, Mian, Bouchard and Ginolhac, *On batch normalization for
SPDnet* (Signal Processing 250, 2027, hal-05742151), implemented in yetanotherspdnet by
the same authors. A batch {S_k} is centred on its batch mean G and moved to a learned
SPD bias Gamma, `S_k -> Gamma G^{-1/2} S_k G^{-1/2} Gamma`, so that its mean lands on
Gamma^2. At inference G is replaced by a running mean R. The choice of the mean fixes
the batch mean, the update of R (the geodesic or curve of that geometry) and the
backpropagation, which the library computes with the closed forms of the paper.

`--use-batch-norm` adds one layer after the ReEig of each BiRe block, as in the paper
and in the library's SPDnet. The layers compute in float64.

| Option | Default | Choices |
|---|---|---|
| `--bn-mean-type` | `geometric_arithmetic_harmonic` (GAH) | `affine_invariant`, `log_euclidean`, `arithmetic`, `harmonic`, `adaptive_geometric_arithmetic_harmonic`, `bures_wasserstein` |
| `--bn-momentum` | 0.1 | update rate gamma of the running mean |
| `--bn-type` | `mean_only` (`BatchNormSPDMean`, the scope of the paper) | `mean_var_scalar` (`BatchNormSPDMeanScalarVariance`, also a scalar dispersion) |
| `--bn-norm-and-bias-mode` | `whitening` (Eq. 2 of the paper) | `parallel_transport` (`mean_only`, not with Bures-Wasserstein) |
| `--bn-position` | `after_reeig` | `before_reeig` (SPDNetBN of Brooks et al., 2019) |
| `--bn-recalibrate` | off | after training, set each running mean to the exact mean of its training inputs |

What is learned. Each layer learns one SPD bias Gamma per block (Algorithm 1 of the
paper, `Covbias` in the library, through a softplus of its eigenvalues). The mean of
the batch is estimated from the data: the batch mean G during training, the running mean
R at evaluation, updated with the momentum gamma (`--bn-momentum`). The 1/2 of GAH,
`A #_{1/2} H`, is the midpoint of the geodesic between the arithmetic mean A and the
harmonic mean H: a fixed part of the definition. `adaptive_geometric_arithmetic_harmonic`
learns that point instead.

Choice of the mean, from the paper. The affine-invariant mean is the theoretical
reference but needs an iterative computation. The closed-form means are much cheaper.
The arithmetic mean suits Wishart data, the harmonic mean inverse-Wishart data, and GAH
is the compromise when no model is known, and the only closed-form mean invariant under
both congruence and inversion (Table 1). GAH and the harmonic mean were competitive or
better on the three datasets of the paper, none of them EEG: HDM05 (motion capture,
GAH best), HyperLeaf (hyperspectral, harmonic best) and Rices90 (hyperspectral,
log-Euclidean best). GAH is the default here.

The layers are those of yetanotherspdnet, used as they are. For the GAH mean, the
library keeps one running mean R and moves it at each batch along the GAH curve towards
the GAH mean of the batch. The paper (Eq. 13) describes another update, with an
arithmetic and a harmonic running mean combined at the end. This repository follows the
library.

Momentum. The default is 0.1, against 0.01 in the library: a federated round runs few
local steps, and with 0.01 the running mean would lag behind the batch means for many
rounds.

Running means after training (`--bn-recalibrate`, `rfl/models/bn_recalibration.py`).
With little data a client runs one step per round, and R moves little: with momentum
0.01, R covers about 26 % of the way from I after 30 rounds. The evaluation then centres
the data on a mean the training never used. The option replaces R, once training is
over, by the exact mean of the layer's inputs on the training data, computed with the
layer's own mean function and the model in evaluation mode, block after block ("precise
BN", Wu and Johnson, 2021). Only the buffer is written: the library is unchanged. With
FedBN (and local training) each client uses its own data. With a shared batch norm each
client computes its mean, the server combines them as it combines running means, and
every client receives the result. `final_metrics` then hold the recalibrated model, and
`final_metrics_before_recalibration` the momentum one.

How the server treats the batch norm. FedBN (default): R and Gamma stay on the client.
They are uploaded but every aggregation rule leaves them out, and a client keeps its own
when it receives the global model, so R accumulates the client's batches round after
round and each client centres its test data on its own mean. Shared (`--no-fedbn`):
the client starts each round from the global R and Gamma, trains one epoch (each batch
centred on its own batch mean, R moving towards the client's batch means), and the
server sets the new R to the weighted mean of the client running means in the geometry
of the layer (`weighted_spd_mean`, A #_{1/2} H of the weighted arithmetic and harmonic
means for GAH) and averages the parameter of Gamma as any parameter outside Stiefel.
Every client then evaluates with the same R and Gamma. EEGNet follows the same logic:
local BatchNorm statistics and parameters with FedBN, FedAvg of all of them otherwise.

ReEig floors. After a batch norm the spectrum is centred near Gamma^2, about the
identity, on another scale than the trace-normalised input. The ReEig floors of blocks 2 and later (and of every
block with `before_reeig`) must therefore be recalibrated on that scale: the paper uses
floors from 0.01 to 0.8 with batch norm, against 4e-6 to 8e-5 in `rfl/data/registry.py`.
`reeig_activity` in `fl_results.json` shows whether a floor is active.

Federated scope. By default the layer stays on each client (FedBN). With `--no-fedbn`
the bias (a trainable parameter, optimised through its softplus parametrisation) is
averaged like the other parameters, and each running mean is combined across clients
in the geometry of its layer: arithmetic, harmonic or log-Euclidean weighted mean,
weighted Karcher mean for the affine-invariant mean, and for GAH the geometric
midpoint of the arithmetic and harmonic weighted means of the client running means. Sharing is offered
for `mean_only` with these five means.

Running means and federation. Each batch-norm layer keeps a running mean R (the
buffer `running_mean`). It replaces the batch mean at evaluation.
It is updated at each local training step, from the matrices that the local trunk W_k
of the client sends to the layer. At the end of a round the model is evaluated with the
aggregated trunk W, while each R was estimated with the local trunks W_k. With FedBN
each client uses its own R. With `--no-fedbn` the clients' R are combined. In both
cases R lags the evaluated trunk by one aggregation. Only local-only runs avoid it. The
same holds for the BatchNorm of EEGNet, as in FedBN (Li et al., 2021). The gap shrinks
when the local trunks stay close to W (late rounds, small learning rate), but it is not
zero.

Study. Plans `mi_spdnet_bn_calib` and `mi_spdnet_bn` compare, for SPDNet, the
normalisation (none, FedBN, shared batch norm), the scope (shared head, FedPer, local
only) and the clients (one subject per client, IID clients), with a centralised
reference with and without batch norm: 18 arms. Every arm trains as in the paper
(plain SGD, `--stiefel trivialization`), with one local epoch per round, and aggregates
with ProjAvg. One protocol serves every arm, personalised or not:

1. ReEig floors, computed from the data. A floor is the q-th percentile of the
   spectrum its ReEig acts on: the trial covariances (trace-normalised) after the
   congruence by the BiMaps (random Stiefel W), after the previous ReEig, and with a
   batch norm after the GAH whitening of the previous block. `analyze_covariances
   --architectures` writes them per candidate architecture, with and without batch
   norm, for q = 1 to 80. The runners read them with `--reeig-percentile q`. Without
   batch norm each floor exceeds the previous one (Poincare separation). With batch
   norm the later floors sit near the identity scale (the paper uses 0.01 to 0.8).
2. Dataset settings, chosen once on the centralised SPDNet (`mi_spdnet_bn_calib`):
   learning rate and batch size, then the percentile q (5, 15, 30) and, with batch
   norm, the momentum of the running mean, for each candidate architecture. The best
   architecture is kept, with and without batch norm. Schirrmeister2017 (128 channels)
   tries four two-block architectures (64,32 64,16 32,16 32,8), the other datasets
   keep their single block. These settings describe the data and the layer, so every
   arm then uses them.
3. Every arm, tuned alike on the mean validation accuracy over 3 seeds: learning rate
   x batch size (12 cells, horizon 200), then the horizon (100, 200, 300). With SGD the
   rate and the batch size act together, hence one joint grid. The horizon is the
   shortest one within one standard error of the best: the training has converged
   when a longer one no longer helps.
4. A choice on the edge of its grid (learning rate, batch size, momentum, q, or the
   longest horizon) gets one more value beyond the edge, until the choice lies inside
   its grid or reaches the bound of its rule (`EXTEND` in `mi_spdnet_bn_calib.py`).
5. Final runs on 10 other seeds. The seed also draws the train/validation/test split,
   the same for every arm, so the arms pair run by run. The last round is reported.

The percentile q is a choice of the method. The calibration treats it as a
hyperparameter, so the share of eigenvalues each ReEig clamps is set by the validation
accuracy. The floors themselves stay tied to the spectrum, hence comparable across
datasets and architectures.

The floors of `rfl/data/registry.py` (the papers' single blocks) are the 15th
percentile of the input spectrum, before the BiMap. After a BiMap that reduces the
dimension, the bottom of that spectrum is gone (Poincare interlacing), so such a floor
clamps much less than 15 %. The study recomputes every floor where its ReEig acts.

`test_acc` pools the test predictions of all clients. `client_mean_acc` averages the
accuracy of each client, so every subject counts the same, and `client_min_acc` is the
worst client. Report both for a personalisation study. On IID clients FedBN and a shared
batch norm should give the same result, since every client sees the same distribution.
A gap between batch norm and no batch norm can remain there: each IID client mixes all
subjects, and the layer also acts within each batch.

A last training batch of a single matrix is dropped, since its batch statistics are
undefined. DP refuses the layer: its running mean is computed on the private batch,
which couples the records and is not covered by the clipping.

### Data quantity

When does a subject gain by joining a federation? Each subject keeps N calibration trials
per class (`rfl/data/reduction.py`), drawn from its training and validation trials of
the usual split. Its test trials do not change, so every N is scored on the same test
set.

| Option | Default | Effect |
|---|---|---|
| `--train-trials-per-class N` | all | keep N trials per class and subject |
| `--calibration-val-fraction` | 0.2 | share of the N trials used for validation (at least 1) |
| `--calibration-order` | `random` | `random`: shuffled per subject with the run seed. `chronological`: the first trials recorded (a short calibration session) |
| `--reduced-subjects K` | 0 (all) | reduce K subjects drawn with the seed, the others keep all their trials |

The subsets are nested (the trials kept at N are kept at 2N) and stratified. The
federated and centralised runners and the TS+LR baseline accept the options, with subject clients only. The
results record the reduced clients (`fl_info.reduced_clients`), and
`rfl/analysis/results.py` adds `reduced_val_acc`, `reduced_test_acc` and
`reduced_refit_test_acc`, the mean over those clients (every client when none is
reduced), with `n_per_class` and `reduced_subjects`.

Baseline: `rfl/experiments/run_tslr.py`, tangent space at each client's Riemannian
mean then logistic regression, per client (`--mode local`) or on all clients re-centred
on their own means (`--mode pooled`, a non-federated transfer reference). It uses the
clients, split and reduction of the federated runs, and picks its regularisation on the
validation accuracy of the reduced clients.

Study (`mi_datasize`, `mi_datasize_eegnet`, `mi_datasize_tslr`, building blocks in
`rfl/experiments/datasize.py`):

1. Scenarios: `U<N>` every subject has N trials per class, `H<N>` a third of the
   subjects has N and the others all theirs (new users with a short calibration in a
   federation of data-rich users), `full` no reduction.
2. Arms: every pair of the two levels of personalisation, head (shared, shared then
   re-fitted on each client, personal FedPer, nothing shared) and normalisation (none,
   FedBN, shared batch norm), for SPDNet and EEGNet, plus TS+LR local and pooled.
   SPDNet keeps the architecture, floors and momentum calibrated by
   `mi_spdnet_bn_calib`, with `--bn-recalibrate`. Both models train with plain SGD.
3. Tuning of every arm on its own: learning rate x batch size, then horizon, selected on
   `reduced_val_acc`. The batch grid sits around half a client's training set in U<N>
   and takes the dataset batches in H<N>. The horizon grid grows by sqrt(40 / N), at most x4,
   since a client with N trials runs few steps per round. Grids extend at their edges.
4. Final runs on 10 seeds. The seed draws the split (on real EEG, a synthetic .npz
   stores its own), the reduction and the reduced subjects, the same for every arm.
   TS+LR builds one client per subject, as the plans do (`--subjects-per-client 1`).

```bash
export RFL_DATASET=BNCI2014_001
bash condor/submit_dag.sh mi_datasize --memory 12GB      # calibration, SPDNet, EEGNet, TS+LR
python -m rfl.analysis.datasize $RFL_RESULTS/mi_BNCI2014_001_datasize*/final \
    --out datasize_BNCI2014_001 --pairs
```

`rfl/analysis/datasize.py` writes a table and one figure per network model (SPDNet,
EEGNet), with the TS+LR baselines on each: test accuracy against N (log scale, "all"
for no reduction), one panel per scenario. The colour gives the paradigm and the head
(local, federated with a shared head, shared then re-fitted, personal), the line style
the normalisation (solid: batch norm on each client, dashed: shared, dotted: none). The
table names each curve in full ("Federated · SPDNet · shared head · FedBN").
Each point is the mean over the final seeds of `reduced_test_acc` (the reduced clients,
every client in U<N> and full), with its standard error over the seeds. `--pairs` adds
the per-seed gain of each federated curve, re-fitted heads included, over the best local
curve of the scenario (a network or TS+LR, chosen on its mean). It needs two seeds or
more.

The two levels of personalisation, as options:

| Arm name | Head | Options |
|---|---|---|
| `..._shared_<norm>_<scen>` | shared | (none). The final stage adds `--head-refit-epochs`: `test_acc` keeps the shared head, `refit_test_acc` the re-fitted one, and `report` names the scope `shared+refit` |
| `..._fedper_<norm>_<scen>` | personal, trained during the rounds | `--local-fc` |
| `fed_<model>_local_<norm>_<scen>` | nothing shared | `--local-only` |

| `<norm>` | SPDNet | EEGNet |
|---|---|---|
| `nobn` | (no batch norm) | `--eegnet-norm-type none` |
| `bn` (kept on each client, FedBN) | `--use-batch-norm --bn-recalibrate` | `--eegnet-norm-type batch --bn-recalibrate` |
| `bnshared` | `--use-batch-norm --bn-recalibrate --no-fedbn` | `--eegnet-norm-type batch --bn-recalibrate --no-fedbn` |

The re-fitted head reuses the values tuned for the shared head. SPDNet and EEGNet share
the scenarios, seeds, clients, trials, plain SGD, learning-rate grid, tuning and
selection. They differ by their architecture and aggregation (SPDNet: orthogonal
parametrisation and ProjAvg, EEGNet: FedAvg). Plain SGD has no state, so the new
optimiser of each federated round (as in FedAvg) changes nothing. With Adam or momentum,
the state carries over from round to round in local-only training (as in centralised
training) and restarts at each federated round. EEGNet standardises each channel of
each trial (`--eegnet-input-norm per_trial`, as in the papers). `per_trial_global`
standardises the trial as a whole and keeps the relative channel powers, which SPDNet
sees. `rfl/analysis/datasize.py` writes one .csv with both models: compare them there.
With `rfl.analysis.report` and `stats`, group a data-quantity folder by
`n_per_class reduced_subjects` as well.

Cost per dataset, before extensions (BNCI2014_001, 8 scenarios): SPDNet 64 arms, about
3,500 runs (2,304 + 576 + 640), EEGNet the same, TS+LR 160 short runs. The batch-norm study: 108 calibration runs, then 648 + 162 + 180. The plan
`synth_datasize` (and `synth_datasize_tslr`) runs the same protocol on synthetic
subjects whose discriminant directions differ (`rfl/data/synthetic.py`,
`--pattern-rotation`), where the local and federated curves cross. `RFL_QUICK=1` makes
it a wiring check (under an hour on 2 cores), in folders of its own.

### Checks before a run

The runners stop with a message, before loading any data, when an option set on the
command line has no effect in the run (`rfl/experiments/cli_checks.py`). Examples:

| Command | Message |
|---|---|
| `--dp-target-eps 5` without `--dp-enabled` | only acts with `--dp-enabled` |
| `--bn-mean-type harmonic` without `--use-batch-norm` | only acts with `--model spdnet --use-batch-norm` |
| `--model eegnet --hidden-dim 8` | only acts with `--model spdnet` |
| `--hidden-dim 12` with `--hidden-dims 16,8` | replaced by `--hidden-dims/--epsilons` |
| `--lr-plateau-patience 5` with the cosine schedule | only acts with `--lr-schedule plateau` |
| `--selection-metric val_loss` with `--report last` and no plateau | only acts with `--report selected`, a plateau or early stopping |
| `--local-only --aggregation rlavg` | no effect with `--local-only` |
| `--no-fedbn` without a normalisation layer | needs a normalisation layer |

Impossible values are refused too: a negative weight decay, `fmin >= fmax`, `fmax`
above the Nyquist frequency of the resampled signal, `--cov-estimator lwf` with zero
shrinkage, early stopping with `--report last`, and the combinations of the earlier
sections (DP with a batch norm, a frozen head without re-fit and a frozen trunk, ...).
Under DP, a warning flags the choices driven by the validation records, which the
guarantee does not cover (`--report selected`, the plateau schedule, early stopping),
and a `delta` that is not well below 1/n. A warning also flags `--bn-position
before_reeig`, whose ReEig floors need recalibrating (SPD batch norm, above). A
configuration error names the command-line option of each field it mentions.

Two checks use the dataset (`rfl/data/stats.py`): the BiMap dimensions must not exceed
the number of EEG channels (from the MOABB metadata), and under DP the batch size must
not exceed the smallest client training set (from the statistics file written by
`condor/fetch_data.sh` and by every run).

`--device` defaults to `auto`: the GPU when there is one, else the CPU, and
`config.json` records the device actually used. `--device cuda` on a machine without
GPU stops the run instead of falling back to the CPU. Synthetic data (SPD matrices) are
refused with EEGNet.

`--print-config` prints the complete configuration of a command, every default
resolved, and exits without running:

```bash
python -m rfl.experiments.run_federated_experiment --dataset Cho2017 --local-fc --print-config
```

The defaults of the runners are those of the configuration classes
(`rfl/utils/config.py`, `rfl/federated/fl_config.py`): a configuration built in Python
without arguments is the run of a bare command line. `tests/test_cli_checks.py`
checks it.

### Training protocol

Defaults: one local epoch per round, Adam (`--optimizer sgd` for plain SGD, `--momentum`
for heavy-ball SGD), rebuilt at every federated round and kept across rounds in
local-only training, cosine schedule from
`--learning-rate` to `--learning-rate / 100` over a fixed number of rounds, no early
stopping, last round reported. This is the protocol of the DP paper, and the
recommended one:

* The same rule serves the centralised, federated and private runs, so a comparison
  changes one factor only.
* Under DP the horizon must be fixed before training: the accountant composes all
  rounds, and validation-driven decisions are outside the guarantee.
* The reported model is the one a deployment would use. Taking the best of hundreds of
  noisy validation scores biases the result upward.
* Decreasing step sizes and a fixed horizon match the convergence analysis.

The validation split is evaluated at every round (`round_history`), for diagnostics. It
only selects the hyperparameters: the learning rate, the batch size and the number of
rounds T, on the mean over a few sweep seeds of the last-round validation metric. Final
results use other seeds. Tune T like the learning rate, on a short grid. Tune each
setting (centralised, federated, private) separately, since DP needs larger batches and
other rates.

Personalisation follows the same protocol. Every option of the
[Personalisation](#personalisation) table runs with the cosine schedule, a fixed T and
the last round, with and without DP. Three points:

* Tune the learning rate and T per family of arms. A local-only client trains on its
  own records only, with no averaging, and usually needs its own values (the ICASSP
  paper used a separate rate for its local arms). Arms that only change the head
  can start from the values of the shared arm, but check them on the sweep seeds.
* Without early stopping, local-only clients all run T rounds on the cosine schedule.
  Their per-client stopping rule is only active with `--early-stop-patience`.
* The head re-fit picks its candidate per client on that client's validation records.
  This is a hyperparameter choice, consistent with the protocol. For a re-fit with no
  validation use, set `--head-refit-selection fixed`.

The protocol of the ICASSP paper remains available for quick exploration, when T is
not tuned yet:
`--lr-schedule plateau --early-stop-patience N --selection-metric val_loss --report selected`.

### SPDNet depth and ReEig floors

The SPDNet BiMap dimension and ReEig floor default to the values of `rfl/data/registry.py`
for the dataset (`--hidden-dim`, `--epsilon`). `--hidden-dims 16,8 --epsilons 1e-4,1e-2`
builds two BiRe blocks, each with its own ReEig floor. Without batch norm the floors
must strictly increase: a congruence by a semi-orthogonal BiMap keeps the eigenvalues
inside the previous spectral interval (Poincare separation), so a floor not above the
previous one makes that ReEig the identity and the blocks collapse into one (Oswald and
Massart, arXiv:2606.02490). The code refuses such a schedule. Increasing floors are
necessary but not sufficient: eigenvalues must actually fall between them. A batch
norm between blocks re-centres the spectrum and breaks this argument, so with
`--use-batch-norm` the floors are not checked and may be equal.

Each federated SPDNet run records `reeig_activity` in `fl_results.json`: the mean
relative change `||ReEig(X) - X|| / ||X||` of each block on the validation records.
A value near 0 flags a block whose ReEig acts as the identity.
`python -m rfl.experiments.analyze_covariances --hidden-dim-block1 16` calibrates two
increasing floors from the covariance spectra.

## Data

MOABB motor-imagery datasets, listed in `rfl/data/registry.py`: BNCI2014_001, Weibo2014,
Cho2017, Schirrmeister2017, PhysionetMI, Stieger2021 and Lee2019_MI. To download one in
advance:

```bash
python -m rfl.experiments.fetch_dataset --dataset BNCI2014_001 --data-dir $HOME/mne_data
```

**Where the data goes.** Every script uses one folder: `--data-dir` (download script),
else the environment variable `MNE_DATA`, else `~/mne_data`
(`rfl/data/registry.py:configure_mne_paths`). Three details of MNE and MOABB make this
less simple than it looks, and the code handles them:

* MOABB does not read `MNE_DATA` first. It reads one key per dataset host
  (`MNE_DATASETS_BNCI_PATH`, `MNE_DATASETS_GIGADB_PATH`, ...), in the environment, then
  in the MNE configuration file `~/.mne/mne-python.json`. An old entry of that file
  (for example a path through a symbolic link, or a folder that no longer exists) then
  wins over `MNE_DATA`. The code sets every key of the registry in the environment,
  so the file is ignored. Its conflicting entries are printed, and it is never written.
* The key of Lee2019_MI is `MNE_DATASETS_LEE2019-MI_PATH`. A shell cannot `export` a
  name with a hyphen, so the keys are set from Python.
* A symbolic link is replaced by its real path, so that every node and every process
  sees the same folder.

Each run prints `MNE data folder: ...` at its start. On MUST, `condor/env.sh` sets
`MNE_DATA` on the shared filesystem (default: `mne_data` next to the repository). Run
`condor/fetch_data.sh` on the user interface for each dataset, even if the files are
already there: MOABB checks every file against its hash and may rename files to its
current layout, which must not happen in parallel jobs. The jobs then need no network.

Each subject is loaded on its own, and the subjects are aligned by channel name: a
channel missing for one subject or one session is dropped for all, with a warning, and a
subject that fails to load stops the run (none is dropped silently). Stieger2021
interpolates its bad channels, so its covariances are rank-deficient: set
`--cov-regularization` (a diagonal floor, in uV^2) if the zero eigenvalues matter.

Trials are band-pass filtered (8 to 32 Hz by default) and resampled to 128 Hz. SPDNet
uses one sample covariance matrix per trial. EEGNet uses the raw trials, each channel
standardised over time. The train, validation and test split (75/10/15, stratified,
seeded by `--seed`) is drawn once over all trials. Each client then keeps the trials of
its subjects (`--subjects-per-client`, default 1: one subject, one client). With
`--client-partition iid` the trials of each split are instead dealt at random to the
same number of clients, stratified by class: every client then mixes all subjects. The
centralised and federated runs of a seed, and the two partitions, therefore use the
same trials.

**Memory.** A run holds every trial of the dataset in float64 (128 Hz), about twice
during loading. Estimated needs per job, to check on the first jobs (`condor_history -l
<job>`, `MemoryUsage`. MUST stops a job above 1.5 times its request):

| Dataset | Subjects (clients) | EEG channels | Trials in memory | `--memory` |
|---|---|---|---|---|
| BNCI2014_001 | 9 | 22 | about 0.5 GB | 8GB |
| Weibo2014 | 10 | 60 | about 1.4 GB | 8GB |
| PhysionetMI | 106 | 64 | about 2 GB | 8GB |
| Cho2017 | 52 | 64 | about 2.5 GB | 12GB |
| Schirrmeister2017 | 14 | 128 | about 7.5 GB | 32GB |

`fetch_dataset` counts the trials one subject at a time, so it needs the memory of one
subject only.

A federated run on real data:

```bash
python -m rfl.experiments.run_federated_experiment --dataset BNCI2014_001 --model spdnet \
    --rounds 200 --batch-size 64 --learning-rate 0.01 --local-fc --dp-enabled --dp-target-eps 10
```

Under DP the batch size must not exceed the smallest client training set.

### Adding a dataset

1. **Registry.** In `rfl/data/registry.py`, add an entry to `DATASET_CONFIGS` whose key
   is the MOABB class name (`moabb.datasets.<Name>`), with the event names kept, the
   number of classes, the subjects, the native sampling rate, and the sessions and epoch
   window if needed. Copy an existing entry. Add the dataset to `MNE_PATH_KEYS` too:
   the MNE configuration key MOABB reads for its download folder
   (`MNE_DATASETS_<SIGN>_PATH`, found in the dataset's MOABB source). The test
   `test_mne_path_keys_are_those_moabb_reads` checks both against MOABB.
2. **Download.** `bash condor/fetch_data.sh <Name>` on MUST, or
   `python -m rfl.experiments.fetch_dataset --dataset <Name>`.
3. **SPDNet floors.** `python -m rfl.experiments.analyze_covariances --datasets <Name>`
   gives the ReEig floor of the dataset (and of a two-block network with
   `--hidden-dim-block1`). Add the BiMap dimension and the floor to `DATASET_DEFAULTS`.
   Without them the runners use generic values (n_channels // 2, 1e-4) and warn.
4. **Batch sizes.** The download script also counts the trials of each subject
   (`<MNE_DATA>/rfl-stats/<Name>.json`, `rfl/data/stats.py`, skip with `--no-stats`). With
   this file, `--check` refuses a DP batch size above the smallest client training
   set, before any run starts. Each run also prints `[Client i] subjects [...]:
   train=...` at its start.
5. **Study.** Copy `rfl/experiments/plans/template_new_dataset.py`, set the dataset, the
   arms and the search space (section [6. Write your own plan](#6-write-your-own-plan)),
   then `--check`, sweep, `--select`, final. Expect to extend a grid after the first
   selection: the values that suit BNCI2014_001 need not suit another dataset.

## Outputs

A federated run writes `config.json`, `environment.json` (library versions, git
commit), `fl_results.json` (configuration, DP budget, reported metrics, re-fit
metrics, ReEig activity per block, selection, per-round history), `fl_history.json`, `global_model_final.pt` (last
and selected global states) and, when something stays on the clients (personal or
re-fitted heads, FedBN layers, local-only models), `client_states.pt`. The global state
of the reported round (`best_state_dict` of `global_model_final.pt` with
`--report selected`, `model_state_dict` with `last`) plus `local_states[cid]` rebuild
the model of client `cid` behind `final_metrics`. Adding `refit_heads[cid]` gives the
re-fitted model behind `personalized_metrics`. Local-only runs save whole models in
`full_states`. A centralised run (`rfl/experiments/run_centralized_experiment.py`)
writes `centralized_results.json` and its checkpoints.

The main metrics are pooled over the predictions of all clients (`global_test_acc`,
`global_test_f1`), as in a centralised run. Client-uniform means are kept as secondary
statistics. After a re-fit, `personalized_metrics` holds the same metrics for the
re-fitted models.

## Analysis

```bash
python -m rfl.analysis.report results --by dataset arch scope --pivot eps     # mean, std, n
python -m rfl.analysis.report results --metric refit_test_acc
python -m rfl.analysis.report results --curves figures/ --curve-figure dataset arch
python -m rfl.analysis.stats results --factor scope --baseline shared --by dataset arch
# in a final folder the shared arms carry the re-fit: --baseline shared+refit
```

`rfl/analysis/results.py` loads every finished run into a pandas table, from the
configuration stored in each result file. Each row also carries `config_id`, a hash of
the configuration without the seed. A group of the report should hold each seed once:
`rfl.analysis.report` warns, and `rfl.analysis.stats` refuses, when a group mixes runs that
differ in more than the seed (for example a learning-rate sweep). Group by more
columns (`--by ... lr batch_size horizon`) or filter with `--where` in that case.

## Other tools

| Command | Purpose |
|---|---|
| `python -m rfl.experiments.run_centralized_experiment` | Centralised reference on the same split |
| `python -m rfl.experiments.accounting_comparison` | Noise of our accountant against PriRFed, no training |
| `python -m rfl.federated.privacy.verify_accountant` | Sign-off checks of the accountant (a few minutes) |
| `python -m rfl.experiments.synthetic_pca all --quick` | Private federated PCA on the Grassmann manifold (NumPy) |
| `python -m rfl.experiments.analyze_covariances` | ReEig floors from the covariance spectra |

## Repository structure

| Path | Content |
|---|---|
| `rfl/` | The package. Its subpackages: |
| `rfl/geometry/` | Stiefel (PyTorch) and Grassmann (NumPy) primitives |
| `rfl/models/` | SPDNet and EEGNet |
| `rfl/data/` | MOABB loading, dataset registry and MNE data folder, dataset statistics for the checks, PyTorch datasets, synthetic SPD generator |
| `rfl/federated/` | Configuration, client, server, aggregation rules, parameter groups, head re-fit |
| `rfl/federated/privacy/` | DP mechanism, DP client, accountants, DP setup |
| `rfl/training/` | Centralised training loop, evaluation, checkpoints, optimiser choice |
| `rfl/synthetic_pca/` | Federated PCA on the Grassmann manifold |
| `rfl/experiments/` | Entry points: the two runners and the TS+LR baseline (`run_tslr.py`), `datasize.py` (data-quantity plans), `cli_checks.py` (option checks), `grid.py` (plans), `protocol.py` (the recommended protocol), `pipeline.py` (a whole study, automatically), dataset download, covariance analysis, accountant comparison, synthetic PCA |
| `rfl/experiments/plans/` | Experiment plans: `smoke`, `smoke_auto`, `mi_protocol`, `mi_spdnet_bn_calib` and `mi_spdnet_bn`, `synth_bn_calib` and `synth_bn`, `mi_datasize` (with `_eegnet`, `_tslr`), `synth_datasize` (with `_tslr`), `mi_study`, `template_new_dataset` |
| `rfl/utils/` | Base configuration, learning-rate schedules, seeding and environment record, logging setup |
| `rfl/analysis/` | Result tables, curves, paired tests |
| `condor/` | HTCondor scripts for MUST (see [Running on MUST](#running-on-must-htcondor)) |
| `environment/` | Lock files of the test and paper environments |
| `tests/` | Tests (pytest) |

Library modules log through `logging.getLogger(__name__)`. The entry points call
`rfl.utils.logs.setup_logging()`, which prints the messages to standard output (and so to
`run.log` on HTCondor). In a notebook, call it too to see the progress messages.

## Tests

```bash
python -m pytest -q -m "not slow"     # a few minutes on a laptop CPU
python -m pytest -q                   # full suite, about seven minutes
```

The tests need no dataset download. `tests/test_personalization.py` checks every
combination of privacy and personalisation: the privatised vector holds exactly the
parameters that train, a personal head is never uploaded, a frozen head never moves,
and the re-fit changes the head only. A GitHub Actions workflow runs the fast tests on
every push.

## Equivalence with the paper repositories

On synthetic data, these runs of this repository and of the paper repositories give
identical round histories (losses, accuracies, learning rates, geometry diagnostics,
selected round, calibrated sigma):

| Paper repository | Settings checked |
|---|---|
| `dp-rfedproj` | SPDNet non-private ProjAvg, private RLAvg (eps 10), private RetAvg with half participation (eps 5), EEGNet without BatchNorm with and without DP, centralised SPDNet |
| `icassp-spdnet-fl` | SPDNet shared, FedPer, FedPer with frozen trunk, local only (with per-client early stopping), two blocks, plateau schedule with early stopping and selected round, EEGNet with FedBN (shared, FedPer, FedPer with frozen trunk), centralised SPDNet with frozen trunk |

These checks use the library commit of the paper repositories (`5b25c30`). With the
commit pinned here (`5c08ebc`), the symmetrised eigh gradient moves SPDNet numbers by
about 1e-7, with identical accuracies on the runs checked.

Two deliberate changes make frozen parameters exactly constant. A frozen BiMap weight
is no longer re-projected after each step, and EEGNet's max-norm no longer rescales
frozen parameters. With a frozen SPDNet trunk this moves the numbers by about 1e-6
(float32 rounding), with identical accuracies on the runs checked.

Flags that changed:

* ICASSP `--local-fc --spdnet-federation-mode bimap_only` is now `--local-fc`.
* ICASSP EEGNet always had BatchNorm. Here it needs `--eegnet-norm-type batch`.
* SPDNet batch norm now sits after every block and runs in float64. Before, there was
  one layer after the last block, and it failed at the first forward pass.
* The aggregation defaults to ProjAvg for SPDNet (FedAvg in the ICASSP runner). SPDNet
  with FedAvg is refused.
* The `head_norm` diagnostic is no longer recorded in local-only runs, where the server
  head never changes.
* The defaults follow the DP paper (cosine schedule, last round, one local epoch,
  `val_acc` selection). The ICASSP protocol needs its flags explicitly (see
  [Training protocol](#training-protocol)).
* `--subjects-per-client` defaults to 1 for every dataset. The DP paper used 3 for
  BNCI2014_001 and 2 for Cho2017 and Weibo2014.

Corrections made after the merge, which change some runs of the paper repositories:

* The centralised plateau schedule now uses the federated rule (decay on the
  `patience`-th epoch without improvement, against the (patience + 1)-th with torch's
  `ReduceLROnPlateau`). Both pipelines share one implementation (`rfl/utils/schedules.py`),
  which also gives the centralised runner the `step` schedule.
* `--task binary` now keeps left hand versus right hand, found by event name. It kept
  labels 0 and 1, which are not these classes for several datasets (for BNCI2014_001,
  feet and left hand).
* A centralised run no longer resumes the checkpoint of a run with another
  configuration, and default run names include a hash of the configuration.
* DP: the clip and the noise act in tangent coordinates at uf(x), in float64, and the
  noise and minibatch streams are drawn from independent seeds. The mechanism is then
  an exact Gaussian mechanism in float32 too. Private runs draw other random numbers
  than the paper repositories, so their numbers differ by the noise, not in
  distribution.
* The `step` schedule keeps the base rate for the first 70% of the rounds (at least
  one), against one round fewer before.
* With `--local-only --report selected` each client keeps the model of its own best
  round, and the reported metrics are those of these models. Before, they were the
  pooled metrics of one round, which matched none of the models.
* A centralised run that resumes keeps its random stream (shuffle, dropout), and one
  that had stopped early does not train again.
* EEGNet with BatchNorm starts with untouched running statistics (a shape probe in
  train mode moved them).
* With `--freeze-trunk`, the trunk stays exactly at its initialisation under the
  Riemannian rules (the projection of identical matrices moved it by round-off).
* EEGNet with `projavg`, `rlavg` or `retavg` is refused. These rules act on Stiefel
  weights only, so such a run was FedAvg recorded under another name.

The sweep protocols of the two papers (grids, seeds, selection) stay in their own
repositories.

## Limitations

* MOABB downloads the EEG datasets from their original hosts, which can be slow or
  unavailable.
* GPU runs are deterministic only up to the nondeterminism of some CUDA kernels.
* The privacy guarantee is record-level, for the training records, with the
  train/validation/test split taken as given. The split is stratified over all trials,
  so a substituted record could change it. The guarantee does not cover the validation
  records, which drive the learning-rate plateau, early stopping and the selected round
  when these options are used.
* The real-data pipeline (MOABB download and preprocessing) is the one of the paper
  repositories. The checks of this repository ran on synthetic data.
* SPDNet uses the layers of yetanotherspdnet as they are. The backward of its `Vech`
  (commit `5c08ebc`) copies the gradient of each off-diagonal entry onto both
  triangles, so the gradient of the parameters before LogEig is not exactly the
  gradient of the loss (cosine about 0.975 on the checks of `tests/test_gradients.py`).
  `test_library_vech_backward` records it as an expected failure. When the library is
  fixed, this test fails and the pin in `requirements.txt` can move.
* The backward of the library whitening (`Whitening` in
  `yetanotherspdnet/functions/spd_linalg.py`, commit `5c08ebc`) prints its two
  gradients at every step: about 8 kB per training step at 16 channels, up to hundreds
  of MB per run with `--use-batch-norm`. The computation is unaffected. The jobs
  (`condor/run_one.sh`, `grid --run`) pass the output through `rfl/utils/log_filter.py`,
  which removes the printed tensors and counts them in a last line of `run.log`
  (`RFL_RAW_LOG=1` keeps them). A run started by hand prints them: pipe it through
  `python -m rfl.utils.log_filter`. `test_library_whitening_backward_is_silent` records
  the print as an expected failure.
* The HTCondor files follow the MUST documentation and were checked by a local
  simulation of the submission (one job per line, same scripts). Run the smoke plan
  first on the cluster (section [Running on MUST](#running-on-must-htcondor)).
* Long private runs compose tens of thousands of steps in the accountant, which can need
  several GB of memory.

## Citation

See `CITATION.cff`.

## Acknowledgements

[yetanotherspdnet](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet)
for the SPD layers, [MOABB](https://moabb.neurotechx.com/) and
[MNE-Python](https://mne.tools/) for the EEG datasets, and Lawhern et al. (2018) for
EEGNet.

## License

MIT, see `LICENSE`.
