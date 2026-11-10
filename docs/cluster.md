# Running on MUST (HTCondor)

[Back to the README](../README.md)

## Running on MUST (HTCondor)

[MUST](https://doc.must-datacentre.fr/) is the LAPP / Université Savoie Mont Blanc
computing centre. The jobs read the repository, the environment, the data and the
results from the shared filesystem, so nothing is copied to the nodes.

| File | Role |
|---|---|
| `condor/env.sh` | Paths of the environment, data and results. Edit once. |
| `condor/setup_must.sh` | Creates the environment, runs the fast tests. Once. |
| `condor/fetch_data.sh` | Downloads datasets and fills the trial cache. On the user interface. |
| `condor/submit.sh` | Submits the pending runs of a plan stage, one job per run. |
| `condor/submit_dag.sh` | Runs a whole study as one DAG (`rfl/experiments/pipeline.py`). |
| `condor/run_one.sh`, `rfl.sub` | The job of one run, and its submit description. |
| `condor/dag_step.sh`, `run_cmd.sh` | PRE/POST scripts of the DAG nodes, job of a command step. |

A **plan** (`rfl/experiments/plans/*.py`) lists the runs of a study.
`rfl/experiments/grid.py` turns it into jobs, checks, runs and monitors them, and
selects hyperparameters.

| Plan | Content |
|---|---|
| `smoke`, `smoke_auto`, `smoke_real` | Cluster checks: main options on synthetic data, the DAG mechanism, five runs on real data |
| `mi_protocol` | 20 arms (SPDNet ProjAvg/RLAvg/RetAvg, EEGNet FedAvg, shared and FedPer, with and without DP) |
| `mi_spdnet_bn_calib`, `mi_spdnet_bn` | SPD batch-norm study |
| `mi_eegnet_bn_calib`, `mi_eegnet_bn` | The same for EEGNet |
| `mi_datasize`, `_eegnet`, `_tslr` | Data-quantity study |
| `mi_study` | All of the above for one dataset, in one DAG |
| `synth_bn_calib`, `synth_bn`, `synth_datasize`, `synth_datasize_tslr` | The studies on synthetic data |
| `template_new_dataset` | A study to copy |

### 1. Connect with VS Code

Install *Remote - SSH*, then add to `~/.ssh/config` on your laptop:

```
Host must
    HostName lappusmb.in2p3.fr       # lappui.in2p3.fr for LAPP accounts
    User <your_login>
```

*Remote-SSH: Connect to Host...* then `must`. The user interface is for editing and
submitting. Computations go to HTCondor. `remote.SSH.serverInstallPath` moves the VS
Code server if your home quota is small.

### 2. Set up once

```bash
cd /uds_data/<lab>/<login>
git clone https://github.com/thibault-pautrel/riemannian-fl-eeg.git && cd riemannian-fl-eeg
nano condor/env.sh                          # check RFL_VENV, RFL_RESULTS, MNE_DATA
bash condor/setup_must.sh                   # Python 3.11+, environment, fast tests
nohup bash condor/fetch_data.sh BNCI2014_001 Cho2017 > fetch.log 2>&1 &   # hours: rerun until no subject is missing
tail -f fetch.log
```

Defaults: environment in `.venv`, results in `results/`, data in `../mne_data`. Every new
terminal runs `source condor/env.sh`.

**Data on the nodes.** The nodes have no network. `fetch_data.sh` writes the
preprocessed trials to `$MNE_DATA/rfl-cache` ([data.md](data.md#trial-cache)). Run it
before the first study of a dataset, or every job recomputes its subjects.

`env.sh` sets `MOABB_DOWNLOAD_PROVIDER=upstream`, so MOABB reads the files of each
dataset's own source. With `auto` (MOABB 1.7) it asks NEMAR first, about 40 s per subject
offline. For a first download from NEMAR, and to check that a dataset loads offline:

```bash
MOABB_DOWNLOAD_PROVIDER=auto bash condor/fetch_data.sh <dataset>
RFL_DATA_CACHE=refresh HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 \
    bash condor/fetch_data.sh <dataset>       # every subject: "ok (on disk, trials counted)"
```

### 3. Check the cluster

```bash
bash condor/submit.sh smoke --flash --cpus 1 --memory 2GB   # 17 jobs, minutes each
condor_q
python -m rfl.experiments.grid smoke --status
python -m rfl.analysis.report $RFL_RESULTS/smoke --by arch aggregation scope participation eps
export RFL_DATASET=BNCI2014_001
bash condor/submit.sh smoke_real --flash --cpus 1 --memory 2GB   # SPDNet, EEGNet, TS+LR on real data
python -m rfl.experiments.grid smoke_real --status
```

Every job should end `done`. The accuracies mean nothing (one seed, no tuning).

### 4. Run a study

**By hand: sweep, select, final** (`mi_protocol`, protocol in
[Training protocol](options.md#training-protocol)):

```bash
export RFL_DATASET=BNCI2014_001             # keep it set for every command
python -m rfl.experiments.grid mi_protocol --stage sweep --check    # validate, run nothing
bash condor/submit.sh mi_protocol sweep --memory 16GB               # 432 runs
python -m rfl.experiments.grid mi_protocol --stage sweep --status
python -m rfl.experiments.grid mi_protocol --stage sweep --select   # "EDGE": extend the grid, resubmit
bash condor/submit.sh mi_protocol final --memory 16GB               # 200 runs
python -m rfl.analysis.report $RFL_RESULTS/mi_BNCI2014_001/final --by kind arch aggregation scope eps
python -m rfl.analysis.stats $RFL_RESULTS/mi_BNCI2014_001/final --factor scope --by arch eps
```

`RFL_DATASET` sets the dataset and `RFL_EPS` the DP budget. Each dataset and budget has
its own folder and arm names, so studies never overwrite each other.

**As one DAG.** Every step, selection and grid extension runs by itself:

```bash
export RFL_DATASET=Cho2017                  # recorded in the DAG
bash condor/submit_dag.sh mi_study --cpus 1 --memory 12GB
bash condor/submit_dag.sh mi_spdnet_bn --memory 12GB --dry-run   # one study, DAG written only
python -m rfl.experiments.pipeline status mi_study
condor_q -dag
```

How a DAG node works:

* PRE lists the jobs still to run, or skips the node. POST selects the stage and extends
  a grid whose choice lies on its edge (the plan's `EXTEND` rules). New values and failed
  jobs run again, up to `--retries` passes (20).
* A diverged run (non-finite loss) is a result with NaN metrics, left out of the
  selection. A run failing 3 times (`RFL_MAX_ATTEMPTS`), or killed 6 times, is abandoned.
* Submitting the same DAG again, once the first has ended, skips finished steps and jobs.
* One DAG per dataset at a time. DAGs of different datasets run side by side.
* Floors go to `$RFL_RESULTS/reeig_floors/`, DAG logs to `$RFL_RESULTS/condor/dag_*/`.

`smoke_auto` checks the mechanism in minutes, `synth_bn --short` the batch-norm protocol
on synthetic data. On a laptop: `python -m rfl.experiments.pipeline run <plan> --jobs N`.

#### Step by step

`--step N` submits step N only (its number in `pipeline status`), with the same
retries, selection and extension. It refuses a step whose predecessor is unfinished.
The chains of `mi_study` (SPDNet, EEGNet datasize, TS+LR, EEGNet BN) are independent.

```bash
bash condor/submit_dag.sh mi_study --step 0 --flash     # ReEig floors
python -m rfl.experiments.pipeline status mi_study      # N0 done?
bash condor/submit_dag.sh mi_study --step 1 --flash     # calibration, sweep 1
python -m rfl.experiments.grid mi_spdnet_bn_calib --stage sweep1 --status
```

### 5. Monitor and fix

| Need | Command |
|---|---|
| Your jobs (two schedds) | `condor_q -name lappusmb9a.in2p3.fr -dag`, same with `9c` |
| Held or removed job | `condor_q -hold`, `condor_history -l <job>` (MUST kills a job above 1.5 x its memory request) |
| Memory used | `condor_history -af MemoryUsage RequestMemory` |
| Remove jobs | `condor_rm <cluster>`, `condor_rm <login>` |
| Log of one run | `$RFL_RESULTS/<NAME>/<stage>/<arm>/<hp>/seed<s>/run.log` |
| HTCondor logs | `$RFL_RESULTS/condor/<NAME>_<stage>/logs/` |
| Resubmit failed runs (no job of the plan in `condor_q`) | `bash condor/submit.sh <plan> <stage> --retry-failed` |
| Run one job by hand | `bash condor/run_one.sh <line of $RFL_RESULTS/condor/<NAME>_<stage>/jobs_*.txt>` |
| Rerun a finished run | `FORCE=1 bash condor/run_one.sh <line>` |
| Free space of the group quota | see below |

```bash
used=$(getfattr --only-values -n ceph.dir.rbytes /mustfs/MUST-DATA/listic)
echo "free: $(( ($(getfattr --only-values -n ceph.quota.max_bytes /mustfs/MUST-DATA/listic) - used) / 1000000000 )) GB"
```

A full quota makes runs fail. Their result is then missing, so the DAG reruns them.

States in `--status`:

| State | Meaning |
|---|---|
| `pending` | never submitted |
| `queued` | submitted, not started |
| `running?` | started, no exit code: running, or killed by HTCondor |
| `failed` | non-zero exit code, or no result file |
| `abandoned` | failed 3 times, left out of the selection (`FORCE=1` reruns it) |
| `done` | result file and exit code 0 (a diverged run is done) |
| `conflict` | the folder holds another command: give the plan a new `NAME` or delete the folder |

`submit.sh` submits only runs neither done nor queued, after checking every command
line. `--select` waits for every job of the stage (`--allow-incomplete` overrides). The
final stage refuses a selection older than its sweep.

`submit.sh <plan> <stage>` options (`submit_dag.sh` takes the same, plus `--retries`,
`--step`, `--priority`, `--dry-run`):

| Option | Default | Meaning |
|---|---|---|
| `--cpus N` | 4 | cores per job |
| `--memory SIZE` | 8GB | memory per job |
| `--gpu [TYPE]` | none | one GPU (`t4`, `v100`, `a100`, ...). SPDNet runs well on CPU |
| `--flash`, `--short`, `--long` | 4 days | 1 hour (reserved slots, 1 core, 2GB), 2 days, 8 days |
| `--disk SIZE` | 1GB | local disk per job |
| `--group NAME` | default | accounting group, e.g. `group_usmb.<group>` |
| `--priority N` | 0 | order among your own idle jobs |
| `--retry-failed` | | also resubmit failed jobs |
| `--dry-run` | | write the jobs, submit nothing |

### 6. Write your own plan

Copy `rfl/experiments/plans/template_new_dataset.py` and edit its `Protocol`
(`rfl/experiments/protocol.py`):

```python
from rfl.experiments.protocol import Protocol, Variant

P = Protocol(
    dataset="Cho2017", eps=10, models=("spdnet",),
    kinds=("central", "fed_shared", "fed_fedper", "dp_shared", "dp_fedper"),
    space={"nodp": {"lr": [1e-3, 1e-2], "batch_size": [32], "horizon": [100, 200]},
           "dp": {"lr": [1e-2, 1e-1], "batch_size": [64], "horizon": [100, 200],
                  "--dp-clip-C": [0.5, 1.0]}},
    model_space={"spdnet": {"--hidden-dim": [8, 16]}},
    aggregations={"spdnet": ("projavg", "rlavg")},
    variants={"nobn": Variant(),
              "bn": Variant(args=("--use-batch-norm",), space={"--bn-momentum": [0.1, 0.3]},
                            kinds=("central", "fed_shared", "fed_fedper"))},   # DP refuses BN
    name="cho2017_spdnet")
NAME, STAGES = P.name, P.stages()
```

* `space["nodp"]` and `space["dp"]`: search spaces of the non-private and private arms.
  `model_space` adds keys for one model, `arm_space` replaces the space of one arm.
* A key is `lr`, `batch_size`, `horizon` (rounds, or epochs when centralised) or any
  runner option as on the command line.
* `common_args` and `model_args` hold fixed options. A swept key may not appear there.
* `variants` gives each arm kind variants (extra arguments, space, kinds). The name is
  appended to the arm name.
* `--select` keeps, per arm, the best mean validation accuracy over the sweep seeds.

| `search=` | Stages | Runs per arm and seed |
|---|---|---|
| `"grid"` (default) | `sweep`, `final` | product of the list sizes |
| `"random"`, `n_samples=N` | `sweep`, `final` | N draws from the grid |
| `"staged"`, `stage_keys=[[...], [...]]` | `sweep1`, `sweep2`, ..., `final` | sum over stages of the group products |

The staged search tunes one group of keys per stage, the earlier groups at their
selected values, the later ones at their first value. Run `--select` after each stage.

A plan can also be any module with a `STAGES` dict of functions returning jobs:

```python
from rfl.experiments.grid import grid

def sweep():
    args = ["--dataset", "Cho2017", "--model", "spdnet", "--local-fc"]
    return grid("fedper", "fed", args, seeds=[1, 2, 3], lrs=[1e-3, 1e-2],
                batch_sizes=[32], horizons=[100, 200],
                space={"--weight-decay": [0, 1e-4]})

STAGES = {"sweep": sweep}
```

Then `python -m rfl.experiments.grid myplan --check`, and `bash condor/submit.sh myplan`
or `--run --jobs N` on a laptop.
