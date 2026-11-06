# Running on MUST (HTCondor)

[Back to the README](../README.md)

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
minutes), `smoke_real` (five short runs on a downloaded dataset, to check the worker
nodes), `smoke_auto` (the automatic pipeline on synthetic data, minutes), `mi_protocol` (a motor-imagery study with the recommended protocol),
`mi_spdnet_bn_calib` then `mi_spdnet_bn` (the SPD batch norm in SPDNet: none, FedBN or shared, with a shared or
personal head, on subject and IID clients), `synth_bn_calib` then `synth_bn` (the same
study on a small synthetic federation, about 2 h on 2 cores), `mi_datasize` with
`mi_datasize_eegnet` and `mi_datasize_tslr` (federated against local training as the
number of calibration trials falls, see [Data quantity](options.md#data-quantity)),
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
bash condor/submit.sh smoke --flash --cpus 1 --memory 2GB   # 17 jobs, minutes each
condor_q                                     # your jobs: idle, running, held
python -m rfl.experiments.grid smoke --status    # done / failed / pending
python -m rfl.analysis.report $RFL_RESULTS/smoke --by arch aggregation scope participation eps
```

Then, once a dataset is downloaded, the same check on real EEG: five short runs
(SPDNet, EEGNet, TS+LR) that read the data on the worker nodes.

```bash
export RFL_DATASET=BNCI2014_001
bash condor/submit.sh smoke_real --flash --cpus 1 --memory 2GB
python -m rfl.experiments.grid smoke_real --status
```

The worker nodes have no network. `fetch_data.sh` writes the preprocessed trials of each
subject to `$MNE_DATA/rfl-cache` (see [data.md](data.md#trial-cache)), and the jobs read
them from there. Fill the cache of a dataset with `fetch_data.sh` before its first
study: a job that misses the cache recomputes the subject from the raw files, and a
study starting hundreds of jobs at once would repeat that work in every job.

`condor/env.sh` sets `MOABB_DOWNLOAD_PROVIDER=upstream`: MOABB then reads the files of
each dataset's own source. With MOABB 1.7 and the default `auto`, the datasets that
have a NEMAR mirror ask NEMAR for every subject before they read the local files (about
40 s per subject offline), and download the NEMAR copy again on the user interface. For
a first download from NEMAR, set it back for that command:
`MOABB_DOWNLOAD_PROVIDER=auto bash condor/fetch_data.sh <dataset>`. To check that a
downloaded dataset loads without network, raw files included, on the user interface:

```bash
RFL_DATA_CACHE=refresh HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 \
    bash condor/fetch_data.sh <dataset>
```

Every subject should read `ok (on disk, trials counted)`, without `Retrying` lines.

Every job should end as `done`. The accuracies of the smoke plan mean nothing (one
seed, no tuning). It only checks that the main options run on the worker nodes. It has
no EEGNet, no real data, no TS+LR and no data reduction.

### 4. Run a study: sweep, select, final

`mi_protocol` follows the protocol recommended in [Training protocol](options.md#training-protocol):
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
[SPD batch norm](options.md#spd-batch-norm), Study). Nothing is left to do by hand until the
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
the calibration of the first is finished, or submit `mi_study`, which runs both. `bash condor/submit_dag.sh smoke_auto --flash --cpus 1 --memory 2GB`
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
of each dataset is in the table of [Data](data.md#data).

#### Step by step

`--step N` puts only step N in the DAG (its number in `pipeline status`, `N3_...`), with
the same retries, selection and grid extension. It refuses a step whose predecessor in
its chain is not finished. Check the results of each step before the next one:

```bash
bash condor/submit_dag.sh mi_study --step 0 --cpus 1 --memory 2GB   # ReEig floors
python -m rfl.experiments.pipeline status mi_study                  # N0 done?
bash condor/submit_dag.sh mi_study --step 1 --cpus 1 --memory 2GB   # calibration, sweep 1
python -m rfl.experiments.grid mi_spdnet_bn_calib --stage sweep1 --status
```

The chains are independent: the EEGNet and TS+LR steps can run while SPDNet goes on.
Short steps (the floors, TS+LR) fit the flash class (`--flash`, one hour per job).

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
| `--flash`, `--short`, `--long` | 4 days (8 days for GPU jobs) | MUST duration classes: 1 hour, 2 days, 8 days. Flash jobs use reserved slots of one core and 2GB, often free when the cluster is full |
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
