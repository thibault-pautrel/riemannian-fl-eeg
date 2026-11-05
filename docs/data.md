# Data, outputs and analysis

[Back to the README](../README.md)

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
   (`<MNE_DATA>/rfl-stats/<Name>.json`, `rfl/data/stats.py`), as it loads each subject. With
   this file, `--check` refuses a DP batch size above the smallest client training
   set, before any run starts. Each run also prints `[Client i] subjects [...]:
   train=...` at its start.
5. **Study.** Copy `rfl/experiments/plans/template_new_dataset.py`, set the dataset, the
   arms and the search space (section [6. Write your own plan](cluster.md#6-write-your-own-plan)),
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
