# Data, outputs and analysis

[Back to the README](../README.md)

## Data

MOABB motor-imagery datasets of `rfl/data/registry.py`: BNCI2014_001, Weibo2014,
Cho2017, Schirrmeister2017, PhysionetMI (subjects 88, 92 and 100 excluded, as in the
literature), Stieger2021, Lee2019_MI.

```bash
python -m rfl.experiments.fetch_dataset --dataset BNCI2014_001 --data-dir $HOME/mne_data
```

**Folder.** `--data-dir`, else `MNE_DATA`, else `~/mne_data`. The code sets every MOABB
path key in the environment, so stale entries of `~/.mne/mne-python.json` are ignored
(and printed). Symbolic links are resolved. Each run prints `MNE data folder: ...`.

**Preprocessing.** Band-pass 8 to 32 Hz, resampling to 128 Hz. SPDNet takes one sample
covariance per trial, EEGNet the raw trials standardised per channel. The split
(75/10/15, stratified, seeded by `--seed`) is drawn once over all trials. Each client
keeps the trials of its subjects (`--subjects-per-client`, default 1). Centralised and
federated runs of a seed use the same trials. Channels missing for one subject are
dropped for all, with a warning.

**Memory per job** (MUST kills a job above 1.5 x its request):

| Dataset | Subjects | Channels | Trials in memory | `--memory` |
|---|---|---|---|---|
| BNCI2014_001 | 9 | 22 | 0.5 GB | 8GB |
| Weibo2014 | 10 | 60 | 1.4 GB | 8GB |
| PhysionetMI | 106 | 64 | 2 GB | 8GB |
| Cho2017 | 52 | 64 | 2.5 GB | 12GB |
| Schirrmeister2017 | 14 | 128 | 7.5 GB | 32GB |

### Trial cache

The first load writes the preprocessed trials of each subject to
`<MNE_DATA>/rfl-cache/<dataset>/<key>/subject_<id>.npz` (`rfl/data/cache.py`). Later
loads read them back. `fetch_dataset` and `condor/fetch_data.sh` fill it, so the jobs
only read.

* The key hashes the classes, band, rate, window, dataset options and MOABB/MNE versions
  (`params.json`). Other settings give another key.
* Exact arrays (float64): same results with or without the cache. About 0.4 GB for
  BNCI2014_001, 5 GB for Schirrmeister2017.
* Atomic writes: parallel jobs never read a partial file.
* `RFL_DATA_CACHE=off` disables it, `RFL_DATA_CACHE=refresh` rewrites it.

A dataset whose raw files were deleted needs `fetch_data.sh` again before its next study.

```bash
python -m rfl.experiments.run_federated_experiment --dataset BNCI2014_001 --model spdnet \
    --rounds 200 --batch-size 64 --learning-rate 0.01 --local-fc --dp-enabled --dp-target-eps 10
```

### Adding a dataset

1. **Registry.** Add an entry to `DATASET_CONFIGS` in `rfl/data/registry.py` (key: the
   MOABB class name) and its download key to `MNE_PATH_KEYS`.
   `test_mne_path_keys_are_those_moabb_reads` checks both.
2. **Download.** `bash condor/fetch_data.sh <Name>`, or
   `python -m rfl.experiments.fetch_dataset --dataset <Name>`. It also writes
   `<MNE_DATA>/rfl-stats/<Name>.json` (trials per subject), used by `--check`.
3. **SPDNet defaults.** `python -m rfl.experiments.analyze_covariances --datasets <Name>`
   gives the ReEig floor. Add it and the BiMap dimension to `DATASET_DEFAULTS`.
4. **Study.** Copy `template_new_dataset.py`
   ([Write your own plan](cluster.md#6-write-your-own-plan)), then `--check`, sweep,
   `--select`, final.

## Outputs

| File | Content |
|---|---|
| `config.json`, `environment.json` | configuration, library versions, git commit |
| `fl_results.json` | configuration, DP budget, final and re-fit metrics, ReEig activity, per-round history |
| `global_model_final.pt` | last and selected global states |
| `client_states.pt` | what stays on the clients: personal heads, FedBN layers, re-fitted heads, local models |
| `centralized_results.json` | centralised runs |

The global state of the reported round plus `local_states[cid]` rebuild client `cid`'s
model behind `final_metrics`. Main metrics pool the predictions of all clients.

## Analysis

```bash
python -m rfl.analysis.report results --by dataset arch scope --pivot eps     # mean, std, n
python -m rfl.analysis.report results --metric refit_test_acc
python -m rfl.analysis.report results --curves figures/ --curve-figure dataset arch
python -m rfl.analysis.report results --runs runs.csv          # one row per run, for pandas
python -m rfl.analysis.stats results --factor scope --baseline shared --by dataset arch
# in a final folder the shared arms carry the re-fit: --baseline shared+refit
```

`rfl/analysis/results.py` loads every finished run into a table. A group should hold
each seed once: `report` warns and `stats` refuses otherwise. Group by more columns or
filter with `--where`.

## Other tools

| Command | Purpose |
|---|---|
| `python -m rfl.experiments.run_centralized_experiment` | Centralised reference on the same split |
| `python -m rfl.experiments.accounting_comparison` | Noise of our accountant against PriRFed |
| `python -m rfl.federated.privacy.verify_accountant` | Checks of the accountant (minutes) |
| `python -m rfl.experiments.synthetic_pca all --quick` | Private federated PCA on the Grassmann manifold |
| `python -m rfl.experiments.analyze_covariances` | ReEig floors from the covariance spectra |
