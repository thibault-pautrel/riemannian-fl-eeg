# Options and protocols

[Back to the README](../README.md). `--help` lists every option of a runner, and
`--print-config` prints the resolved configuration without running.

## Personalisation

| Option | Effect |
|---|---|
| `--local-fc` | personal head (FedPer), only the trunk is aggregated |
| `--local-head-mode frozen` | the personal head stays at its initialisation (use with a re-fit) |
| `--head-refit-epochs 25,50,100` | after training, each client re-fits its head, candidate chosen on its validation data |
| `--local-only` | no aggregation, each client trains alone |
| `--freeze-trunk` | the trunk stays at its random initialisation (random-feature control) |
| `--no-fedbn` | share the normalisation layers (default: kept on each client, FedBN) |
| `--client-partition iid` | trials dealt at random to the clients instead of one subject per client |

## SPD batch norm

`--use-batch-norm` adds the SPD batch norm of Gallet et al. (*On batch normalization for
SPDnet*, Signal Processing, 2027) after each SPDNet block, from yetanotherspdnet. It
centres each batch on its mean G and moves it to a learned SPD bias Gamma. At test time
G is a running mean.

| Option | Default |
|---|---|
| `--bn-mean-type` | `geometric_arithmetic_harmonic`, or `affine_invariant`, `log_euclidean`, `arithmetic`, `harmonic`, ... |
| `--bn-momentum` | 0.1, update rate of the running statistics (also for EEGNet) |
| `--bn-recalibrate` | off. After training, set the running statistics to their exact value on the training data |

A shared batch norm (`--no-fedbn`) combines the client running means in the geometry of
the layer. Batch norm is refused under DP.

## Differential privacy

`--dp-enabled` privatises every local step: per-sample gradients projected on the
tangent space, clipped to `--dp-clip-C` (default 1), averaged, plus Gaussian noise.
sigma is calibrated before training for `--dp-target-eps` and `--dp-target-delta`
(default 10 and 1e-5). Every parameter that trains is covered, a personal head included.
The guarantee is record-level for the training records, the split taken as given.

## Data quantity

| Option | Effect |
|---|---|
| `--train-trials-per-class N` | keep N calibration trials per class and subject (test trials unchanged) |
| `--reduced-subjects K` | reduce only K subjects, the others keep all their trials |
| `--calibration-val-fraction` | share of the N trials used for validation (default 0.2) |

Results then add `reduced_test_acc`, the mean accuracy of the reduced subjects.

## Data

Trials are band-pass filtered (8 to 32 Hz) and resampled to 128 Hz. SPDNet uses one
covariance matrix per trial, EEGNet the raw trials. The split (75/10/15, stratified,
seeded by `--seed`) is the same for centralised and federated runs. The preprocessed
trials are cached in `$MNE_DATA/rfl-cache` (`RFL_DATA_CACHE=off` disables it).

To add a dataset: an entry in `DATASET_CONFIGS` and `MNE_PATH_KEYS`
(`rfl/data/registry.py`), then its ReEig floor from
`python -m rfl.experiments.analyze_covariances --datasets <Name>`.

## Protocol

Defaults: one local epoch per round, cosine learning-rate schedule over a fixed horizon,
last round reported. Validation only chooses the hyperparameters, and final results use
other seeds. A plan sweeps, selects and runs the final arms:

```bash
python -m rfl.experiments.grid mi_protocol --stage sweep --check     # validate the runs
python -m rfl.experiments.grid mi_protocol --stage sweep --run --jobs 8
python -m rfl.experiments.grid mi_protocol --stage sweep --select
python -m rfl.experiments.grid mi_protocol --stage final --run --jobs 8
```

`python -m rfl.experiments.pipeline run <plan>` chains these steps with automatic grid
extension. To write a plan, copy `rfl/experiments/plans/template_new_dataset.py`.

## Outputs and analysis

A federated run writes `fl_results.json` (configuration, metrics, round history) and the
final models. `test_acc` pools the predictions of all clients, `client_mean_acc`
averages the clients.

```bash
python -m rfl.analysis.report results --by dataset arch scope --pivot eps
python -m rfl.analysis.report results --curves figures/
python -m rfl.analysis.stats results --factor scope --baseline shared --by dataset arch
```
