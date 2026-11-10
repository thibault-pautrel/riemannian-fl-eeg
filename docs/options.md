# Options

[Back to the README](../README.md)

## Options

`python -m rfl.experiments.run_federated_experiment --help` lists every option.
`--print-config` prints the resolved configuration and exits:

```bash
python -m rfl.experiments.run_federated_experiment --dataset Cho2017 --local-fc --print-config
```

### Personalisation

| Option | Effect |
|---|---|
| (none) | Trunk and head shared and aggregated. |
| `--local-fc` | Personal head (FedPer). Only the trunk is aggregated. |
| `--local-head-mode frozen` | The personal head stays at its (common) initialisation during the rounds. Use with a re-fit. Default `train`. |
| `--head-refit-epochs 25,50,100,200` | After the last round, each client fits its head on its data, the rest frozen. Candidate (epochs, weight decay) chosen per client on validation (`--head-refit-weight-decays`, `--head-refit-selection`, `--head-refit-lr`). |
| `--local-only` | No aggregation. Each client trains alone. |
| `--freeze-trunk` | Everything but the head stays at its random initialisation. Random-feature control. |
| `--eegnet-norm-type batch` | EEGNet with BatchNorm. Non-private only. |
| `--use-batch-norm` | SPDNet with one SPD batch norm per block. Non-private only. |
| `--no-fedbn` | Share the normalisation layers. Default FedBN: they stay on each client. |
| `--client-partition iid` | IID clients: the trials of each split dealt at random to the same number of clients. Default `subject`. |

Result tables call the scheme `scope`: `shared`, `fedper`, `fedper_frozen`, `local`,
with `_rnd` for a frozen trunk and `+refit` with a re-fit.

### Privacy

| Option | Effect |
|---|---|
| `--dp-enabled` | Record-level DP of everything the clients send. |
| `--dp-target-eps`, `--dp-target-delta` | Budget (default 10, 1e-5). sigma is calibrated before training. |
| `--dp-clip-C` | Clip constant (default 1). |
| `--dp-accounting prirfed` | PriRFed accountant (baseline). |
| `--dp-compare-prirfed` | Also record the sigma PriRFed would need. |

One private local step, on a minibatch of B records drawn without replacement:

```
g~ = (1/B) sum_l clip_C( P_x grad f(x; z_l) ) + P_x xi,    xi ~ N(0, sigma^2 I)
```

P_x is the tangent projection for BiMap weights, the identity elsewhere. Clip and noise
act in orthonormal tangent coordinates at uf(x), in float64
(`rfl/geometry/stiefel.py:StiefelChart`): an exact Gaussian mechanism of sensitivity
2C/B. A non-finite per-sample gradient contributes zero. The accountant composes
`T * local_epochs * ceil(m_max / B)` steps at rate `B / m_min`, a bound for every client.
`fl_results.json` records the realised budget (`dp_budget`).

What the guarantee covers:

* Every parameter that trains is privatised, a personal head in `train` mode included.
  Frozen parts stay out of the clipped vector.
* The head re-fit sends nothing. The re-fitted head itself is not protected.
* With `--local-only`, DP protects each client's own model.
* Participation is visible: `--client-fraction` gives no amplification.
* The split is taken as given. Per-round training metrics, validation and test metrics
  are diagnostics outside the guarantee.

The runners refuse batch norm under DP and global normalisation under DP. They warn
when DP protects less than the transcript (local only, frozen trunk with personal head).

### SPD batch norm

Gallet, Mian, Bouchard and Ginolhac, *On batch normalization for SPDnet* (Signal
Processing 250, 2027, hal-05742151), layers of yetanotherspdnet used as they are. A batch
{S_k} with batch mean G becomes `Gamma G^{-1/2} S_k G^{-1/2} Gamma`, with a learned SPD
bias Gamma. At evaluation G is replaced by the running mean R. `--use-batch-norm` adds
one layer after the ReEig of each block, in float64.

| Option | Default | Choices |
|---|---|---|
| `--bn-mean-type` | `geometric_arithmetic_harmonic` (GAH) | `affine_invariant`, `log_euclidean`, `arithmetic`, `harmonic`, `adaptive_geometric_arithmetic_harmonic`, `bures_wasserstein` |
| `--bn-momentum` | 0.1 | update rate of R (also of the EEGNet BatchNorm statistics) |
| `--bn-type` | `mean_only` | `mean_var_scalar` (also a scalar dispersion) |
| `--bn-norm-and-bias-mode` | `whitening` | `parallel_transport` |
| `--bn-position` | `after_reeig` | `before_reeig` (Brooks et al., 2019) |
| `--bn-recalibrate` | off | after training, set R to the exact mean of the training inputs |

* **GAH** is `A #_{1/2} H`, the geodesic midpoint of the arithmetic and harmonic means:
  closed form, invariant under congruence and inversion. The library moves R along the
  GAH curve at each batch. The paper (Eq. 13) keeps two running means instead. This
  repository follows the library.
* **Momentum.** 0.1 against 0.01 in the library: a round runs few local steps.
* **`--bn-recalibrate`** ("precise BN", Wu and Johnson, 2021). With few data R moves
  little. The option sets R, block after block, to the exact mean of the layer inputs
  in evaluation mode. Shared BN: the server combines the client means as it combines
  running means. `final_metrics_before_recalibration` keeps the momentum model.
* **FedBN** (default): R and Gamma stay on each client. **Shared** (`--no-fedbn`): the
  server sets R to the weighted mean of the client running means in the layer geometry
  (GAH: `A #_{1/2} H` of the weighted means), and averages the parameter of Gamma.
  Offered for `mean_only` with the five closed-form or Karcher means. EEGNet: FedAvg of
  its BatchNorm parameters and statistics.
* R is estimated with the local trunks and used with the aggregated one: it lags by one
  aggregation, as in FedBN (Li et al., 2021).
* After a batch norm the spectrum sits near Gamma^2, so later ReEig floors must be
  recalibrated (the paper uses 0.01 to 0.8). `reeig_activity` shows whether a floor acts.
* A last batch of one matrix is dropped. DP refuses the layer.

**Study** (`mi_spdnet_bn_calib`, `mi_spdnet_bn`). Normalisation (none, FedBN, shared) x
scope (shared head, FedPer, local) x clients (subject, IID), plus centralised with and
without BN: 18 arms. Plain SGD, `--stiefel trivialization`, one local epoch per round,
ProjAvg.

1. **Floors.** The q-th percentile of the spectrum each ReEig acts on, per candidate
   architecture, with and without BN (`analyze_covariances --architectures`, read with
   `--reeig-percentile q`).
2. **Calibration** on the centralised SPDNet: lr x batch, then q (5, 15, 30) and the
   momentum. The best architecture is kept. Schirrmeister2017 tries four two-block
   architectures.
3. **Every arm**: lr x batch (12 cells, horizon 200), then the horizon (100, 200, 300),
   the shortest within one standard error of the best. Mean validation accuracy over 3
   seeds.
4. **Extension.** A choice on the edge of its grid adds one value beyond it, within the
   bounds of `EXTEND` (`mi_spdnet_bn_calib.py`).
5. **Final** on 10 other seeds. The seed draws the split, the same for every arm. Last
   round reported.

**The same study for EEGNet** (`mi_eegnet_bn_calib`, `mi_eegnet_bn`): BatchNorm, plain
SGD, FedAvg. The calibration chooses the BatchNorm momentum (0.1, 0.01, 0.3) after lr x
batch. Steps 3 to 5 as above. In both studies the test statistics are the momentum
averages of training.

`test_acc` pools the predictions of all clients. `client_mean_acc` averages the clients
and `client_min_acc` is the worst one. On IID clients, FedBN and a shared BN should agree.

### Data quantity

Each subject keeps N calibration trials per class (`rfl/data/reduction.py`), from its
training and validation trials. Test trials are unchanged.

| Option | Default | Effect |
|---|---|---|
| `--train-trials-per-class N` | all | N trials per class and subject |
| `--calibration-val-fraction` | 0.2 | share of the N trials for validation (at least 1) |
| `--calibration-order` | `random` | or `chronological` (the first trials recorded) |
| `--reduced-subjects K` | 0 (all) | reduce K subjects drawn with the seed |

Subsets are nested and stratified. The results add `reduced_val_acc`,
`reduced_test_acc` and `reduced_refit_test_acc` (mean over the reduced clients), with
`n_per_class` and `reduced_subjects`. TS+LR (`rfl/experiments/run_tslr.py`): tangent
space at each client's mean then logistic regression, per client (`--mode local`) or on
all clients re-centred (`--mode pooled`).

**Study** (`mi_datasize`, `mi_datasize_eegnet`, `mi_datasize_tslr`,
`rfl/experiments/datasize.py`):

1. Scenarios: `U<N>` every subject has N trials per class, `H<N>` a third of them,
   `full` no reduction.
2. Arms: head (shared, shared + re-fit, FedPer, local) x normalisation (none, FedBN,
   shared), for SPDNet and EEGNet, with `--bn-recalibrate`, plus TS+LR.
3. Each arm tuned on its own (lr x batch, then horizon) on `reduced_val_acc`. The
   horizon grid grows by sqrt(40 / N), at most x4.
4. Final on 10 seeds, which draw the split, the reduction and the reduced subjects.

| Arm | Head | Options |
|---|---|---|
| `..._shared_<norm>_<scen>` | shared, + re-fit in the final stage | (none) |
| `..._fedper_<norm>_<scen>` | personal | `--local-fc` |
| `fed_<model>_local_<norm>_<scen>` | none shared | `--local-only` |

| `<norm>` | SPDNet | EEGNet |
|---|---|---|
| `nobn` | (none) | `--eegnet-norm-type none` |
| `bn` | `--use-batch-norm --bn-recalibrate` | `--eegnet-norm-type batch --bn-recalibrate` |
| `bnshared` | `... --no-fedbn` | `... --no-fedbn` |

```bash
export RFL_DATASET=BNCI2014_001
bash condor/submit_dag.sh mi_datasize --memory 12GB
python -m rfl.analysis.datasize $RFL_RESULTS/mi_BNCI2014_001_datasize*/final \
    --out datasize_BNCI2014_001 --pairs
```

`rfl/analysis/datasize.py` writes a table and one figure per model: `reduced_test_acc`
against N, one panel per scenario, colour for the head, line style for the
normalisation. `--pairs` adds the per-seed gain of each federated curve over the best
local one. With `report` and `stats`, also group by `n_per_class reduced_subjects`.
`synth_datasize` runs the protocol on synthetic subjects (`RFL_QUICK=1`: wiring check).

### Checks before a run

The runners stop before loading data when an option has no effect or a value is
impossible (`rfl/experiments/cli_checks.py`). Examples:

| Command | Message |
|---|---|
| `--dp-target-eps 5` without `--dp-enabled` | only acts with `--dp-enabled` |
| `--bn-mean-type harmonic` without `--use-batch-norm` | only acts with `--model spdnet --use-batch-norm` |
| `--model eegnet --hidden-dim 8` | only acts with `--model spdnet` |
| `--lr-plateau-patience 5` with the cosine schedule | only acts with `--lr-schedule plateau` |
| `--local-only --aggregation rlavg` | no effect with `--local-only` |
| `--no-fedbn` without a normalisation layer | needs a normalisation layer |

Also refused: negative weight decay, `fmin >= fmax`, `fmax` above Nyquist, BiMap
dimensions above the channel count, a DP batch above the smallest client training set.
`--device auto` (default) uses the GPU when there is one. The defaults of the runners are
those of `rfl/utils/config.py` and `rfl/federated/fl_config.py`.

### Training protocol

Defaults, those of the DP paper: one local epoch per round, Adam (`--optimizer sgd`,
`--momentum`), rebuilt at every round, cosine schedule from `--learning-rate` to 1/100 of
it over a fixed horizon, no early stopping, last round reported.

* One rule for centralised, federated and private runs, so a comparison changes one
  factor.
* Under DP the horizon must be fixed in advance.
* The reported model is the one a deployment would use.

Validation only selects the hyperparameters (learning rate, batch, horizon T), on the
mean over sweep seeds of the last-round metric. Final results use other seeds. Tune each
family of arms separately (centralised, federated, private, local-only).
`--head-refit-selection fixed` gives a re-fit with no validation use.

The ICASSP protocol, for quick exploration:
`--lr-schedule plateau --early-stop-patience N --selection-metric val_loss --report selected`.

### SPDNet depth and ReEig floors

`--hidden-dim` and `--epsilon` default to `rfl/data/registry.py`.
`--hidden-dims 16,8 --epsilons 1e-4,1e-2` builds two blocks. Without batch norm the floors
must strictly increase, or a ReEig is the identity (Poincare separation, Oswald and
Massart, arXiv:2606.02490). `reeig_activity` in `fl_results.json` gives the relative
change of each ReEig (near 0: inactive).
`python -m rfl.experiments.analyze_covariances --hidden-dim-block1 16` calibrates two
increasing floors.
