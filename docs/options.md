# Options

[Back to the README](../README.md)

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
