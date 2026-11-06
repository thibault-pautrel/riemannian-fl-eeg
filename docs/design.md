# Design, equivalence with the papers, limitations

[Back to the README](../README.md)

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
[Repository structure](../README.md#repository-structure)). Installed with `pip install -e .`, it
runs from any folder.

## Adding a mean on the Stiefel manifold

The server looks up the aggregation rule by name in `AGGREGATION_RULES`
(`rfl/federated/aggregation.py`), and so do `FLConfig` and `--aggregation`. A new mean
on St(n, p) is one function of the client matrices, their normalised weights and the
current global matrix:

```python
# rfl/federated/my_means.py
def grassmann_orthogonal_mean(Ws, weights, base):
    """Ws: client BiMap weights, each (n_in, n_out) with W^T W = I. Returns one."""
    ...
```

registered at the end of `aggregation.py`:

```python
from rfl.federated.my_means import grassmann_orthogonal_mean
register_stiefel_mean("gromean", grassmann_orthogonal_mean,
                      "Grassmann mean of the subspaces, then mean in O(p)")
```

The rule then applies to every BiMap weight, the other parameters are averaged in
Euclidean space, FedBN and FedPer are unchanged, and a run takes
`--aggregation gromean`. A plan compares it with the others through
`aggregations={"spdnet": ("projavg", "gromean")}`. Differential privacy is unaffected:
the clients privatise their updates, and the server rule is post-processing.
`tests/test_aggregation.py` shows the checks a new mean should pass (result on the
manifold, consensus recovered, weights respected).

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
  [Training protocol](options.md#training-protocol)).
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
  first on the cluster (section [Running on MUST](cluster.md#running-on-must-htcondor)).
* Long private runs compose tens of thousands of steps in the accountant, which can need
  several GB of memory.
