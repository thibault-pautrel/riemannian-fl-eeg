# Design, equivalence with the papers, limitations

[Back to the README](../README.md)

## How the code is organised

1. **A client is a group of subjects.** One split of all trials (75/10/15, seeded by
   `--seed`). Each client keeps its subjects (`--subjects-per-client`, default 1). A
   centralised run of the same seed uses the same trials.
2. **A round** is broadcast, local training (`--local-epochs`), aggregation
   (`rfl/federated/aggregation.py`), evaluation on every client.
3. **SPDNet weights live on a manifold.** A BiMap weight satisfies W^T W = I (Stiefel).
   The plain average leaves the manifold, so SPDNet never uses FedAvg. ProjAvg takes the
   polar factor of the average (default). RLAvg and RetAvg average in a tangent space.
   On a client, W stays on the manifold by a polar projection after each step
   (`--stiefel projection`, default, the only choice under DP) or by the library's
   orthogonal parametrisation (`--stiefel trivialization`). The server sees W in both
   cases.
4. **Personalisation decides what stays on the client:** personal head, FedBN layers,
   local-only models, head re-fit.
5. **Privacy changes the local step only:** per-sample clipping, averaging, Gaussian
   noise on every parameter that trains.

One package, `rfl`. Entry points: `python -m rfl.experiments.<name>`.

## Adding a mean on the Stiefel manifold

The server looks up its rule by name in `AGGREGATION_RULES`
(`rfl/federated/aggregation.py`). A new mean on St(n, p) is one function, registered
once. Nothing else changes: the server, `--aggregation`, `FLConfig`, FedBN, FedPer and DP
pick it up.

| File | What to do |
|---|---|
| `rfl/federated/my_means.py` (new) | Write the mean |
| `rfl/federated/aggregation.py`, end of file | Register it |
| `tests/test_aggregation.py` | Add its tests |

```python
# rfl/federated/my_means.py
import torch

def grassmann_orthogonal_mean(Ws, weights, base):
    """Ws: list of client BiMap weights, each a (n, p) torch tensor with W^T W = I.
    weights: list of floats summing to 1. base: the current global matrix W_t.
    Returns one (n, p) matrix with W^T W = I."""
    ...
```

```python
# end of rfl/federated/aggregation.py
from rfl.federated.my_means import grassmann_orthogonal_mean
register_stiefel_mean("gromean", grassmann_orthogonal_mean,
                      "Grassmann mean of the subspaces, then mean in O(p)")
```

The mean is applied to each BiMap weight separately. Every other parameter is averaged
in Euclidean space, and the FedBN keys stay local. `rfl/geometry/stiefel.py` has the
tools: `polar_projection`, `tangent_projection`, `retraction`, `lifting`,
`inverse_polar_retraction`. `rfl/geometry/grassmann.py` has those of the Grassmann
manifold (NumPy).

The tests a mean should pass, as for ProjAvg (`tests/test_aggregation.py`): the result
is on St(n, p), equal client matrices give that matrix back, a client with more weight
pulls harder, and `get_rule("gromean").aggregate(...)` returns the keys of ProjAvg.

```bash
python -m pytest -q tests/test_aggregation.py
python -m rfl.data.synthetic --n-clients 4 --samples-per-client 200 --n-channels 12 \
    --n-classes 2 --mode iid --signal-strength 1.0 --output results/synth.npz
python -m rfl.experiments.run_federated_experiment --synthetic-clients results/synth.npz \
    --model spdnet --hidden-dim 6 --rounds 10 --batch-size 32 --learning-rate 1e-2 \
    --aggregation gromean --experiment-name gromean
```

A plan compares rules with `aggregations={"spdnet": ("projavg", "gromean")}`. DP is
unaffected: the server rule is post-processing.

## Equivalence with the paper repositories

On synthetic data these runs give the same round histories (losses, accuracies, learning
rates, geometry diagnostics, selected round, sigma) as the paper repositories:

| Repository | Settings checked |
|---|---|
| `dp-rfedproj` | SPDNet ProjAvg, private RLAvg (eps 10), private RetAvg with half participation (eps 5), EEGNet without BatchNorm with and without DP, centralised SPDNet |
| `icassp-spdnet-fl` | SPDNet shared, FedPer, FedPer with frozen trunk, local only, two blocks, plateau schedule with early stopping, EEGNet with FedBN, centralised SPDNet with frozen trunk |

The checks use library commit `5b25c30`. The pinned `5c08ebc` moves SPDNet numbers by
about 1e-7. Frozen parameters are now exactly constant (about 1e-6 change with a frozen
trunk). Accuracies are identical on every run checked.

**Changed flags.** ICASSP `--local-fc --spdnet-federation-mode bimap_only` is now
`--local-fc`. EEGNet BatchNorm needs `--eegnet-norm-type batch`. SPDNet batch norm sits
after every block, in float64. SPDNet defaults to ProjAvg and refuses FedAvg. The
defaults follow the DP paper (cosine, last round, one local epoch). `--subjects-per-client`
defaults to 1 (the DP paper used 3 for BNCI2014_001, 2 for Cho2017 and Weibo2014).

**Corrections after the merge**, which change some runs of the paper repositories:

* The centralised plateau schedule uses the federated rule (`rfl/utils/schedules.py`).
* `--task binary` keeps left versus right hand, by event name.
* A centralised run resumes only a checkpoint of its own configuration.
* DP clip and noise act in float64 tangent coordinates, with independent noise and
  minibatch streams: an exact Gaussian mechanism in float32 too.
* The `step` schedule keeps the base rate for the first 70 % of the rounds.
* `--local-only --report selected` reports each client's own best round.
* EEGNet BatchNorm starts with untouched running statistics.
* EEGNet with `projavg`, `rlavg` or `retavg` is refused.

## Limitations

* MOABB downloads from the original hosts, which can be slow or unavailable.
* GPU runs are deterministic up to some CUDA kernels.
* The DP guarantee is record-level, for the training records, the split taken as given.
  Validation-driven choices (plateau, early stopping, selected round) are outside it.
* The library `Vech` backward (`5c08ebc`) copies the gradient of each off-diagonal entry
  to both triangles (gradient cosine about 0.975). `test_library_vech_backward` records
  it as an expected failure.
* The library `Whitening` backward prints its gradients at every step.
  `rfl/utils/log_filter.py` removes them from `run.log` (`RFL_RAW_LOG=1` keeps them). For
  a run by hand: `... | python -m rfl.utils.log_filter`.
* Long private runs can need several GB for the accountant.
