"""Tests for rfl/data/registry.py, rfl/data/datasets.py, rfl/data/data_utils.py and rfl/data/synthetic.py.

No MOABB download: only the pure-NumPy parts are exercised.
"""
import numpy as np
import pandas as pd
import pytest

from rfl.data.data_utils import create_splits, filter_binary_task, resolve_resample
from rfl.data.datasets import build_dataset, make_loader
from rfl.data.registry import DATASET_CONFIGS, get_config, get_sampling_rate, get_subjects
from rfl.data.synthetic import (
    generate_synthetic_spd, load_clients, load_dataset, make_class_scale_matrices,
    pool_clients, sample_wishart_spd, save_dataset,
)


# ---- registry -----------------------------------------------------------------
def test_registry_entries_are_consistent():
    for name, cfg in DATASET_CONFIGS.items():
        assert cfg["n_classes"] == len(cfg["events"]), name
        assert get_sampling_rate(name) > 0
        assert len(get_subjects(name)) > 0


def test_registry_unknown_names():
    assert get_sampling_rate("synthetic", default=None) is None
    with pytest.raises(KeyError):
        get_config("NoSuchDataset")


# ---- datasets -----------------------------------------------------------------
def _spd(n=5, d=8, seed=0):
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, d, 20)
    return (Z @ Z.transpose(0, 2, 1) / 20 + 1e-6 * np.eye(d)).astype(np.float32)


def test_spd_dataset_passes_matrices_through():
    spd = _spd()
    x0, y0 = build_dataset(spd, np.arange(5), "spd")[0]
    assert np.allclose(x0.numpy(), spd[0]) and int(y0) == 0


def test_covariance_dataset_is_spd():
    raw = np.random.RandomState(0).randn(5, 8, 100).astype(np.float32)
    c0, _ = build_dataset(raw, np.zeros(5), "covariance")[0]
    c0 = c0.numpy()
    assert c0.shape == (8, 8)
    assert np.allclose(c0, c0.T)
    assert (np.linalg.eigvalsh(c0) > 0).all()


def test_covariance_shrinkage_matches_formula():
    raw = np.random.RandomState(1).randn(2, 4, 50).astype(np.float32)
    S = build_dataset(raw, np.zeros(2), "covariance", regularization=0.0)[0][0].numpy()
    a = 0.3
    L = build_dataset(raw, np.zeros(2), "covariance", estimator="lwf", shrinkage=a,
                      regularization=0.0)[0][0].numpy()
    expected = (1 - a) * S + a * np.trace(S) / 4 * np.eye(4)
    assert np.allclose(L, expected, atol=1e-5)


def test_covariance_dataset_rejects_spd_input():
    with pytest.raises(ValueError):
        build_dataset(np.zeros((3, 4)), np.zeros(3), "covariance")


def test_raw_dataset_per_trial_normalisation():
    raw = (np.random.RandomState(2).randn(4, 3, 200) * 5 + 2).astype(np.float32)
    ds = build_dataset(raw, np.zeros(4), "raw")
    x, _ = ds[0]
    assert x.shape == (1, 3, 200)
    assert np.allclose(x.numpy().mean(-1), 0, atol=1e-5)
    assert np.allclose(x.numpy().std(-1), 1, atol=1e-3)


def test_build_dataset_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_dataset(_spd(), np.zeros(5), "image")


def test_make_loader_doubles_eval_batch():
    ds = build_dataset(_spd(), np.zeros(5), "spd")
    assert make_loader(ds, 2, shuffle=True).batch_size == 2
    assert make_loader(ds, 2, shuffle=False).batch_size == 4


# ---- data_utils ---------------------------------------------------------------
def test_resolve_resample_never_upsamples():
    assert resolve_resample(128, 250) == 128.0
    assert resolve_resample(500, 250) == 250
    assert resolve_resample(None, 250) == 250
    assert resolve_resample(0, 250) == 250
    assert resolve_resample(128, None) == 128


def _toy_eeg(n_per_class=20, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    X = rng.randn(len(y), 4, 16)
    meta = pd.DataFrame({"subject": np.tile([1, 2], len(y) // 2),
                         "session": np.repeat(["0train", "1test"], len(y) // 2)})
    return X, y, meta


def test_create_random_split_fractions_and_disjointness():
    X, y, meta = _toy_eeg()
    out = create_splits(X, y, meta, split_type="random", val_size=0.1, test_size=0.2,
                        random_state=0, verbose=False)
    sizes = [len(out[k][1]) for k in ("train", "val", "test")]
    assert sum(sizes) == len(y)
    assert sizes[2] == 12 and sizes[1] == 6


def test_create_split_is_deterministic():
    X, y, meta = _toy_eeg()
    a = create_splits(X, y, meta, split_type="random", random_state=3, verbose=False)
    b = create_splits(X, y, meta, split_type="random", random_state=3, verbose=False)
    assert np.array_equal(a["test"][0], b["test"][0])


def test_create_session_split_uses_sessions():
    X, y, meta = _toy_eeg()
    out = create_splits(X, y, meta, split_type="session", random_state=0, verbose=False)
    assert len(out["test"][1]) == (meta["session"] == "1test").sum()


def test_filter_all_classes_is_identity():
    X, y, meta = _toy_eeg()
    X2, y2, _ = filter_binary_task(X, y, meta, task="all_classes", verbose=False)
    assert X2 is X and y2 is y


def test_filter_binary_gives_two_labels_0_and_1():
    X, y, meta = _toy_eeg()
    _, y2, meta2 = filter_binary_task(X, y, meta, task="binary", verbose=False)
    assert set(y2) == {0, 1} and len(meta2) == len(y2)


def test_filter_no_rest_drops_rest_and_remaps():
    # Weibo2014 events sorted: feet, hands, left_hand, left_hand_right_foot, rest,
    # right_hand, right_hand_left_foot -> rest = 4.
    X, y, meta = _toy_eeg(n_classes=7)
    X2, y2, _ = filter_binary_task(X, y, meta, task="no_rest",
                                   dataset_name="Weibo2014", verbose=False)
    assert len(y2) == 120 and set(y2) == set(range(6))
    assert np.array_equal(X2, X[y != 4])


def test_filter_rejects_unknown_task():
    X, y, meta = _toy_eeg()
    with pytest.raises(ValueError):
        filter_binary_task(X, y, meta, task="three_way", verbose=False)


# ---- synthetic ----------------------------------------------------------------
def test_class_scale_matrices_are_spd_and_distinct():
    scales = make_class_scale_matrices(8, 3, seed=0)
    assert len(scales) == 3
    for S in scales:
        assert (np.linalg.eigvalsh(S) > 0).all()
    assert not np.allclose(scales[0], scales[1])
    with pytest.raises(ValueError):
        make_class_scale_matrices(4, 3, signal_rank=2)


def test_wishart_draws_have_the_right_mean():
    S = make_class_scale_matrices(4, 2, seed=1)[0]
    draws = sample_wishart_spd(S, df=40, n_samples=400, rng=np.random.RandomState(0))
    assert draws.shape == (400, 4, 4) and draws.dtype == np.float32
    assert np.allclose(draws.mean(0), S, atol=0.1)


def test_generate_synthetic_spd_shapes_and_seed():
    kw = dict(n_clients=3, samples_per_client=30, n_channels=6, n_classes=2, seed=5)
    a, b = generate_synthetic_spd(**kw), generate_synthetic_spd(**kw)
    assert len(a["clients"]) == 3
    c = a["clients"][0]
    assert c["X_train"].shape[1:] == (6, 6) and c["X_train"].dtype == np.float32
    # 30 / (1 - 0.10 - 0.15) = 40 samples, split 30 / 4 / 6.
    assert (len(c["y_train"]), len(c["y_val"]), len(c["y_test"])) == (30, 4, 6)
    assert np.array_equal(a["clients"][1]["X_test"], b["clients"][1]["X_test"])
    assert a["metadata"]["label_skew_alpha"] == 0.5          # noniid preset


def test_iid_mode_is_balanced():
    ds = generate_synthetic_spd(n_clients=2, samples_per_client=40, n_channels=6,
                                n_classes=2, mode="iid", seed=0)
    y = np.concatenate([ds["clients"][0][f"y_{s}"] for s in ("train", "val", "test")])
    assert np.bincount(y)[0] == np.bincount(y)[1]


def test_synthetic_save_load_roundtrip(tmp_path):
    ds = generate_synthetic_spd(n_clients=2, samples_per_client=20, n_channels=4,
                                n_classes=2, seed=0)
    path = str(tmp_path / "syn.npz")
    save_dataset(ds, path)
    back = load_dataset(path)
    assert back["metadata"] == ds["metadata"]
    assert np.array_equal(load_clients(path)[1]["X_val"], ds["clients"][1]["X_val"])
    pooled = pool_clients(back)
    assert len(pooled["train"][1]) == sum(len(c["y_train"]) for c in ds["clients"])


def test_binary_task_selects_left_and_right_hand_by_name():
    """Labels follow the alphabetical order of the events, so 0/1 is not left/right."""
    import pandas as pd
    from rfl.data.data_utils import filter_binary_task
    ev = np.array(["feet", "left_hand", "right_hand", "tongue"] * 3)
    y = np.searchsorted(sorted(set(ev)), ev)                   # feet=0, left_hand=1, ...
    X = np.arange(12)[:, None, None] * np.ones((12, 2, 3))
    for meta in (pd.DataFrame({"event": ev}), pd.DataFrame({"subject": [1] * 12})):
        X2, y2, _ = filter_binary_task(X, y, meta, "binary", "BNCI2014_001",
                                       verbose=False)
        kept = X2[:, 0, 0].astype(int)
        assert list(ev[kept]) == ["left_hand", "right_hand"] * 3
        assert list(y2) == [0, 1] * 3


# --------------------------------------------------------------------------- #
#  Location of the MOABB downloads
# --------------------------------------------------------------------------- #
def test_mne_path_keys_are_those_moabb_reads(monkeypatch):
    """Each dataset of the registry reads the key listed in MNE_PATH_KEYS."""
    pytest.importorskip("moabb")
    import moabb.datasets.download as dl

    from rfl.data.registry import (DATASET_CONFIGS, MNE_PATH_KEYS, get_moabb_class,
                               get_subjects)

    class Stop(Exception):
        pass

    seen = []
    orig = dl.get_config

    def fake(key=None, default=None, *a, **k):
        if key and key.startswith("MNE_DATASETS_"):
            seen.append(key)
            raise Stop                       # before any download
        return orig(key, default, *a, **k)

    monkeypatch.setattr(dl, "get_config", fake)
    assert set(MNE_PATH_KEYS) == set(DATASET_CONFIGS)
    for name in DATASET_CONFIGS:
        seen.clear()
        with pytest.raises(Stop):
            get_moabb_class(name)().data_path(get_subjects(name)[0])
        assert seen == [MNE_PATH_KEYS[name]], name


def test_configure_mne_paths_follows_links_and_ignores_the_config_file(tmp_path,
                                                                       monkeypatch):
    mne = pytest.importorskip("mne")
    import json
    import os

    from rfl.data.registry import MNE_PATH_KEYS, configure_mne_paths
    home = tmp_path / "home"
    (home / ".mne").mkdir(parents=True)
    cfg = home / ".mne" / "mne-python.json"
    old = {"MNE_DATA": str(tmp_path / "old"), "MNE_DATASETS_BNCI_PATH": str(tmp_path / "old")}
    cfg.write_text(json.dumps(old))
    real = tmp_path / "shared" / "mne_data"
    real.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(home))
    for k in ["MNE_DATA", *MNE_PATH_KEYS.values()]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MNE_DATA", str(link))
    got = configure_mne_paths(verbose=False)
    assert got == str(real.resolve())
    for k in MNE_PATH_KEYS.values():
        assert os.environ[k] == got
        assert mne.get_config(k) == got                      # env wins over the file
    assert json.loads(cfg.read_text()) == old                 # file never written


# ---- MOABB loading: channels aligned by name ----------------------------------
class _FakeEpochs:
    def __init__(self, X, ch_names):
        self._X, self.ch_names = X, list(ch_names)

    def get_data(self):
        return self._X


class _FakeDataset:
    unit_factor = 1e6
    chans = {}                      # subject -> {session: channel names}
    fail = set()                    # subjects whose load raises

    def __init__(self, **kw):
        self.sessions = None

    def _sessions(self, s):
        return [k for k in self.chans[s] if self.sessions is None or k in self.sessions]


class _FakeParadigm:
    def __init__(self, **kw):
        pass

    def get_data(self, dataset, subjects, return_epochs=False):
        assert return_epochs
        (s,) = subjects
        if s in dataset.fail:
            raise OSError("corrupt file")
        sess = dataset._sessions(s)
        names = [dataset.chans[s][k] for k in sess]
        if any(n != names[0] for n in names):
            raise ValueError("all the input array dimensions must match (channels)")
        n = 4 * len(sess)
        X = np.zeros((n, len(names[0]), 8))
        for j, c in enumerate(names[0]):
            X[:, j, :] = (ord(c[-1]) + 100 * s) * 1e-6     # value encodes subject and channel
        y = np.array(["left_hand", "right_hand"] * (n // 2))
        meta = pd.DataFrame({"subject": s, "session": [str(k) for k in sess for _ in range(4)]})
        return _FakeEpochs(X, names[0]), y, meta


@pytest.fixture
def fake_moabb(monkeypatch):
    import moabb.paradigms as mp

    import rfl.data.data_utils as du
    import rfl.data.stats as st
    monkeypatch.setattr(du, "_MOABB_READY", True)
    monkeypatch.setattr(du, "_get_moabb_class", lambda name: _FakeDataset)
    monkeypatch.setattr(mp, "MotorImagery", _FakeParadigm)
    monkeypatch.setattr(st, "update_stats", lambda *a, **k: None)
    _FakeDataset.fail = set()
    return _FakeDataset


def test_subjects_are_aligned_by_channel_name(fake_moabb):
    from rfl.data.data_utils import load_moabb_dataset
    fake_moabb.chans = {1: {"0": ["C3", "Cz", "C4"]}, 2: {"0": ["C4", "C3", "Cz"]},
                        3: {"0": ["C3", "Cz", "C4", "Pz"]}}
    X, y, meta = load_moabb_dataset("Cho2017", [1, 2, 3], resample=None, verbose=False)
    assert X.shape == (12, 3, 8)                              # Pz dropped, order of subject 1
    for k, subj in enumerate((1, 2, 3)):
        rows = X[4 * k:4 * k + 4, :, 0]
        assert np.allclose(rows, [[ord("3") + 100 * subj, ord("z") + 100 * subj,
                                   ord("4") + 100 * subj]] * 4)
    assert list(meta["subject"]) == [1] * 4 + [2] * 4 + [3] * 4


def test_sessions_with_different_channels_are_loaded_one_by_one(fake_moabb):
    from rfl.data.data_utils import load_moabb_dataset
    fake_moabb.chans = {1: {"1": ["C3", "C4"], "2": ["C4", "C3", "Fz"]}}
    _FakeDataset.__init__ = lambda self, **kw: setattr(self, "sessions", ["1", "2"])
    try:
        X, _, meta = load_moabb_dataset("Cho2017", [1], resample=None, verbose=False)
    finally:
        _FakeDataset.__init__ = lambda self, **kw: setattr(self, "sessions", None)
    assert X.shape == (8, 2, 8) and set(meta["session"]) == {"1", "2"}
    assert np.allclose(X[4:, :, 0], [[ord("3") + 100, ord("4") + 100]] * 4)


def test_a_subject_that_fails_stops_the_load(fake_moabb):
    from rfl.data.data_utils import load_moabb_dataset
    fake_moabb.chans = {1: {"0": ["C3"]}, 2: {"0": ["C3"]}}
    fake_moabb.fail = {2}
    with pytest.raises(OSError, match="corrupt"):
        load_moabb_dataset("Cho2017", [1, 2], resample=None, verbose=False)


def test_no_rest_on_a_dataset_without_rest_is_an_error():
    X = np.zeros((4, 2, 8))
    y = np.array([0, 1, 0, 1])
    meta = pd.DataFrame({"event": ["left_hand", "right_hand"] * 2})
    with pytest.raises(ValueError, match="no 'rest'"):
        filter_binary_task(X, y, meta, task="no_rest", verbose=False)



# ---- trial cache (rfl/data/cache.py) ------------------------------------------
def test_trial_cache_returns_the_same_arrays(tmp_path):
    from rfl.data.cache import load_subject, save_subject
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"subject": np.array([3, 3, 3], dtype=np.int64),
                         "session": np.array(["0train", "0train", "1test"], dtype=object),
                         "run": np.array(["0", "1", "0"], dtype=object)})
    chunks = [(rng.standard_normal((3, 4, 5)), np.array(["feet", "tongue", "feet"]), meta,
               ["C3", "Cz", "C4", "Pz"]),
              (rng.standard_normal((2, 3, 5)), np.array(["a", "b"], dtype=object),
               meta.iloc[:2].reset_index(drop=True), ["C3", "Cz", "C4"])]
    path = tmp_path / "c" / "subject_003.npz"
    save_subject(path, chunks, {"dataset": "x"})
    got = load_subject(path)
    assert len(got) == 2 and (tmp_path / "c" / "params.json").exists()
    for (X, y, m, ch), (X2, y2, m2, ch2) in zip(chunks, got):
        assert X2.dtype == X.dtype and np.array_equal(X, X2)
        assert y2.dtype == y.dtype and np.array_equal(y, y2)
        pd.testing.assert_frame_equal(m, m2)
        assert ch2 == ch
    assert load_subject(tmp_path / "missing.npz") is None
    (tmp_path / "c" / "bad.npz").write_bytes(b"not a zip")
    assert load_subject(tmp_path / "c" / "bad.npz") is None


def test_loader_reads_the_cache(fake_moabb, tmp_path, monkeypatch):
    from rfl.data.data_utils import load_moabb_dataset
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    monkeypatch.setenv("RFL_DATA_CACHE", "on")
    calls = []
    real = _FakeParadigm.get_data
    monkeypatch.setattr(_FakeParadigm, "get_data", lambda self, d, subjects, **k: (
        calls.append(subjects[0]) or real(self, d, subjects, **k)))
    fake_moabb.chans = {1: {"0": ["C3", "Cz"]}, 2: {"0": ["Cz", "C3"]}}
    first = load_moabb_dataset("Cho2017", [1, 2], resample=128.0, verbose=False)
    assert calls == [1, 2] and len(list(tmp_path.glob("rfl-cache/Cho2017/*/subject_*.npz"))) == 2
    again = load_moabb_dataset("Cho2017", [1, 2], resample=128.0, verbose=False)
    assert calls == [1, 2]                                        # read from the cache
    for a, b in zip(first[:2], again[:2]):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    pd.testing.assert_frame_equal(first[2], again[2])
    load_moabb_dataset("Cho2017", [1], resample=128.0, fmax=30.0, verbose=False)
    assert calls == [1, 2, 1]                                     # other settings, other key
    monkeypatch.setenv("RFL_DATA_CACHE", "refresh")
    load_moabb_dataset("Cho2017", [2], resample=128.0, verbose=False)
    assert calls == [1, 2, 1, 2]
    monkeypatch.setenv("RFL_DATA_CACHE", "off")
    load_moabb_dataset("Cho2017", [1], resample=64.0, verbose=False)
    assert len(list(tmp_path.glob("rfl-cache/Cho2017/*"))) == 2  # nothing written
