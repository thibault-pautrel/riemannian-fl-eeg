"""Loading and preparation of MOABB motor-imagery datasets.

Provides `load_moabb_dataset` (band-pass filtering, optional resampling, epoching with
the MOABB MotorImagery paradigm), `filter_binary_task` (class subsets) and
`create_splits` (train/validation/test splits). Dataset metadata comes from
rfl/data/registry.py. moabb and mne are imported on first use, so the synthetic SPD
workflow does not need them.

Data location: the MNE data folder is read from the `MNE_DATA` environment variable,
then from the MNE configuration file, and defaults to `<repository>/mne_data`. Dataset
specific variables (`MNE_DATASETS_<NAME>_PATH`) take precedence, as usual in MNE.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from rfl.data.registry import DATASET_CONFIGS

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=r"warnEpochs .*", module=r"moabb\.datasets\.preprocessing")
warnings.filterwarnings("ignore", message=r"Could not acquire lock file.*", category=RuntimeWarning)

_MOABB_READY = False


def _ensure_moabb():
    """Import moabb and mne, and set the MNE data directory, once."""
    global _MOABB_READY
    if _MOABB_READY:
        return
    import moabb

    from rfl.data.registry import configure_mne_paths
    configure_mne_paths()                    # this process only, see rfl/data/registry.py
    moabb.set_log_level("warning")
    _MOABB_READY = True


def _get_moabb_class(name: str):
    """Return the MOABB dataset class for `name` (lazy import)."""
    import moabb.datasets as mds
    return getattr(mds, name)


def _apply_session_filter(
    X: np.ndarray,
    y: np.ndarray,
    meta,
    dataset_name: str,
    verbose: bool = True
):
    """Keep the sessions listed in DATASET_CONFIGS[dataset_name]['sessions'], if any.

    Sessions are compared as strings, since MOABB stores them as ints or strings
    depending on the dataset.
    """
    config = DATASET_CONFIGS[dataset_name]
    sessions = config.get('sessions', None)
    if sessions is None:
        return X, y, meta

    sessions_str = {str(s) for s in sessions}
    session_mask = meta['session'].astype(str).isin(sessions_str)
    n_before = len(y)
    X    = X[session_mask.values]
    y    = y[session_mask.values]
    meta = meta[session_mask].reset_index(drop=True)
    n_after = len(y)

    if verbose:
        logger.info(f"[{dataset_name}] Session filter: kept sessions {sessions} "
              f"→ {n_after}/{n_before} trials")
    return X, y, meta


def _is_timeout(err: Exception) -> bool:
    msg = str(err).lower()
    return "timeout" in msg or "timed out" in msg


def _load_subject(paradigm, dataset, subj: int, max_retries: int, retry_delay: int,
                  verbose: bool) -> list:
    """Epochs of one subject: [(X, y, meta, ch_names)], one entry per session if needed.

    Download timeouts are retried. If the sessions of the subject have different
    channel sets (MNE cannot concatenate them), the sessions are loaded one by one and
    aligned later by name. Any other error stops the load: a subject is never dropped
    silently.
    """
    def get(sessions=None):
        orig = getattr(dataset, "sessions", None)
        if sessions is not None:
            dataset.sessions = sessions
        try:
            for attempt in range(max_retries):
                try:
                    epochs, y, meta = paradigm.get_data(dataset, subjects=[subj],
                                                        return_epochs=True)
                    X = epochs.get_data() * dataset.unit_factor
                    return X, np.asarray(y), meta.reset_index(drop=True), list(epochs.ch_names)
                except Exception as err:                             # noqa: BLE001
                    if not _is_timeout(err) or attempt == max_retries - 1:
                        raise
                    if verbose:
                        logger.info(f"  subject {subj}: timeout, retry {attempt + 2}/"
                                    f"{max_retries} in {retry_delay} s")
                    time.sleep(retry_delay)
        finally:
            if sessions is not None:
                dataset.sessions = orig

    try:
        return [get()]
    except Exception as err:                                         # noqa: BLE001
        msg = str(err).lower()
        if not any(k in msg for k in ("channel", "dimension", "concatenat")):
            raise
        sessions = getattr(dataset, "sessions", None)
        if not sessions:
            raise RuntimeError(f"subject {subj}: the sessions have different channels "
                               f"and the dataset cannot be loaded session by session "
                               f"({err})") from err
        if verbose:
            logger.info(f"  subject {subj}: sessions with different channels, loaded "
                        "one by one")
        return [get([sess]) for sess in sessions]


def _align_channels(chunks: list, dataset_name: str, verbose: bool):
    """Concatenate (X, y, meta, ch_names) chunks on the channels common to all, by name.

    The channel order is that of the first chunk. When every chunk has the same
    channels in the same order, the result equals a plain concatenation.
    """
    first = chunks[0][3]
    common = [c for c in first if all(c in ch for *_, ch in chunks)]
    if not common:
        raise RuntimeError(f"{dataset_name}: no EEG channel common to all subjects")
    dropped = sorted(set().union(*(set(ch) for *_, ch in chunks)) - set(common))
    if dropped:
        logger.warning(f"{dataset_name}: {len(dropped)} channel(s) missing in some "
                       f"subject(s) or session(s) are dropped: {dropped}")
    reordered = any(ch != common for *_, ch in chunks)
    if reordered and verbose and not dropped:
        logger.info(f"{dataset_name}: channel order differs across subjects; aligned "
                    "by name")
    X = np.concatenate([Xi[:, [ch.index(c) for c in common], :]
                        for Xi, _, _, ch in chunks], axis=0)
    y = np.concatenate([yi for _, yi, _, _ in chunks], axis=0)
    meta = pd.concat([mi for _, _, mi, _ in chunks], ignore_index=True)
    return X, y, meta


def resolve_resample(requested: Optional[float], native: Optional[float]) -> Optional[float]:
    """Effective sampling rate: `requested`, but never above `native` (no upsampling).

    Returns `native` when `requested` is None or <= 0, and `requested` unchanged when
    `native` is None (for example synthetic data).
    """
    if native is None:
        return requested
    if requested is None or requested <= 0:
        return native
    return float(requested) if requested < native else native


def load_moabb_dataset(
    dataset_name: str,
    subjects: Optional[List[int]] = None,
    resample: Optional[float] = None,
    fmin: float = 8.0,
    fmax: float = 32.0,
    max_retries: int = 3,
    retry_delay: int = 5,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load epoched trials of a MOABB motor-imagery dataset (downloads if needed).

    Uses MotorImagery(n_classes, events, resample, fmin, fmax, tmin, tmax) with the
    registry settings. Timeouts are retried. If subjects (or sessions) have different
    channel sets, they are loaded one by one and restricted to the common channels.
    The configured sessions are then kept, and string labels are encoded as integers
    in alphabetical order of the event names (LabelEncoder).

    Args:
        dataset_name: Registry name, for example 'BNCI2014_001'.
        subjects: Subject ids. None loads the first two subjects.
        resample: Target sampling rate in Hz. None keeps the native rate.
        fmin: Band-pass lower cutoff (Hz).
        fmax: Band-pass upper cutoff (Hz).
        max_retries: Attempts on download timeouts.
        retry_delay: Seconds between attempts.
        verbose: Print progress.

    Returns:
        (X, y, meta): X of shape (N, C, T), integer labels y of shape (N,), and the
        MOABB metadata DataFrame (one row per trial, with 'subject' and 'session').

    Raises:
        ValueError: On an unknown dataset or invalid subject ids.
        RuntimeError: If loading fails after all attempts.
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset: '{dataset_name}'. "
                        f"Available: {available}")

    config = DATASET_CONFIGS[dataset_name]

    if subjects is None:
        subjects = config['subjects'][:2]
        if verbose:
            logger.info(f"No subjects specified, using first 2: {subjects}")

    invalid = [s for s in subjects if s not in config['subjects']]
    if invalid:
        raise ValueError(f"Invalid subject IDs: {invalid}. "
                        f"Valid range: {min(config['subjects'])}-{max(config['subjects'])}")

    if verbose:
        logger.info(f"\nLoading {dataset_name}...")
        logger.info(f"Subjects: {subjects}")
        logger.info(f"Classes: {config['n_classes']} {config['events']}")
        logger.info(f"Filters: {fmin}-{fmax} Hz")
        if resample:
            logger.info(f"Resampling: {resample} Hz")

    # Pass `sessions` to the dataset constructor when the registry sets them and the
    # constructor accepts them, so that other sessions are not downloaded.
    import inspect as _inspect
    _ensure_moabb()
    from moabb.paradigms import MotorImagery
    cls = _get_moabb_class(dataset_name)
    init_sig = _inspect.signature(cls.__init__).parameters
    dataset_kwargs = {}
    if 'sessions' in init_sig and config.get('sessions') is not None:
        dataset_kwargs['sessions'] = config['sessions']
    dataset = cls(**dataset_kwargs)
    paradigm = MotorImagery(
        n_classes=config['n_classes'],
        events=config['events'],
        resample=resample,
        fmin=fmin,
        fmax=fmax,
        tmin=config.get('tmin', 0.0),
        tmax=config.get('tmax', None)
    )

    # One subject at a time, as MNE Epochs: their channel names let the subjects be
    # aligned by name. The arrays are those of paradigm.get_data (Epochs.get_data times
    # the dataset unit factor, as MOABB computes them).
    chunks = [_load_subject(paradigm, dataset, subj, max_retries, retry_delay, verbose)
              for subj in subjects]
    chunks = [c for group in chunks for c in group]
    X, y, meta = _align_channels(chunks, dataset_name, verbose)

    X, y, meta = _apply_session_filter(X, y, meta, dataset_name, verbose)

    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        # Keep the event names: the tasks select classes by name (filter_binary_task).
        if hasattr(meta, "columns") and "event" not in meta.columns:
            meta = meta.assign(event=np.asarray(y).astype(str))
        le = LabelEncoder().fit(y)
        if verbose:
            logger.info(f"Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
        y = le.transform(y)

    from rfl.data.stats import update_stats
    update_stats(dataset_name, meta, X.shape[1])     # for the checks of later runs

    if verbose:
        logger.info("Data loaded successfully!")
        logger.info(f"Shape: {X.shape} (trials, channels, timepoints)")
        logger.info(f"Labels: {len(y)} trials, {len(np.unique(y))} classes")

    return X, y, meta


def filter_binary_task(
    X: np.ndarray,
    y: np.ndarray,
    meta: Dict,
    task: str = 'all_classes',
    dataset_name: str = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Select the trials of a classification task.

    Tasks:
        'all_classes'  everything, unchanged.
        'binary'       left hand (label 0) versus right hand (label 1). The classes
                       are found by name, from meta['event'] (written by
                       load_moabb_dataset) or else from the registry events in
                       alphabetical order, the order of the label encoder. Without
                       either (synthetic data), labels 0 and 1 are kept.
        'no_rest'      drop the 'rest' class (found from meta['event'] if present,
                       otherwise from the registry event list in alphabetical order),
                       then remap the labels to 0..K-1.

    Args:
        X: (N, C, T) trials.
        y: (N,) integer labels.
        meta: Metadata DataFrame aligned with X.
        task: 'binary', 'all_classes' or 'no_rest'.
        dataset_name: Registry name, used by 'no_rest'.
        verbose: Print the class distribution.

    Returns:
        (X, y, meta) restricted to the task.

    Raises:
        ValueError: On an unknown task, or 'no_rest' for a dataset without 'rest'.
    """
    if task == 'binary':
        names = None
        if hasattr(meta, "columns") and 'event' in meta.columns:
            names = meta['event'].astype(str).str.lower().to_numpy()
        elif dataset_name and dataset_name in DATASET_CONFIGS:
            events = sorted(DATASET_CONFIGS[dataset_name]['events'])
            names = np.array([events[int(v)] for v in y])
        if names is not None:
            if not {'left_hand', 'right_hand'} <= set(names):
                raise ValueError(f"task='binary' needs left_hand and right_hand trials; "
                                 f"found {sorted(set(names))}")
            binary_mask = (names == 'left_hand') | (names == 'right_hand')
            new_y = (names[binary_mask] == 'right_hand').astype(np.int64)
        else:
            if verbose:
                logger.warning("no event names: keeping labels 0 and 1 as the two classes")
            binary_mask = (y == 0) | (y == 1)
            new_y = np.asarray(y)[binary_mask]
        X = X[binary_mask]
        meta = meta.iloc[binary_mask].reset_index(drop=True)
        y = new_y
        if verbose:
            logger.info("\nBinary task: left hand (0) versus right hand (1)")
            logger.info(f"  Kept {binary_mask.sum()}/{len(binary_mask)} trials")
            logger.info(f"  Class distribution: {np.bincount(y)}")

    elif task == 'no_rest':
        if 'event' in meta.columns:
            rest_mask = meta['event'].str.lower() == 'rest'
            if not rest_mask.any():
                raise ValueError("task='no_rest': no 'rest' trial in this dataset, so the "
                                 "task would change nothing. Use 'all_classes'.")
            keep_mask = ~rest_mask
        else:
            # LabelEncoder assigns labels in alphabetical order of the event names.
            if dataset_name and dataset_name in DATASET_CONFIGS:
                events = sorted(DATASET_CONFIGS[dataset_name]['events'])
                if 'rest' in events:
                    rest_label = events.index('rest')
                    if verbose:
                        logger.info(f"  Found 'rest' at label {rest_label} from dataset config (sorted events: {events})")
                    keep_mask = y != rest_label
                else:
                    raise ValueError(f"Dataset {dataset_name} has no 'rest' class in config: {events}")
            else:
                # Unreliable fallback: treat the most frequent class as 'rest'.
                unique, counts = np.unique(y, return_counts=True)
                rest_label = unique[np.argmax(counts)]
                if verbose:
                    logger.warning("no 'event' column and no dataset config")
                    logger.info(f"  Assuming rest class is label {rest_label} (most frequent) - THIS MAY BE WRONG!")
                keep_mask = y != rest_label

        n_original = len(y)
        X = X[keep_mask]
        y_filtered = y[keep_mask]
        meta = meta.iloc[keep_mask].reset_index(drop=True)

        unique_labels = np.unique(y_filtered)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y_filtered])

        if verbose:
            logger.info("\nExcluded 'rest' class")
            logger.info(f"  Kept {keep_mask.sum()}/{n_original} trials")
            logger.info(f"  Label remapping: {label_map}")
            logger.info(f"  Class distribution: {np.bincount(y)}")

    elif task == 'all_classes':
        if verbose:
            logger.info("\nUsing all classes")
            logger.info(f"  Class distribution: {np.bincount(y)}")

    else:
        raise ValueError(f"Unknown task: {task}. Use 'binary', 'all_classes', or 'no_rest'")

    return X, y, meta


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    meta: Dict,
    split_type: str = 'session',
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split trials into train, validation and test sets.

    Split types:
        'session'        test on the evaluation session(s), validation drawn from the
                         training sessions (stratified). Falls back to 'random' without
                         session information or with a single session.
        'subject'        stratified train/val/test split within each subject
                         (subjects with fewer than 10 trials are skipped).
        'cross-subject'  the last subjects (fraction test_size) form the test set.
        'random'         stratified split of all trials.

    In every case the validation fraction is val_size of the total, obtained as
    val_size / (1 - test_size) of the non-test part.

    Args:
        X: (N, C, T) trials.
        y: (N,) labels.
        meta: Metadata DataFrame aligned with X.
        split_type: One of the types above.
        val_size: Validation fraction.
        test_size: Test fraction.
        random_state: Seed of the splits.
        verbose: Print split sizes.

    Returns:
        Dict with keys 'train', 'val', 'test', each an (X, y) pair. 'val' may be empty.

    Raises:
        ValueError: With fewer than two classes, missing metadata, or an unknown
            split type.
    """
    n_trials = len(y)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes for stratified split")

    train_mask = None
    val_mask = None
    test_mask = None

    if split_type == 'session':
        if 'session' not in meta.columns:
            if verbose:
                logger.info("No 'session' column, falling back to 'random' split")
            split_type = 'random'
        else:
            session = np.asarray(meta['session'])
            unique_sessions = sorted(set(session))

            if verbose:
                logger.info("\nSession-based split:")
                logger.info(f"Unique sessions: {unique_sessions}")

            train_sessions = []
            test_sessions = []

            for s in unique_sessions:
                s_str = str(s).lower()
                if 'train' in s_str or 'session_t' in s_str or s_str in ['0', 'session_0']:
                    train_sessions.append(s)
                elif 'test' in s_str or 'session_e' in s_str or 'eval' in s_str or s_str in ['1', 'session_1']:
                    test_sessions.append(s)

            # Otherwise: all sessions but the last for training, the last for test.
            if not train_sessions or not test_sessions:
                if len(unique_sessions) >= 2:
                    train_sessions = unique_sessions[:-1]
                    test_sessions = [unique_sessions[-1]]
                    if verbose:
                        logger.info("Using fallback: first sessions for train, last for test")
                else:
                    if verbose:
                        logger.info("Only one session, falling back to 'random' split")
                    split_type = 'random'

            if split_type == 'session':
                if verbose:
                    logger.info(f"Train sessions: {train_sessions}")
                    logger.info(f"Test sessions: {test_sessions}")

                train_mask = np.isin(session, train_sessions)
                test_mask = np.isin(session, test_sessions)

                train_indices = np.where(train_mask)[0]
                if len(train_indices) > 20:
                    train_idx, val_idx = train_test_split(
                        train_indices,
                        test_size=val_size/(1-test_size),
                        random_state=random_state,
                        stratify=y[train_indices]
                    )
                    train_mask = np.zeros(n_trials, dtype=bool)
                    train_mask[train_idx] = True
                    val_mask = np.zeros(n_trials, dtype=bool)
                    val_mask[val_idx] = True
                else:
                    val_mask = np.zeros(n_trials, dtype=bool)

    if split_type == 'subject' and train_mask is None:
        if 'subject' not in meta.columns:
            raise ValueError("No 'subject' column. Use split_type='random'")

        subjects = np.asarray(meta['subject'])
        train_mask = np.zeros(n_trials, dtype=bool)
        val_mask = np.zeros(n_trials, dtype=bool)
        test_mask = np.zeros(n_trials, dtype=bool)

        for subj in np.unique(subjects):
            subj_mask = subjects == subj
            subj_indices = np.where(subj_mask)[0]

            if len(subj_indices) < 10:
                if verbose:
                    logger.info(f"Subject {subj} has only {len(subj_indices)} trials, skipping")
                continue

            train_idx, test_idx = train_test_split(
                subj_indices, test_size=test_size,
                random_state=random_state, stratify=y[subj_indices]
            )
            train_idx, val_idx = train_test_split(
                train_idx, test_size=val_size/(1-test_size),
                random_state=random_state, stratify=y[train_idx]
            )

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

        if verbose:
            logger.info(f"\nWithin-subject split for {len(np.unique(subjects))} subjects")

    if split_type == 'cross-subject' and train_mask is None:
        if 'subject' not in meta.columns:
            raise ValueError("No 'subject' column. Use split_type='random'")

        subjects = np.asarray(meta['subject'])
        unique_subjects = sorted(np.unique(subjects))

        n_test_subjects = max(1, int(len(unique_subjects) * test_size))
        test_subjects = unique_subjects[-n_test_subjects:]
        train_val_subjects = unique_subjects[:-n_test_subjects]

        test_mask = np.isin(subjects, test_subjects)
        train_val_mask = np.isin(subjects, train_val_subjects)

        train_val_indices = np.where(train_val_mask)[0]
        train_idx, val_idx = train_test_split(
            train_val_indices, test_size=val_size/(1-test_size),
            random_state=random_state, stratify=y[train_val_indices]
        )

        train_mask = np.zeros(n_trials, dtype=bool)
        train_mask[train_idx] = True
        val_mask = np.zeros(n_trials, dtype=bool)
        val_mask[val_idx] = True

        if verbose:
            logger.info("\nCross-subject split:")
            logger.info(f"Train subjects: {train_val_subjects}")
            logger.info(f"Test subjects: {test_subjects}")

    if split_type == 'random' and train_mask is None:
        train_idx, test_idx = train_test_split(
            np.arange(n_trials), test_size=test_size,
            random_state=random_state, stratify=y
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size),
            random_state=random_state, stratify=y[train_idx]
        )

        train_mask = np.zeros(n_trials, dtype=bool)
        train_mask[train_idx] = True
        val_mask = np.zeros(n_trials, dtype=bool)
        val_mask[val_idx] = True
        test_mask = np.zeros(n_trials, dtype=bool)
        test_mask[test_idx] = True

        if verbose:
            logger.info("\nRandom stratified split")

    if train_mask is None:
        raise ValueError(f"Failed to create splits with split_type='{split_type}'")

    if verbose:
        logger.info(f"Split sizes - Train: {train_mask.sum()}, "
              f"Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        logger.info(f"Proportions - Train: {train_mask.sum()/n_trials:.1%}, "
              f"Val: {val_mask.sum()/n_trials:.1%}, "
              f"Test: {test_mask.sum()/n_trials:.1%}")

    return {
        'train': (X[train_mask], y[train_mask]),
        'val': (X[val_mask], y[val_mask]) if val_mask.sum() > 0 else (X[train_mask][:0], y[train_mask][:0]),
        'test': (X[test_mask], y[test_mask])
    }
