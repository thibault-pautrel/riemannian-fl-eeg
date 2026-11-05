#!/usr/bin/env python3
# file: rfl/experiments/fetch_dataset.py
"""
Download a MOABB dataset subject by subject, with retries, before running experiments.

Downloading once up front avoids many parallel runs fetching the same files at the
same time, which can overload the data servers (HTTP 502) and corrupt files. MNE checks
the files already present against their published hash and downloads them again when
they are incomplete. Rerun the script until it reports no missing subject.

Files go to one folder: --data-dir, else MNE_DATA, else ~/mne_data (see
rfl/data/registry.py:configure_mne_paths, the same rule as the runners). The MNE
configuration file is read but never written.

Usage (from any folder once the repository is installed, see README):
    python -m rfl.experiments.fetch_dataset --dataset BNCI2014_001
    python -m rfl.experiments.fetch_dataset --dataset PhysionetMI --data-dir /path/to/mne_data

Requirements: MOABB and network access.
"""
import argparse
import sys
import time


from rfl.data.registry import DATASET_CONFIGS

def main() -> int:
    """Download every subject of one dataset. Returns 1 if a subject is still missing."""
    from rfl.utils.logs import setup_logging
    setup_logging()
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS),
                   help="A dataset of rfl/data/registry.py.")
    p.add_argument("--data-dir", default=None,
                   help="Set the MNE data variables to this directory for this run.")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--wait", type=float, default=30.0, help="seconds between retries")
    p.add_argument("--no-stats", action="store_true",
                   help="skip the trial counts used by the checks (they load every "
                        "subject once)")
    a = p.parse_args()

    # Before MOABB reads them: MNE_DATA and the per-dataset keys (rfl/data/registry.py).
    from rfl.data.registry import configure_mne_paths, get_moabb_class, get_subjects
    configure_mne_paths(a.data_dir)

    ds = get_moabb_class(a.dataset)()
    missing = []
    for s in get_subjects(a.dataset):
        for attempt in range(1, a.retries + 1):
            try:
                ds.data_path(s)
                print(f"subject {s:3d}: ok", flush=True)
                break
            except Exception as e:                      # noqa: BLE001
                print(f"subject {s:3d}: attempt {attempt} failed ({type(e).__name__}: "
                      f"{str(e)[:100]})", flush=True)
                time.sleep(a.wait)
        else:
            missing.append(s)
    print(f"\n{a.dataset}: {len(missing)} subject(s) missing" +
          (f": {missing}" if missing else ""))
    if missing:
        return 1
    if not a.no_stats:
        # Trial counts and channels, for the checks of the runners (rfl/data/stats.py).
        # One subject at a time: the file merges the subjects, and the memory stays that
        # of one subject (Schirrmeister2017 at once would need well over 8 GB).
        from rfl.data.data_utils import load_moabb_dataset
        from rfl.data.stats import stats_path
        for s in get_subjects(a.dataset):
            load_moabb_dataset(a.dataset, [s], resample=128.0, verbose=False)
            print(f"subject {s:3d}: trials counted", flush=True)
        print(f"statistics written: {stats_path(a.dataset)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
