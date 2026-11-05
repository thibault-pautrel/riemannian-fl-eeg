"""Remove printed tensors from the output of a run, before it reaches run.log.

The backward of the whitening of yetanotherspdnet (`Whitening` in
functions/spd_linalg.py, commit 5c08ebc) prints its two gradients at every training
step of a model with the SPD batch norm: tens to hundreds of MB per run. The library is
used as it is, so the jobs filter its output instead (condor/run_one.sh and
`rfl.experiments.grid --run`). A tensor printed by Python starts a line with `tensor(`
and ends where its parentheses close; every such block is dropped, every other line is
kept, and one summary line at the end counts the dropped blocks.

    python -m rfl.experiments.run_federated_experiment ... 2>&1 | python -m rfl.utils.log_filter

Set RFL_RAW_LOG=1 to keep the raw output (condor/run_one.sh).
"""
from __future__ import annotations

import sys
from typing import Iterable, Iterator, TextIO


class TensorPrintFilter:
    """Line filter: `keep(line)` is False for the lines of a printed tensor."""

    def __init__(self) -> None:
        self.depth = 0          # open parentheses of the tensor being dropped
        self.dropped = 0        # tensors dropped

    def keep(self, line: str) -> bool:
        if self.depth == 0:
            if not line.startswith("tensor("):
                return True
            self.dropped += 1
        self.depth = max(0, self.depth + line.count("(") - line.count(")"))
        return False

    def summary(self) -> str:
        return (f"[log_filter] removed {self.dropped} printed tensors (the debug print "
                "of yetanotherspdnet's Whitening.backward, see rfl/utils/log_filter.py)\n")

    def filter(self, lines: Iterable[str]) -> Iterator[str]:
        for line in lines:
            if self.keep(line):
                yield line
        if self.dropped:
            yield self.summary()


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    for line in TensorPrintFilter().filter(stdin):
        stdout.write(line)
        stdout.flush()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(__doc__)
        sys.exit(0 if sys.argv[1] in ("-h", "--help") else 2)
    sys.exit(main())
