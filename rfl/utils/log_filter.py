"""Remove printed tensors from the output of a run, before it reaches run.log.

The backward of the whitening of yetanotherspdnet (`Whitening` in
functions/spd_linalg.py, commit 5c08ebc) prints its two gradients at every training
step of a model with the SPD batch norm: tens to hundreds of MB per run. The library is
used as it is, so the jobs filter its output instead (condor/run_one.sh and
`rfl.experiments.grid --run`). A tensor printed by Python starts a line with `tensor(`
and ends where its parentheses close; every such block is dropped, also when it starts
after other text on its line (a tqdm progress bar), everything else is kept, and one
summary line at the end counts the dropped blocks.

    python -m rfl.experiments.run_federated_experiment ... 2>&1 | python -m rfl.utils.log_filter

Set RFL_RAW_LOG=1 to keep the raw output (condor/run_one.sh).
"""
from __future__ import annotations

import sys
from typing import Iterable, Iterator, Optional, TextIO


class TensorPrintFilter:
    """Line filter: drops every printed tensor, from `tensor(` to its closing parenthesis.

    A tensor may start in the middle of a line: in a centralised run the progress bar of
    tqdm ends with a carriage return and no newline, so the print follows it on the same
    line. The text around a tensor is kept, and a line left blank by it is dropped.
    """

    def __init__(self) -> None:
        self.depth = 0          # open parentheses of the tensor being dropped
        self.dropped = 0        # tensors dropped

    def cut(self, line: str) -> Optional[str]:
        """`line` without its tensor parts, or None when nothing else is left."""
        if self.depth == 0 and "tensor(" not in line:
            return line
        kept, i, n = [], 0, len(line)
        while i < n:
            if self.depth == 0:
                j = line.find("tensor(", i)
                if j < 0:
                    kept.append(line[i:])
                    break
                kept.append(line[i:j])
                self.dropped += 1
                self.depth, i = 1, j + len("tensor(")
                continue
            while i < n and self.depth > 0:
                c = line[i]
                self.depth += (c == "(") - (c == ")")
                i += 1
        rest = "".join(kept)
        if not rest.strip():
            return None
        return rest if rest.endswith("\n") or not line.endswith("\n") else rest + "\n"

    def keep(self, line: str) -> bool:
        """Whether something of `line` is kept (`cut`)."""
        return self.cut(line) is not None

    def summary(self) -> str:
        return (f"[log_filter] removed {self.dropped} printed tensors (the debug print "
                "of yetanotherspdnet's Whitening.backward, see rfl/utils/log_filter.py)\n")

    def filter(self, lines: Iterable[str]) -> Iterator[str]:
        for line in lines:
            rest = self.cut(line)
            if rest is not None:
                yield rest
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
