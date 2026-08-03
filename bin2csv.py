# Convert recorder .bin (raw float32 XYZ, no header) → .csv with x,y,z header.
# Usage in pages/notes.md.
import glob
import os
import sys

import numpy as np


def convert(path):
    xyz = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
    out = os.path.splitext(path)[0] + ".csv"
    with open(out, "w", newline="") as f:
        f.write("x,y,z\n")
        np.savetxt(f, xyz, delimiter=",", fmt="%.6f")
    print(f"{path} -> {out}  ({len(xyz)} samples)")


def _expand(args):
    for a in args:
        if os.path.isdir(a):
            yield from sorted(glob.glob(os.path.join(a, "*.bin")))
        else:
            yield a


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python bin2csv.py <file.bin | dir> ...")
    for p in _expand(sys.argv[1:]):
        convert(p)
