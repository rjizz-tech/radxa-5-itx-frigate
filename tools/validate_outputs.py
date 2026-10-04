"""Compare decoded Frigate rows against normalized AXERA reference results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'overlay'))
from radxa_frigate.yolo26 import decode


def compare(actual, reference, score_tolerance=0.01, coordinate_tolerance=2/640):
    actual = np.asarray(actual)[np.asarray(actual)[:, 1] > 0]
    reference = np.asarray(reference, dtype=np.float32).reshape(-1, 6)
    reference = reference[reference[:, 1] > 0]
    if len(actual) != len(reference):
        raise ValueError(f'Detection count differs: {len(actual)} vs {len(reference)}')
    remaining = list(range(len(reference)))
    for row in actual:
        candidates = [i for i in remaining if reference[i, 0] == row[0]]
        if not candidates:
            raise ValueError('Reference class mismatch')
        best = min(candidates, key=lambda i: np.max(np.abs(reference[i, 2:]-row[2:])))
        if abs(reference[best, 1]-row[1]) > score_tolerance or np.max(np.abs(reference[best, 2:]-row[2:])) > coordinate_tolerance:
            raise ValueError('Reference scores/boxes exceed tolerance')
        remaining.remove(best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=Path, required=True, help='NPZ with output_0 through output_5')
    parser.add_argument('--reference', type=Path, required=True, help='JSON rows [class,score,y0,x0,y1,x1], normalized')
    args = parser.parse_args()
    with np.load(args.heads, allow_pickle=False) as values:
        rows = decode([values[f'output_{i}'] for i in range(6)])
    compare(rows, json.loads(args.reference.read_text()))
    print('AXERA reference comparison passed')
