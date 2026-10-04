"""Run exact preprocessed input on AX-M1 and save raw tensors for comparison."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'overlay'))
from radxa_frigate.runtime import ModelSession


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('/models/manifest.json'))
    parser.add_argument('--model', choices=['yolov9', 'yolo26n', 'yolo26s', 'yunet', 'arcface'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    session = ModelSession(args.manifest, args.model, kind='object', timeout=10)
    outputs = session.run(np.load(args.input, allow_pickle=False))
    np.savez(args.output, **{f'output_{i}': item for i, item in enumerate(outputs)})
    print('Saved raw output tensors; no CPU fallback was used')
