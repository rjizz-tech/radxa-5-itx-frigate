"""Prepare Pulsar2 INT8 calibration/configs; register compiled models explicitly."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
YUNET_OUTPUTS = [f'{group}_{stride}' for group in ('cls', 'obj', 'bbox', 'kps') for stride in (8, 16, 32)]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(args):
    import numpy as np
    specs = json.loads((ROOT/'models/manifest.json').read_text())['face_sources']
    for name in ('facedet.onnx', 'arcface.onnx'):
        path = args.sources/name
        if digest(path) != specs[name]['sha256']:
            raise ValueError(f'Unpinned source: {name}')
    args.output.mkdir(parents=True, exist_ok=True)
    # Inputs are deliberately prepared outside the compiler: no hidden color or
    # normalization transforms. ArcFace samples must be native-aligned faces.
    for model, source, tensor, size in (('yunet', 'facedet.onnx', 'input', 640), ('arcface', 'arcface.onnx', 'data', 112)):
        samples = sorted((args.calibration/model).glob('*.npy'))
        if len(samples) < 32:
            raise ValueError(f'Provide at least 32 representative {model} calibration .npy tensors')
        archive = args.output/f'{model}-calibration.tar'
        with tarfile.open(archive, 'w') as handle:
            for number, path in enumerate(samples):
                value = np.load(path, allow_pickle=False)
                if value.shape != (1, 3, size, size) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f'Invalid calibration tensor: {path}')
                buffer = io.BytesIO()
                np.save(buffer, value)
                info = tarfile.TarInfo(f'{number:06}.npy')
                info.size = len(buffer.getvalue())
                handle.addfile(info, io.BytesIO(buffer.getvalue()))
        config = {
            'input': str((args.sources/source).resolve()),
            'output_dir': str((args.output/model).resolve()), 'output_name': f'{model}.axmodel',
            'target_hardware': 'AX650', 'npu_mode': 'NPU3',
            'input_shapes': f'{tensor}:1x3x{size}x{size}',
            'quant': {'input_configs': [{'tensor_name': tensor, 'calibration_dataset': str(archive.resolve()),
                                        'calibration_format': 'Numpy', 'calibration_size': len(samples)}]},
            'input_processors': [{'tensor_name': tensor, 'tensor_layout': 'NCHW', 'src_layout': 'NCHW',
                                  'src_dtype': 'FP32', 'mean': [], 'std': []}],
            'output_processors': [{'tensor_name': 'DEFAULT', 'output_dtype': 'FP32'}],
        }
        (args.output/f'{model}.json').write_text(json.dumps(config, indent=2)+'\n')
    print('Configs ready. Compile in Pulsar2 4.2 with: pulsar2 build --config <model>.json')


def register(args):
    target = args.artifacts/f'{args.model}.axmodel'
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.file.resolve() != target.resolve():
        shutil.copy2(args.file, target)
    path = args.artifacts/'manifest.json'
    manifest = json.loads(path.read_text())
    size = 640 if args.model == 'yunet' else 112
    manifest['models'][args.model] = {
        'file': target.name, 'sha256': digest(target),
        'input': {'shape': [1, 3, size, size], 'dtype': 'float32'},
        'output_names': YUNET_OUTPUTS if args.model == 'yunet' else ['1333'],
        'format': 'yunet-raw-heads' if args.model == 'yunet' else 'native-arcface',
        'compiler': args.compiler, 'calibration_sha256': digest(args.calibration_archive),
        'validation': 'pending-board-acceptance',
    }
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(path)
    print('Model registered; verify compiled metadata and accuracy on AX-M1 before acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--sources', type=Path, default=ROOT/'.cache/face-sources')
    prep.add_argument('--calibration', type=Path, required=True)
    prep.add_argument('--output', type=Path, default=ROOT/'.cache/face-build')
    reg = sub.add_parser('register')
    reg.add_argument('model', choices=['yunet', 'arcface'])
    reg.add_argument('--file', type=Path, required=True)
    reg.add_argument('--compiler', required=True, help='Exact compiler version and container digest')
    reg.add_argument('--calibration-archive', type=Path, required=True)
    reg.add_argument('--artifacts', type=Path, default=ROOT/'models/artifacts')
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else register(args)


if __name__ == '__main__':
    main()
