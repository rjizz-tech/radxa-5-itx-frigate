"""Download immutable model versions, verifying SHA256 before installing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fetch(spec, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec['file']
    if Path(spec['file']).name != spec['file']:
        raise ValueError('Manifest filenames must be basenames')
    if target.exists() and checksum(target) == spec['sha256']:
        return target
    temporary = target.with_suffix(target.suffix + '.download')
    try:
        with urllib.request.urlopen(spec['url'], timeout=120) as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output)
        if checksum(temporary) != spec['sha256']:
            raise ValueError(f'Checksum mismatch: {target.name}')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='+', choices=['yolov9', 'yolo26n', 'yolo26s'], default=['yolov9'])
    parser.add_argument('--face-sources', action='store_true')
    parser.add_argument('--config-dir', type=Path, help='Install persistent CPU landmark model here')
    args = parser.parse_args()
    lock = json.loads((ROOT/'models/manifest.json').read_text())
    artifacts = ROOT/'models/artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema_version': 1, 'models': {}}
    for profile in args.profiles:
        spec = lock['models'][profile]
        fetch(spec, artifacts)
        manifest['models'][profile] = spec
        print(f'Verified {profile}')
    if args.face_sources:
        for spec in lock['face_sources'].values():
            fetch(spec, ROOT/'.cache/face-sources')
        if args.config_dir:
            target = args.config_dir/'model_cache/facedet'
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/'.cache/face-sources/landmarkdet.yaml', target/'landmarkdet.yaml')
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(manifest_path)


if __name__ == '__main__':
    main()
