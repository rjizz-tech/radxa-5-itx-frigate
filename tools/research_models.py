"""Resolve and download reproducible model inputs for local verification."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache' / 'upstream'


def download(url, path):
    with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    manifest = {'schema_version': 1, 'models': {}}
    for filename, key in [('yolov9-info.json', 'yolov9'), ('yolo26-info.json', 'yolo26')]:
        info = json.loads((CACHE / filename).read_text())
        for entry in info['siblings']:
            name = entry['rfilename']
            if name not in ('frigate-yolov9-tiny.axmodel', 'ax650/yolo26n.axmodel', 'ax650/yolo26s.axmodel'):
                continue
            model_id = 'yolov9' if key == 'yolov9' else Path(name).stem
            manifest['models'][model_id] = {
                'file': Path(name).name,
                'url': f"https://huggingface.co/{info['id']}/resolve/{info['sha']}/{name}",
                'sha256': entry['lfs']['sha256'],
                'source_revision': info['sha'],
                'input': {'shape': [1, 320 if key == 'yolov9' else 640, 320 if key == 'yolov9' else 640, 3], 'dtype': 'uint8'},
                'format': 'frigate-yolo' if key == 'yolov9' else 'axera-yolo26-heads',
            }
    release = next(r for r in json.loads((CACHE / 'pyaxengine-releases.json').read_text()) if r['tag_name'] == '0.1.3-frigate')
    wheel = release['assets'][0]
    manifest['runtime'] = {'url': wheel['browser_download_url'], 'sha256': wheel['digest'].split(':')[1], 'file': wheel['name']}
    download(wheel['browser_download_url'], CACHE / wheel['name'])
    with urllib.request.urlopen('https://api.github.com/repos/NickM-27/facenet-onnx/releases/tags/v1.0', timeout=45) as response:
        faces = json.load(response)
    sources = {}
    for entry in faces['assets']:
        if entry['name'] not in ('facedet.onnx', 'arcface.onnx', 'landmarkdet.yaml'):
            continue
        target = CACHE / entry['name']
        sha = download(entry['browser_download_url'], target)
        sources[entry['name']] = {'url': entry['browser_download_url'], 'sha256': sha, 'file': entry['name']}
        print(entry['name'], target.stat().st_size, sha)
    manifest['face_sources'] = sources
    (ROOT / 'models').mkdir(exist_ok=True)
    (ROOT / 'models' / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
