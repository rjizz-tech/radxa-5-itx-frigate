"""Fetch small upstream references and immutable artifact metadata, never credentials."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "upstream"


def get(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": "radxa-frigate-build", **(headers or {})})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read(), dict(response.headers)


def main():
    paths = [
        "frigate/detectors/plugins/axengine.py", "frigate/detectors/detector_config.py",
        "frigate/data_processing/real_time/face.py", "frigate/data_processing/common/face/model.py",
        "frigate/embeddings/onnx/face_embedding.py", "frigate/embeddings/onnx/base_embedding.py",
        "frigate/util/model.py", "frigate/record/maintainer.py",
        "frigate/data_processing/real_time/api.py", "frigate/embeddings/maintainer.py",
        "docker/rockchip/rk.hcl", "docker/main/Dockerfile", "docker/main/requirements.txt",
        "frigate/config/classification.py", "frigate/config/config.py", "frigate/const.py",
        "frigate/video/__init__.py", "frigate/video/ffmpeg.py", "frigate/detectors/__init__.py",
    ]
    CACHE.mkdir(parents=True, exist_ok=True)
    def fetch(path):
        try:
            body, _ = get(f"https://raw.githubusercontent.com/blakeblackshear/frigate/v0.18.0/{path}")
            target = CACHE / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            return path, hashlib.sha256(body).hexdigest()
        except Exception as exc:
            return path, str(exc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        print(json.dumps(dict(executor.map(fetch, paths)), indent=2))
    for name, url in {
        "axera-detection.hpp": "https://raw.githubusercontent.com/AXERA-TECH/axcl-samples/main/examples/base/detection.hpp",
        "axera-common.hpp": "https://raw.githubusercontent.com/AXERA-TECH/axcl-samples/main/examples/base/common.hpp",
        "pyaxengine-releases.json": "https://api.github.com/repos/AXERA-TECH/pyaxengine/releases",
        "frigate-tree.json": "https://api.github.com/repos/blakeblackshear/frigate/git/trees/v0.18.0?recursive=1",
        "yolo26-info.json": "https://huggingface.co/api/models/AXERA-TECH/yolo26?blobs=true",
        "yolov9-info.json": "https://huggingface.co/api/models/AXERA-TECH/frigate-resource/revision/axmodel?blobs=true",
        "pulsar-config.html": "https://pulsar2-docs.readthedocs.io/en/latest/user_guides_config/config.html",
    }.items():
        try:
            body, _ = get(url)
            (CACHE / name).write_bytes(body)
            print(f"Saved {name}")
        except Exception as exc:
            print(f"Unavailable {name}: {exc}")
    token = json.loads(get("https://ghcr.io/token?scope=repository:blakeblackshear/frigate:pull")[0])["token"]
    body, headers = get("https://ghcr.io/v2/blakeblackshear/frigate/manifests/0.18.0-rk", {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json",
    })
    (CACHE / "image-manifest.json").write_bytes(body)
    print("Frigate image digest:", next((v for k, v in headers.items() if k.lower() == "docker-content-digest"), hashlib.sha256(body).hexdigest()))


if __name__ == "__main__":
    main()
