"""Apply narrow integration edits only to exact upstream 0.18.0 sources."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil

EXPECTED = {
    "frigate/detectors/plugins/axengine.py": "a7d3de43017db9da24ca7788d4c6dcda285cbfe9e157057cbeb61f4b910fa91d",
    "frigate/config/classification.py": "fe8899b2e28106ddb1ab41422f4f81fe17300c7084803e2f4290091ff791cd8f",
    "frigate/data_processing/common/face/model.py": "7ed5a4d71d62393a809aaf964d236468915b852172f685968ba839d451948b55",
    "frigate/data_processing/real_time/face.py": "49e683c871c32620906fc2915375af6aafc0866a92adb869c8308fdb3e8e7961",
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one patch anchor: {old[:70]}")
    return source.replace(old, new, 1)


def patch(root, overlay):
    sources = {}
    for relative, checksum in EXPECTED.items():
        body = (root / relative).read_bytes()
        if hashlib.sha256(body).hexdigest() != checksum:
            raise ValueError(f"Unsupported upstream source: {relative}; inspect before upgrading")
        sources[relative] = body.decode()
    config = "frigate/config/classification.py"
    sources[config] = replace_once(sources[config], "from enum import Enum\n", "from enum import Enum\nfrom typing import Literal\n")
    sources[config] = replace_once(sources[config], "class FaceRecognitionConfig(FrigateBaseModel):\n", '''class FaceRecognitionConfig(FrigateBaseModel):
    backend: Literal["native", "axcl"] = Field(default="native", title="Face inference backend")
    axcl_manifest: str = Field(default="/models/manifest.json", title="AXCL model manifest")
    axcl_wait_ms: int = Field(default=20, ge=1, le=1000, title="Maximum AXCL queue wait (ms)")

''')
    model = "frigate/data_processing/common/face/model.py"
    sources[model] = replace_once(sources[model], "self.face_embedder: ArcfaceEmbedding = ArcfaceEmbedding(config.face_recognition)", '''if config.face_recognition.backend == "axcl":
            from radxa_frigate.face import create_arcface_embedder
            self.face_embedder = create_arcface_embedder(config.face_recognition)
        else:
            self.face_embedder = ArcfaceEmbedding(config.face_recognition)''')
    anchor = "img = self.align_face(img, img.shape[1], img.shape[0])\n                    emb = self.face_embedder([img])[0].squeeze()  # type: ignore[arg-type]\n                    face_embeddings_map[name].append(emb)"
    sources[model] = replace_once(sources[model], anchor, '''from radxa_frigate.face import safe_library_embedding
                    emb = safe_library_embedding(self, img)
                    if emb is not None:
                        face_embeddings_map[name].append(emb)''')
    face = "frigate/data_processing/real_time/face.py"
    sources[face] = replace_once(sources[face], "import cv2\n", "from radxa_frigate.face import AxclFaceDetector, guarded_face_call\nimport cv2\n")
    sources[face] = replace_once(sources[face], "        if not all(\n", '''        if self.face_config.backend == "axcl":
            # Neural face detection uses the compiled model, not OpenCV DNN.
            self.model_files.pop("facedet.onnx")
            if self.face_config.model_size != "large":
                raise ValueError("AXCL face recognition requires model_size: large")

        if not all(
''')
    sources[face] = replace_once(sources[face], "    def __build_detector(self) -> None:\n", '''    def __build_detector(self) -> None:
        if self.face_config.backend == "axcl":
            self.face_detector = AxclFaceDetector(self.face_config)
            self.faces_per_second.start()
            return
''')
    sources[face] = replace_once(sources[face], "    def process_frame(self,", "    @guarded_face_call\n    def process_frame(self,")
    sources[face] = replace_once(sources[face], "    def handle_request(\n", "    @guarded_face_call\n    def handle_request(\n")
    # Inference settings are restart-required: do not silently retain old sessions.
    sources[face] = replace_once(sources[face], "        previous_min_area =", '''        for field in ("backend", "axcl_manifest", "axcl_wait_ms", "model_size"):
            if getattr(payload, field) != getattr(self.face_config, field):
                logger.warning("Face backend/model changes require a container restart; retaining running backend")
                return
        previous_min_area =''')
    sources["frigate/detectors/plugins/axengine.py"] = (overlay / "axcl_detector.py").read_text()
    # Verify every result before writing any source file.
    for relative, source in sources.items():
        ast.parse(source, filename=relative)
    for relative, source in sources.items():
        (root / relative).write_text(source, encoding="utf-8", newline="\n")
    shutil.copytree(overlay / "radxa_frigate", root / "radxa_frigate", dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/frigate"))
    parser.add_argument("--overlay", type=Path, default=Path("/opt/radxa/overlay"))
    args = parser.parse_args()
    patch(args.root, args.overlay)
