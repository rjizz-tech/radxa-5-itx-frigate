# Model preparation and validation

The locked baseline is AXERA's Frigate YOLOv9-tiny 320×320. YOLO26n/s use AXERA's
**AX650** compiled 640×640 INT8 variants. AXERA publishes compatible chip models
and an AXCL sample, confirming chip/toolchain support. Frigate integration and
accuracy on this particular 8GB AX-M1 remain acceptance tasks.

## Select object models

```sh
python3 tools/fetch_models.py --profiles yolov9 yolo26n yolo26s
python3 tools/render_config.py --profile yolo26n --template /srv/frigate/config/config.yml --output /srv/frigate/config/config.next.yml
```

Supply the same `--mqtt-tls` / `--mqtt-port` options as appropriate. Back up, then
replace the live configuration with the reviewed next file and restart the
service. The manifest records immutable Hugging Face revisions and SHA256.
Retain prior weights/manifests alongside the prior image for rollback.

The YOLO26 adapter expects six dequantized box/class tensors, paired at strides
8/16/32 in NHWC or flattened channel-last layout. It uses raw class logits,
sigmoid, grid-centered l/t/r/b distances and class-agnostic NMS. Output is
Frigate's `[class, score, ymin, xmin, ymax, xmax]`, limited to 20 detections.
The input is BGR uint8 NHWC. A generic Ultralytics end-to-end export with `[N,6]`
output is rejected. Validate output order against the actual compiled metadata;
do not silently permute six unverified tensors to make a test pass.

## Compile face models

The project pins Frigate's original YuNet and ArcFace ONNX files, plus the CPU
LBF landmark model. Neural inference moves to AXCL; native face alignment,
enrollment folders, cosine matching, labels and MQTT handling remain upstream.

1. Fetch sources with `tools/fetch_models.py --face-sources --config-dir ...`.
2. Prepare representative **real deployment** calibration samples. At least
   32 of each are required; use several hundred with varied lighting, distance,
   pose and identities for serious quantization/accuracy evaluation. Keep
   calibration identities/images separate from accuracy evaluation inputs.
3. Place `.npy` tensors in `calibration/yunet/` and `calibration/arcface/`:
   - YuNet: `[1,3,640,640]` float32, raw BGR pixels, black letterboxing exactly
     through `radxa_frigate.face.yunet_input(image)[0]`.
   - ArcFace: `[1,3,112,112]` float32, **native LBF-aligned** enrollment crops
     processed by Frigate's `ArcfaceEmbedding._preprocess_inputs`. RGB conversion,
     PIL resize/padding and `(pixel/127.5)-1` must match runtime. Avoid using
     unaligned JPEGs or an alternative five-point aligner as calibration inputs.
4. Run in the conversion environment:

   ```sh
   python3 tools/prepare_face.py prepare --calibration calibration
   pulsar2 --version
   pulsar2 build --config .cache/face-build/yunet.json
   pulsar2 build --config .cache/face-build/arcface.json
   ```

   Use the AXERA/Radxa-supported **Pulsar2 4.2** toolchain for the intended AXCL
   runtime, recording its exact version and container digest. Compiler access is
   a deployment dependency; this repository does not download a proprietary
   compiler or claim its conversion has succeeded. The generated configs use
   AX650/NPU3, internal INT8 quantization, float32 NCHW boundaries and float32
   outputs. ArcFace batch is fixed to one. Review configs against that installed
   compiler's schema/help before compiling; current public documentation tracks
   newer compiler versions.
5. Check compiled input/output metadata on the AX-M1. YuNet needs `cls_8/16/32`,
   `obj_8/16/32`, `bbox_8/16/32`, `kps_8/16/32`. ArcFace needs a single `[1,512]`
   float output, originally named `1333`. If compilation renames outputs, review
   and explicitly update the installed manifest's `output_names` mapping to
   the compiled names in those semantic positions. Runtime validates the set
   against actual metadata and reorders locally. It never accepts a guessed
   output count/layout or implicit CPU fallback.
6. Register real compiled files and calibration provenance:

   ```sh
   python3 tools/prepare_face.py register yunet --file .cache/face-build/yunet/yunet.axmodel --compiler 'Pulsar2 4.2; container sha256:ACTUAL_DIGEST' --calibration-archive .cache/face-build/yunet-calibration.tar
   python3 tools/prepare_face.py register arcface --file .cache/face-build/arcface/arcface.axmodel --compiler 'Pulsar2 4.2; container sha256:ACTUAL_DIGEST' --calibration-archive .cache/face-build/arcface-calibration.tar
   ```

   Registration hashes actual binaries. Their manifest status starts as
   `pending-board-acceptance`, rather than claiming validation. Save source,
   calibration, compiler logs and validation report with the release.

## Compare to references

Use held-out inputs with exact preprocessing and preserve raw outputs:

```sh
docker exec radxa-frigate python3 /opt/radxa/tools/infer_dump.py --model yolo26n --input /config/validation/input.npy --output /config/validation/heads.npz
python3 tools/validate_outputs.py --heads /srv/frigate/config/validation/heads.npz --reference reference.json
```

Generate `reference.json` from AXERA's C++ sample running the **same locked model**
and identical input buffer. Record its source commit and compare post-NMS boxes
in model coordinates, before image de-letterboxing. Convert to normalized
`[class,score,ymin,xmin,ymax,xmax]`, retaining its top 20 at confidence 0.45/NMS0.45.
The helper matches by class/coordinates, allows 0.01 score and two-pixel coordinate
differences, and rejects missing/extra detections. Human inspection of overlay
images and a labeled held-out accuracy set is also required. Unit-test scalar
equations do not replace this vendor comparison.

Compare compiled YuNet boxes/keypoints to OpenCV FaceDetectorYN on the identical
640×640 input. Compare ArcFace embeddings to original ONNX: report cosine
agreement plus known/unknown false-match and missed-match rates. Set acceptance
thresholds before choosing between models. Reject a conversion with systematic
color, layout, alignment or normalization disagreement.

Sources: [Pulsar2 config fields](https://pulsar2-docs.readthedocs.io/en/latest/user_guides_config/config.html),
[YOLO26 chip models](https://huggingface.co/AXERA-TECH/yolo26),
[AXERA decoder reference](https://github.com/AXERA-TECH/axcl-samples/blob/main/examples/base/detection.hpp),
[native ArcFace preprocessing](https://github.com/blakeblackshear/frigate/blob/v0.18.0/frigate/embeddings/onnx/face_embedding.py).
