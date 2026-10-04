# Software validation — 2026-10-04

Custom image built and loaded in local Docker:

- Tag: `radxa-frigate:0.18.0-radxa.1`
- Platform: `linux/arm64`
- Local image identity: `sha256:ed7608a295f9bb0e504f9eca4a6077f3e7e823d0bb37a1c18193aa5a4ce2794e`
- ARM64 base manifest: `sha256:88e8d411a06fee0926e4fa1d86a8bd48f14256901672f758c8747131e39e7925`

| Check | Result |
|---|---|
| Docker build, exact upstream source hashes and patch anchors | PASS |
| AXEngine package version 0.1.3 in image | PASS |
| Python compile checks | PASS |
| 13 adapter/deployment tests on Windows | PASS |
| Same 13 tests under ARM64 Linux emulation, including POSIX locks | PASS |
| Compose schema and resolved settings with dummy credentials | PASS |
| Native Frigate parser: YOLOv9, YOLO26n, YOLO26s profiles | PASS |
| Native Frigate parser: eight processing pipelines sharing test RTSP | PASS |
| Generated FFmpeg arguments: RKMPP, RGA scaling, recording stream copy | PASS |
| AXCL ArcFace adapter inherits native RGB/resize/padding/normalization | PASS |
| Container `/dev/shm`: tmpfs, 4294967296 bytes | PASS |
| Container `/tmp/cache`: separate tmpfs, 4294967296 bytes | PASS |
| Temporary cache file write/read on tmpfs | PASS |
| Published ports 8971 / 8554 / 8555 TCP+UDP, no 5000 | PASS |

The image smoke test used an overridden Python entrypoint, dummy credentials,
no camera connections, no hardware devices and no published ports. ONNX Runtime
printed CPU-vendor warnings under emulation. Neither accelerator provider
execution nor real recording was exercised by that test.

Unit AXCL sessions are mocked. Numerical decoder tests check independent
scalar grid/stride calculations and error handling; they do not substitute
for comparison against AXERA's executable on held-out inputs.

Remaining release gates: actual Debian 13/vendor kernel/AXCL compatibility,
8GB firmware, compiled YuNet/ArcFace weights and accuracy, YOLO26 vendor output
comparison, native enrollment/MQTT labels, concurrent inference, eight-stream
24-hour SATA/RAM/swap/drop measurements, outage/reboot and controlled power-loss
recovery. These require the board and deployment inputs. Use
[the acceptance runbook](acceptance.md) and [report template](../deploy/acceptance-report.md).

To repeat the image smoke test, bind this project read-only at `/test`, give the
container `--shm-size 4g` and `--tmpfs /tmp/cache:rw,size=4294967296,mode=1777`,
and run `python3 /test/tools/smoke_image.py` with the image's normal entrypoint
overridden. This tests RAM mount configuration without starting the NVR.
