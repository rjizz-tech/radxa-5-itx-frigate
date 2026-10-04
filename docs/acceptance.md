# Board acceptance runbook

All rows start **NOT RUN**. Complete `reports/acceptance.md` from
`deploy/acceptance-report.md` with timestamps, versions, commands and evidence.
Use eight real streams for final acceptance; eight copies of the test Wyze
stream only establish an initial processing load. The intended deployment must
pass with facial recognition and the selected object model enabled.

## Storage, RAM and codecs

1. Run `sudo python3 tools/start.py check` and save the output. Record kernel
   package/header version, AXCL driver/runtime/firmware versions and the 8GB
   device's `axcl-smi` output. Confirm UUID and rw filesystem; no formatting.
2. Inspect `docker inspect radxa-frigate`: `HostConfig.ShmSize` must be
   4294967296. Run `docker exec radxa-frigate python3 /opt/radxa/tools/health.py`:
   `/dev/shm` and `/tmp/cache` must both be `tmpfs`, each with 4294967296-byte
   capacity. `/media/frigate` must be the bound SATA filesystem.
3. While streaming, inspect `/tmp/cache/*.mp4` and completed files under
   `/srv/frigate/media/recordings`. Confirm temporary paths resolve onto tmpfs,
   and that completed segments move to SATA promptly. Keep cache occupancy
   well below the 75% warning; investigate any segment older than 180 seconds.
4. Inspect actual FFmpeg command lines/logs: decoding must use `h264_rkmpp` or
   the camera's matching RKMPP codec, and hardware scaling/RGA as supported
   by `preset-rkmpp`. Recording video must use stream copy. Verify 720p/5FPS
   detection and original resolution of durable recordings with `ffprobe`.
5. Play completed recordings through Frigate and externally. Restart the
   service, confirm database integrity and replay the same recordings.

## Eight-stream 24-hour run

First identify the physical SATA device backing the UUID with `lsblk`; never
assume its name from a different boot. Run:

```sh
sudo python3 tools/endurance.py --device sda --hours 24 --interval 60 --output reports/eight-stream-24h.jsonl
```

Replace `sda` with that physical device. The collector reads disk write sectors
(512 bytes per sector in `/proc/diskstats`), swap page counters, host memory,
both RAM mount occupancies, disk occupancy, Frigate camera FPS/drop telemetry
and detector stats. It flags unavailable/stale health samples in the output.
Write sectors include every host workload: compare to an idle baseline and
account for other SATA writes. Use `vmstat 60` and `iostat -dx 60` alongside it
if available. A SATA-backed swap spike undermines the intended RAM-write saving.

Record maxima and trends, camera `camera_fps`, `process_fps`, `skipped_fps` /
drop counters exposed by `/api/stats`, detector latency and completed recording
continuity. Acceptance requires no sustained unexplained skipped-frame backlog,
no missing continuous segments outside the intentional fault tests, no cache
growth trend/stall, no memory exhaustion and no persistent swap churn caused by
the two RAM pools. State observed tolerances and camera rates in the report;
they are not established merely by configuring 5 FPS.

## AI, native enrollment and labels

- Run [model reference comparisons](models.md) on held-out images for baseline
  and any YOLO26 profile being accepted. Save raw tensors, compiled metadata,
  labeled-image results and AXERA reference version. Neither YOLO26 profile
  becomes an accepted default solely because the chip can load it.
- Enroll at least two people through Frigate's native Face Library. Confirm
  persistent images under `media/clips/faces`, rebuild after restart and test
  new images of each enrolled person plus unknown people. Confirm recognized
  `person.sub_label` and corresponding native MQTT event payload labels.
- Run enrollment, ordinary face recognition and eight-camera object detection
  concurrently. Measure detector/face latency with `axcl-smi`, Frigate stats
  and runtime RAM status. Busy faces can be skipped; object detection must not
  be starved. Tune `axcl_wait_ms` only with measurements; native AXCL calls
  have no Python execution-time cancellation.
- Disconnect MQTT briefly. Video files must continue arriving. Reconnect and
  verify labels/stats resume. Face inference failures must remain contained;
  they may delay enrollment/recognition while recording continues.

## Fault and reboot tests

| Test | Expected behavior / evidence |
|---|---|
| Camera unavailable, then restored | Other cameras keep recording; failed camera reconnects; outage appears as a recording gap |
| Wrong UUID or unmounted SATA before startup | Guard rejects startup; no media directory silently created on root filesystem |
| Mount disappears during operation | systemd mount binding stops the service; repair mount before restart |
| Missing/corrupt model at startup | Preflight/checksum refuses launch; error identifies the model |
| Face inference exception/provider failure during operation | Warning in RAM health, failed enrollment response, recorder remains operational |
| Object accelerator failure | Visible detector failure/Frigate watchdog behavior; no CPU inference fallback; document recording interruption and recovery time |
| Broker outage | Recording continues; health publication resumes when reachable |
| Restart/reboot | systemd checks storage/firmware before starting; both tmpfs pools recreated; native Face Library retained |
| Controlled power interruption | Unfinished RAM segments may disappear; completed recordings and DB recover; report filesystem/journal and playback results |

Perform the power-loss test during commissioning with backed-up metadata and
disposable test footage. Graceful restart is **not** a power-loss test. Durable
SATA writes still depend on filesystem/drive flush behavior; the system uses
normal Linux writeback and offers no new power-loss durability guarantee.

For occupancy warnings, check thresholds using a controlled test environment,
not by filling the production video cache. Verify MQTT health JSON warnings
at 75% RAM and 85% disk and verify cache-stall warning separately. A Docker
`unhealthy` state is telemetry, not itself an automatic restart policy.
