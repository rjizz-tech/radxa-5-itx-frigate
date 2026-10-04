# Backup, upgrades and recovery

## Before an upgrade

Run from the project directory on the board:

```sh
sudo python3 tools/backup.py --output /path/to/backup/frigate-before-upgrade.tar.gz
docker image inspect radxa-frigate:0.18.0-radxa.1 --format '{{.Id}}'
docker image save -o /path/to/backup/frigate-image.tar radxa-frigate:0.18.0-radxa.1
```

The tool stops the service/container, backs up configuration, SQLite DB and its
sidecars, persistent model cache, complete native Face Library and model
manifest/binaries, then restarts a previously running deployment. Archive
permissions are 0600 because `.env` contains credentials. Check exit status,
archive readability and service health. Keep copies away from the recording
disk. Footage is deliberately not copied; retained media remains on SATA.
The backup briefly interrupts recording. A `.partial` file means an incomplete
backup, and must not be considered restore-ready.

Keep a distinct image tag per upgrade. Build a new tag before changing
`FRIGATE_IMAGE`; preserve the old image/model artifacts. Upgrade source patches
must be reviewed against the new upstream release rather than bypassing the
checksum guard. Re-run native enrollment/model accuracy checks after any
compiler/model change. Do not delete old images or manifests until acceptance.

## Restore / rollback

1. Stop `frigate.service`. Confirm the expected media UUID is mounted.
2. Extract a **trusted** backup into an empty staging directory, inspect
   `backup.json` and its `config`, `faces`, `models` and `project` contents.
3. Preserve the current persistent configuration, Face Library and model
   directory under separate recovery names. Restore the complete backed-up
   `config` directory (including DB/WAL/SHM from that quiesced backup), restore
   `faces` to `/srv/frigate/media/clips/faces`, and restore `models` to
   `<project>/models/artifacts`. Do not mix a newer DB/WAL with an older DB.
4. Restore the matching reviewed config/Compose/.env and image tag; use
   `docker image load -i ...` if the old image was exported. Restore the old
   application source too if its Compose or patch schema differs.
5. Run `tools/start.py check`, start the service, check DB integrity and replay
   retained recordings. Re-test recognized labels. Database migrations may
   not be backward compatible, which is why the quiesced DB accompanies the
   matching image. A restored older DB will not automatically index all
   footage created since its backup.

## Faults

- **Missing SATA / wrong UUID:** keep Frigate stopped. Repair the existing
  mount, verify UUID and rw status, then run preflight. Do not create an empty
  replacement recording path on the root filesystem.
- **Cache above 75% / segments older than 180s:** inspect recorder/maintainer
  logs, SATA latency, free space and permissions. Restore normal movement of
  segments before considering more RAM. A full cache can lose new footage.
- **Disk above 85%:** verify seven-day retention and actual camera bitrates,
  resolve stalled cleanup, or adjust the accepted retention/storage plan.
  Do not manually remove arbitrary live DB-linked files as a first response.
- **Face backend unavailable:** read `radxa-status` / `radxa-health.json` under
  container `/dev/shm`; inspect AXCL runtime/firmware, provider visibility,
  manifest hashes and compiled metadata. Enrollment can return a retry/error
  response. Recording continues during contained face inference failures.
  Temporarily disabling faces requires an explicit config change and restart.
- **AX-M1 object detection unavailable:** inspect detector logs and hardware;
  Frigate may restart detection or the application. No CPU fallback is added.
  Recovery/recording gaps must be measured in acceptance, not assumed away.
- **MQTT unavailable:** repair broker connectivity/credentials/TLS. Health
  publish errors are contained and do not stop recording.
- **Power loss:** unfinished `/tmp/cache` segments and decoded frames vanish.
  Files already handed to SATA can still be affected by ordinary drive/page
  cache durability. Let the filesystem recover normally, verify mount/UUID,
  inspect `PRAGMA integrity_check` with Frigate stopped, then play completed
  recordings. Preserve damaged DB/files for investigation before restoring.

Never put the DB, Face Library or final recordings on tmpfs. Do not increase
global dirty-page delays as a write-saving technique.
