# Install and update directly on the ROCK 5 ITX

Run these commands on the board. No Windows machine, image export or cross-build
host is needed. The image builds natively from the checkout's pinned ARM64 base.
The board still needs the validated Debian 13/vendor Rockchip 6.1 BSP, matching
headers, AX-M1 driver/firmware/runtime, Docker Engine with Compose v2, Git,
Python 3.11+ and PyYAML (`sudo apt install git python3 python3-yaml`).
The installer checks prerequisites; it does not replace the BSP or format disks.

## First installation

Clone as your normal board user into a directory you own, without spaces:

```sh
git clone https://github.com/rjizz-tech/radxa-5-itx-frigate.git ~/radxa-frigate
cd ~/radxa-frigate
cp .env.example .env
chmod 600 .env
nano .env
```

Fill the MQTT and camera credentials and the existing recording filesystem UUID.
Configure `/etc/fstab` using `deploy/fstab.example`, mount the existing filesystem
at `/srv/frigate/media`, and verify its UUID and writable status with `findmnt`.
See the [board deployment prerequisites](../README.md#board-deployment).
Keep `MEDIA_DIR=/srv/frigate/media`; the systemd mount binding requires it.
For MQTT TLS, also install the broker CA as `/srv/frigate/config/mqtt-ca.crt`.

Commission recording and object detection before face model conversion:

```sh
sudo python3 tools/deploy.py install --without-faces
```

This creates `config.yml` only if absent, downloads and verifies the baseline
object model, runs preflight, builds an image tagged with the Git revision and
deployment time, installs the systemd service and waits for Docker health.
An existing configuration is preserved; `--without-faces` cannot modify it and
requires faces to be disabled already when retrying an installation.
To start with faces enabled, first complete [model conversion](models.md), then
run `install` without `--without-faces`. Missing compiled face models fail preflight.

After startup, open `https://BOARD_ADDRESS:8971`. Find the initial admin password
with `sudo docker logs radxa-frigate` and change it. Hardware/model accuracy still
requires [board acceptance](acceptance.md).

## Update from Git

Run Git as the same normal user who cloned the repository:

```sh
cd ~/radxa-frigate
git pull --ff-only
sudo python3 tools/deploy.py update
```

Read the new revision's release instructions before activating it. Updates reject
uncommitted source changes and preserve ignored `.env`, model artifacts, backups,
and the external persistent config and recordings. The update builds a new image
while the old recorder runs. It then makes a quiesced backup under `backups/`,
switches `.env` to the new image, refreshes the service and waits for health.
There are brief recording interruptions for backup and activation. Previous Docker
images are retained. Copy backups off the board; they contain credentials.

Git fetch/pull failures leave the running container unchanged. Build, preflight
and backup failures do not switch the image. After activation, a failed health
check returns an error and leaves the selected image for diagnosis; database
migrations make automatic image-only rollback unsafe. Follow
[recovery](recovery.md) to restore the matching database, configuration and image.
Docker health establishes startup/storage health, not camera or model accuracy.

Updates do not overwrite models or regenerate camera configuration. A release
requiring new models or configuration needs the documented migration and a
quiesced backup before changing those files. Compiled face models and enrollment
must be revalidated after model changes.

```sh
sudo systemctl status frigate.service
sudo journalctl -u frigate.service -n 100 --no-pager
sudo docker logs --tail 100 radxa-frigate
```

The service repeats hardware, model and storage UUID preflight on every start and
is bound to the recording mount. Manage normal starts/stops through systemd.
Do not use `git clean -fdx`, which deletes ignored deployment inputs and backups.
