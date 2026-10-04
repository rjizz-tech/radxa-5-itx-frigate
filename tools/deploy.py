"""Install or update a Git checkout directly on the Linux ARM64 board."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from deployment import ROOT, environment
from preflight import check, media_check


def run(*command, **kwargs):
    return subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def revision():
    # Local deployment inputs are ignored; tracked edits make a release ambiguous.
    git = ('git', '-c', f'safe.directory={ROOT}')
    dirty = run(*git, 'status', '--porcelain', '--untracked-files=normal',
                capture_output=True, text=True).stdout
    if dirty.strip():
        raise ValueError('Commit or stash source changes before deploying')
    return run(*git, 'rev-parse', '--short=12', 'HEAD',
               capture_output=True, text=True).stdout.strip()


def service_text(project, media):
    # Systemd command lines interpret whitespace, %, quotes and backslashes.
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(project)):
        raise ValueError('Use a project path with only letters, digits, /, _, . and -')
    if media != '/srv/frigate/media':
        raise ValueError('The supplied service requires MEDIA_DIR=/srv/frigate/media')
    return (ROOT/'deploy/frigate.service.in').read_text().replace('@PROJECT_DIR@', str(project))


def set_image(path, image):
    original = path.stat()
    lines = path.read_text().splitlines()
    lines = [line for line in lines if not line.strip().startswith('FRIGATE_IMAGE=')]
    temporary = path.with_name(path.name+'.deploy-tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        output.write('\n'.join(lines)+f'\nFRIGATE_IMAGE={image}\n')
    if hasattr(os, 'chown'):
        os.chown(temporary, original.st_uid, original.st_gid)
    temporary.replace(path)


def wait_healthy(timeout):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        result = subprocess.run(['docker', 'inspect', 'radxa-frigate', '--format',
                                 '{{json .State}}'], capture_output=True, text=True)
        if result.returncode == 0:
            state = json.loads(result.stdout)
            health = state.get('Health', {}).get('Status')
            if state.get('Running') and health == 'healthy':
                return
            if health == 'unhealthy' or state.get('Status') == 'dead':
                break
        time.sleep(5)
    raise RuntimeError('Deployment did not become healthy; inspect journalctl -u frigate '
                       'and docker logs radxa-frigate. Restore the matching backup/image '
                       'using docs/recovery.md before rolling back database migrations.')


def deploy(action, without_faces=False, timeout=600):
    import yaml
    env = environment()
    unit = service_text(ROOT, env.get('MEDIA_DIR', '/srv/frigate/media'))
    if action == 'install' and Path('/etc/systemd/system/frigate.service').exists():
        raise ValueError('Service already installed; use update to back up the existing deployment')
    if action == 'update' and not Path('/etc/systemd/system/frigate.service').exists():
        raise ValueError('Service is not installed; run install first')
    sha = revision()
    run('docker', 'compose', 'version')
    media_check(env)  # Never create a recording directory on an unmounted root disk.
    config_dir = Path(env.get('CONFIG_DIR', '/srv/frigate/config'))
    if not config_dir.is_absolute():
        raise ValueError('CONFIG_DIR must be absolute')
    config_path = config_dir/'config.yml'
    if action == 'install' and not config_path.exists():
        from render_config import render
        config = render(tls=env.get('FRIGATE_MQTT_TLS', 'false').lower() == 'true',
                        mqtt_port=int(env.get('FRIGATE_MQTT_PORT', 1883)))
        if without_faces:
            config['face_recognition']['enabled'] = False
        config_dir.mkdir(parents=True, exist_ok=True)
        with config_path.open('x') as output:
            output.write(yaml.safe_dump(config, sort_keys=False))
    config = yaml.safe_load(config_path.read_text())
    if without_faces and config.get('face_recognition', {}).get('enabled'):
        raise ValueError('--without-faces cannot disable faces in an existing config; '
                         'edit it explicitly')
    selected = config['detectors']['ax_m1']['axcl_model']
    fetch = [sys.executable, str(ROOT/'tools/fetch_models.py'), '--profiles', selected]
    if config.get('face_recognition', {}).get('enabled'):
        fetch += ['--face-sources', '--config-dir', str(config_dir)]
    # Updates leave live model files untouched; model changes require a separate
    # quiesced backup and board accuracy validation (docs/models.md).
    if action == 'install':
        run(*fetch)
    report = check(env)
    print(json.dumps(report, indent=2), flush=True)
    if report['failures']:
        raise ValueError('Preflight failed; deployment was not activated')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    image = f'radxa-frigate:git-{sha}-{stamp.lower()}'
    run(sys.executable, str(ROOT/'tools/start.py'), 'build', '--image', image)
    run(sys.executable, str(ROOT/'tools/start.py'), 'config', '--image', image)
    if action == 'update':
        backup = ROOT/'backups'/f'before-{stamp}.tar.gz'
        run(sys.executable, str(ROOT/'tools/backup.py'), '--output', str(backup))
        print(f'Rollback metadata/config/models: {backup}', flush=True)
    # Only switch the service after a successful build and (for updates) backup.
    if action == 'update':
        run('systemctl', 'stop', 'frigate.service')
    set_image(ROOT/'.env', image)
    Path('/etc/systemd/system/frigate.service').write_text(unit)
    run('systemctl', 'daemon-reload')
    run('systemctl', 'enable', 'frigate.service')
    run('systemctl', 'restart', 'frigate.service')
    wait_healthy(timeout)
    print(f'Healthy deployment: {sha}, image {image}. Previous images are retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'update'))
    parser.add_argument('--without-faces', action='store_true',
                        help='Create initial object/recording config with faces disabled')
    parser.add_argument('--timeout', type=int, default=600, help='Health wait in seconds')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        parser.error('Run this command directly on the Linux ARM64 Radxa board')
    if os.geteuid() != 0:
        parser.error('Run with sudo: service installation and quiesced backups need root')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    # Prevent concurrent install/update runs, including from another checkout.
    import fcntl
    with open('/run/lock/radxa-frigate-deploy.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            deploy(args.action, args.without_faces, args.timeout)
        except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            parser.exit(1, f'Deployment failed: {exc}\n')


if __name__ == '__main__':
    main()
