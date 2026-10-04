"""Read-only hardware/storage/model checks. Never formats, mounts or tunes disks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from deployment import ROOT, environment

DEVICES = ['/dev/dri', '/dev/dma_heap', '/dev/rga', '/dev/mpp_service',
           '/dev/axcl_host', '/dev/ax_mmb_dev', '/dev/msg_userdev']


def media_check(env):
    path = Path(env.get('MEDIA_DIR', '/srv/frigate/media'))
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('Media mount must be an absolute, non-symlink path')
    output = subprocess.check_output(['findmnt', '--json', '--mountpoint', str(path),
                                      '--output', 'TARGET,UUID,OPTIONS'], text=True, timeout=10)
    entries = json.loads(output).get('filesystems', [])
    if len(entries) != 1 or entries[0]['target'] != str(path):
        raise ValueError('Dedicated media mount is missing')
    entry = entries[0]
    if (entry.get('uuid') or '').lower() != env['MEDIA_UUID'].lower():
        raise ValueError('Mounted media UUID differs from deployment UUID')
    if 'rw' not in entry['options'].split(','):
        raise ValueError('Media filesystem is not writable')
    return entry


def check(env):
    import yaml
    failures, warnings = [], []
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        failures.append('Requires Linux ARM64 on the ROCK 5 ITX')
    release = platform.release()
    if not release.startswith('6.1.') or not any(tag in release.lower() for tag in ('rockchip', 'vendor-rk35')):
        failures.append('Requires validated Armbian vendor 6.1 Rockchip kernel; verify BSP provenance')
    if not Path(f'/lib/modules/{release}/build').exists():
        failures.append('Matching running-kernel headers missing')
    os_release = Path('/etc/os-release')
    if not os_release.exists() or 'VERSION_ID="13"' not in os_release.read_text():
        failures.append('Requires Debian 13 userspace')
    for node in DEVICES + ['/usr/bin/axcl', '/usr/lib/axcl']:
        if not Path(node).exists():
            failures.append(f'Missing device/runtime: {node}')
    try:
        media = media_check(env)
    except Exception as exc:
        media = None
        failures.append(f'Storage guard failed: {exc}')
    config_dir = Path(env.get('CONFIG_DIR', '/srv/frigate/config'))
    config_path = config_dir/'config.yml'
    try:
        config = yaml.safe_load(config_path.read_text())
        if config['mqtt']['port'] != int(env.get('FRIGATE_MQTT_PORT', 1883)):
            failures.append('MQTT port differs between .env and config.yml; regenerate config')
        tls = bool(config['mqtt'].get('tls_ca_certs'))
        if tls != (env.get('FRIGATE_MQTT_TLS', 'false').lower() == 'true'):
            failures.append('MQTT TLS settings differ between .env and config.yml')
        if tls and not (config_dir/'mqtt-ca.crt').is_file():
            failures.append('Missing MQTT CA certificate')
        selected = config['detectors']['ax_m1']['axcl_model']
        required = [selected]
        if config.get('face_recognition', {}).get('enabled'):
            required += ['yunet', 'arcface']
            if not (config_dir/'model_cache/facedet/landmarkdet.yaml').is_file():
                failures.append('Missing persistent landmark model; run fetch_models --face-sources --config-dir')
        manifest = json.loads((ROOT/'models/artifacts/manifest.json').read_text())
        for name in required:
            spec = manifest['models'][name]
            file = (ROOT/'models/artifacts'/spec['file']).resolve()
            if not file.is_relative_to((ROOT/'models/artifacts').resolve()):
                raise ValueError('Model path leaves artifacts directory')
            with file.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != spec['sha256']:
                    failures.append(f'Model checksum mismatch: {name}')
        if selected != 'yolov9':
            warnings.append('YOLO26 selected; board/reference accuracy acceptance is required')
    except Exception as exc:
        failures.append(f'Configuration/model preflight failed: {exc}')
    if config_dir.exists() and not os.access(config_dir, os.W_OK):
        failures.append('Persistent config directory is not writable')
    try:
        result = subprocess.run(['/usr/bin/axcl/axcl-smi'], capture_output=True, text=True, timeout=15)
        if result.returncode or not result.stdout.strip():
            failures.append('axcl-smi failed; verify firmware/driver compatibility and 8GB card visibility')
        axcl = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        failures.append('axcl-smi unavailable or timed out')
        axcl = ''
    return {'kernel': release, 'media': media, 'axcl_smi': axcl, 'warnings': warnings, 'failures': failures}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path)
    args = parser.parse_args()
    result = check(environment(args.env))
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['failures']))
