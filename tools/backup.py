"""Quiesced configuration/database/Face Library/model backup (no recordings)."""
import argparse
import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from deployment import ROOT, environment
from preflight import media_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New backup archive path')
    args = parser.parse_args()
    env = environment()
    media_check(env)
    if args.output.exists():
        parser.error('Backup output already exists')
    service_active = subprocess.run(['systemctl', 'is-active', '--quiet', 'frigate.service']).returncode == 0
    compose = ['docker', 'compose', '--project-directory', str(ROOT), '--env-file', os.devnull]
    active = subprocess.run(compose+['ps', '--status', 'running', '-q', 'frigate'],
                            env=env, capture_output=True, text=True, check=True).stdout.strip()
    if service_active:
        subprocess.run(['systemctl', 'stop', 'frigate.service'], check=True)
    elif active:
        subprocess.run(compose+['stop', 'frigate'], env=env, check=True)
    temporary = args.output.with_suffix(args.output.suffix+'.partial')
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as file, tarfile.open(fileobj=file, mode='w:gz') as archive:
            archive.add(env.get('CONFIG_DIR', '/srv/frigate/config'), arcname='config')
            faces = Path(env.get('MEDIA_DIR', '/srv/frigate/media'))/'clips/faces'
            if faces.exists():
                archive.add(faces, arcname='faces')
            archive.add(ROOT/'models/artifacts', arcname='models')
            for name in ('.env', 'compose.yaml', 'Dockerfile', 'models/manifest.json'):
                archive.add(ROOT/name, arcname=f'project/{name}')
            image = subprocess.check_output(['docker', 'image', 'inspect', env.get('FRIGATE_IMAGE', 'radxa-frigate:0.18.0-radxa.1'),
                                             '--format', '{{.Id}}'], text=True).strip()
            metadata = json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'image_id': image}).encode()
            entry = tarfile.TarInfo('backup.json')
            entry.size = len(metadata)
            archive.addfile(entry, io.BytesIO(metadata))
        temporary.replace(args.output)
    finally:
        if service_active:
            subprocess.run(['systemctl', 'start', 'frigate.service'], check=True)
        elif active:
            media_check(env)
            subprocess.run(compose+['start', 'frigate'], env=env, check=True)
    print(f'Backup: {args.output} (contains credentials; permissions 0600)')


if __name__ == '__main__':
    main()
