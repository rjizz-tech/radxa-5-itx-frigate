"""Guarded lifecycle for a board deployment, including URL credential escaping."""
import argparse
import json
import os
import subprocess
from deployment import ROOT, environment
from preflight import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'build', 'up', 'down', 'config'])
    parser.add_argument('--attached', action='store_true', help='Foreground supervision by systemd')
    args = parser.parse_args()
    if args.action == 'down':
        try:
            env = environment(require_inputs=False)
        except ValueError:
            # Stopping must still work after a damaged credentials file.
            env = environment(ROOT/'.missing-stop-env', require_inputs=False)
    else:
        env = environment()
    # Both build and launch fail closed when the storage/hardware deployment is
    # incomplete; direct cross-building can use docker compose build separately.
    if args.action in ('check', 'up'):
        report = check(env)
        print(json.dumps(report, indent=2))
        if report['failures']:
            raise SystemExit(1)
        if args.action == 'check':
            return
    command = ['docker', 'compose', '--project-directory', str(ROOT), '--env-file', os.devnull]
    command += {'build': ['build', 'frigate'], 'up': ['up', '--no-build']+([] if args.attached else ['-d']),
                'down': ['down'], 'config': ['config', '--quiet']}[args.action]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
