"""Collect physical SATA writes, swap and Frigate/RAM stats for a bounded run."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def diskstats(text, device):
    for line in text.splitlines():
        fields = line.split()
        if fields[2] == device:
            return {'writes_completed': int(fields[7]), 'sectors_written': int(fields[9]),
                    'bytes_written': int(fields[9])*512, 'write_ms': int(fields[10])}
    raise ValueError(f'Block device {device} not found in /proc/diskstats')


def vmstats(text):
    values = dict(line.split() for line in text.splitlines())
    return {key: int(values[key]) for key in ('pswpin', 'pswpout', 'pgmajfault')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True, help='Physical disk basename, e.g. sda; never a partition')
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.hours <= 0 or args.interval < 10 or '/' in args.device:
        parser.error('Use positive hours, interval >=10s and a disk basename')
    if Path(f'/sys/class/block/{args.device}/partition').exists():
        parser.error('Measure the physical disk rather than its partition')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = diskstats(Path('/proc/diskstats').read_text(), args.device)
    swap_start = vmstats(Path('/proc/vmstat').read_text())
    start = time.monotonic()
    with args.output.open('x') as output:
        while True:
            elapsed = time.monotonic()-start
            current = diskstats(Path('/proc/diskstats').read_text(), args.device)
            swap = vmstats(Path('/proc/vmstat').read_text())
            result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'elapsed_seconds': elapsed, 'disk': current,
                      'disk_bytes_written_since_start': current['bytes_written']-baseline['bytes_written'],
                      'swap_pages_since_start': {key: swap[key]-swap_start[key] for key in swap}}
            try:
                process = subprocess.run(['docker', 'exec', 'radxa-frigate', 'cat', '/dev/shm/radxa-health.json'],
                                         capture_output=True, text=True, timeout=15)
                if process.returncode:
                    raise ValueError('Container health telemetry unavailable')
                result['health'] = json.loads(process.stdout)
                result['health_age_seconds'] = time.time()-result['health']['timestamp']
            except Exception as exc:
                result['collection_error'] = str(exc)
            output.write(json.dumps(result)+'\n')
            output.flush()
            if elapsed >= args.hours*3600:
                break
            time.sleep(min(args.interval, args.hours*3600-elapsed))
    print(f'Completed: {args.output}; physical disk deltas include all host workloads.')


if __name__ == '__main__':
    main()
