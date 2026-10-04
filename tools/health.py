"""Container health and resource telemetry. Diagnostics are kept in RAM."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def occupancy(path):
    stats = os.statvfs(path)
    total = stats.f_blocks * stats.f_frsize
    available = stats.f_bavail * stats.f_frsize
    return {'total_bytes': total, 'available_bytes': available, 'used_percent': 100*(total-available)/total if total else 100}


def memory_stats(text):
    values = {}
    for line in text.splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.strip().split()[0])*1024
    return {key: values.get(key) for key in ('MemAvailable', 'SwapTotal', 'SwapFree', 'Dirty', 'Writeback')}


def mounts(text):
    return {line.split()[4]: line.split(' - ', 1)[1].split()[0] for line in text.splitlines()}


def snapshot():
    result = {'timestamp': time.time(), 'warnings': [], 'failures': []}
    mount_types = mounts(Path('/proc/self/mountinfo').read_text())
    for path, key, threshold in (('/dev/shm', 'frames', 75), ('/tmp/cache', 'cache', 75), ('/media/frigate', 'disk', 85)):
        try:
            result[key] = occupancy(path)
            result[key]['filesystem'] = mount_types.get(path)
            if key in ('frames', 'cache'):
                if mount_types.get(path) != 'tmpfs':
                    result['failures'].append(f'{path} is not tmpfs')
                if result[key]['total_bytes'] < 4*1024**3:
                    result['failures'].append(f'{path} capacity is below 4GiB')
            if result[key]['used_percent'] >= threshold:
                result['warnings'].append(f'{key} usage above {threshold}%')
        except OSError:
            result['failures'].append(f'{path} unavailable')
    result['host_memory'] = memory_stats(Path('/proc/meminfo').read_text())
    swap = result['host_memory']
    if swap['SwapTotal'] and swap['SwapFree'] < swap['SwapTotal']:
        result['warnings'].append('Host swap is in use; inspect vmstat si/so for cache-related writes')
    # The normal maintainer should move completed camera segments promptly.
    old_segments = []
    for path in Path('/tmp/cache').glob('*.mp4'):
        try:
            if time.time()-path.stat().st_mtime > 180:
                old_segments.append(path)
        except FileNotFoundError:
            pass  # Normal race with the recording maintainer moving segments.
    result['old_cache_segments'] = len(old_segments)
    if old_segments:
        result['warnings'].append('Recording cache contains segments older than 180s; check maintainer/disk health')
    try:
        with urllib.request.urlopen('http://127.0.0.1:5000/api/stats', timeout=5) as response:
            stats = json.load(response)
        result['cameras'] = stats.get('cameras', {})
        result['detectors'] = stats.get('detectors', {})
    except Exception:
        result['failures'].append('Frigate internal stats API unavailable')
    result['inference'] = {}
    for path in Path('/dev/shm/radxa-status').glob('*.json'):
        try:
            entry = json.loads(path.read_text())
            result['inference'][path.stem] = entry
            if entry.get('error'):
                result['warnings'].append(f'{path.stem} inference unavailable')
        except (ValueError, OSError):
            pass
    return result


def publish(result):
    """MQTT connectivity is optional for health; broker outages are contained."""
    try:
        import yaml
        import paho.mqtt.publish
        config = yaml.safe_load(Path('/config/config.yml').read_text()).get('mqtt', {})
        def resolve(value):
            if isinstance(value, str) and value.startswith('{FRIGATE_') and value.endswith('}'):
                return os.environ[value[1:-1]]
            return value
        if not config.get('enabled', False):
            return
        tls = None
        if config.get('tls_ca_certs'):
            tls = {'ca_certs': resolve(config['tls_ca_certs'])}
        elif os.environ.get('FRIGATE_MQTT_TLS', 'false').lower() == 'true':
            raise ValueError('MQTT TLS enabled in env but CA config is missing')
        auth = {'username': resolve(config['user']), 'password': resolve(config['password'])} if config.get('user') else None
        paho.mqtt.publish.single(
            f"{config.get('topic_prefix', 'frigate')}/radxa/health",
            json.dumps(result), hostname=resolve(config['host']), port=int(resolve(config.get('port', 1883))),
            auth=auth, tls=tls, retain=True, client_id='radxa_health',
        )
    except Exception:
        # Do not log secrets or make local NVR health depend on the remote broker.
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = snapshot()
    Path('/dev/shm/radxa-health.json').write_text(json.dumps(result))
    # Publish in a child: an unreachable broker cannot hold the healthcheck open.
    import multiprocessing
    child = multiprocessing.Process(target=publish, args=(result,))
    child.start()
    child.join(5)
    if child.is_alive():
        child.terminate()
        child.join(1)
    if not args.check:
        print(json.dumps(result, indent=2))
    raise SystemExit(1 if result['failures'] else 0)


if __name__ == '__main__':
    main()
