"""Generate a config without storing camera/broker passwords in YAML."""
import argparse
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def render(profile="yolov9", tls=False, template=None, mqtt_port=None, test_cameras=1):
    config = yaml.safe_load((template or ROOT / "config/config.example.yml").read_text())
    selected = yaml.safe_load((ROOT / "config/model-profiles.yml").read_text())[profile]
    config['detectors']['ax_m1']['axcl_model'] = selected['axcl_model']
    for key in ('path', 'width', 'height'):
        config['model'][key] = selected[key]
    if mqtt_port is not None:
        if not 1 <= mqtt_port <= 65535:
            raise ValueError('MQTT port must be 1..65535')
        config['mqtt']['port'] = mqtt_port
    if tls:
        config['mqtt']['tls_ca_certs'] = '/config/mqtt-ca.crt'
        config['mqtt']['tls_insecure'] = False
    if not 1 <= test_cameras <= 8:
        raise ValueError('Test camera count must be 1..8')
    if test_cameras > 1:
        # Eight independent Frigate pipelines share one go2rtc RTSP source, so
        # the Wyze does not need eight direct connections. This is a load test.
        sample = config['cameras']['wyze_test']
        config['cameras'] = {f'test_{number:02}': copy.deepcopy(sample) for number in range(1, test_cameras+1)}
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('yolov9', 'yolo26n', 'yolo26s'), default='yolov9')
    parser.add_argument('--mqtt-tls', action='store_true', help='Requires mqtt-ca.crt in persistent /config')
    parser.add_argument('--mqtt-port', type=int, default=None)
    parser.add_argument('--test-cameras', type=int, default=1, help='1..8 processing pipelines sharing the test stream; not real-camera acceptance')
    parser.add_argument('--template', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'config/config.yml')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; back it up and choose a new output path before replacing a live config')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(render(args.profile, args.mqtt_tls, args.template, args.mqtt_port, args.test_cameras), sort_keys=False))
