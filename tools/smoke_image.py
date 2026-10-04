"""Run inside the custom image; tests schema/preprocessing/RAM, not hardware."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/test'))
    args = parser.parse_args()
    # Dummy inputs only. Set before Frigate snapshots FRIGATE_* variables.
    for key, value in {
        'FRIGATE_MQTT_HOST': 'broker.invalid', 'FRIGATE_MQTT_USER': 'dummy',
        'FRIGATE_MQTT_PASSWORD': 'dummy', 'FRIGATE_RTSP_HOST': 'camera.invalid',
        'FRIGATE_RTSP_URL_USER': 'viewer', 'FRIGATE_RTSP_URL_PASSWORD': 'dummy',
        'FRIGATE_RESTREAM_PASSWORD': 'dummy', 'FRIGATE_RESTREAM_URL_PASSWORD': 'dummy',
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(args.project/'tools'))
    sys.path.insert(0, '/opt/frigate')
    from frigate.config import FrigateConfig
    from render_config import render
    from radxa_frigate.face import create_arcface_embedder
    from health import mounts, occupancy
    import numpy as np

    for profile in ('yolov9', 'yolo26n', 'yolo26s'):
        config = FrigateConfig.parse_object(render(profile))
        assert config.detectors['ax_m1'].axcl_model == profile
        assert config.face_recognition.backend == 'axcl'
        assert config.cameras['wyze_test'].detect.fps == 5
        print(f'Native Frigate configuration validated: {profile}')
    eight = FrigateConfig.parse_object(render(test_cameras=8))
    assert len(eight.cameras) == 8
    print('Native Frigate configuration validated: eight test pipelines')
    commands = str(eight.cameras['test_01'].ffmpeg_cmds)
    assert 'rkmpp' in commands
    assert 'rkrga' in commands
    assert 'copy' in commands
    print('Generated FFmpeg arguments include RKMPP, RGA scaling and recording copy')
    embedder = create_arcface_embedder(config.face_recognition)
    image = np.full((112, 112, 3), [10, 20, 30], dtype=np.uint8)
    tensor = embedder._preprocess_inputs([image])[0]['data']
    np.testing.assert_allclose(tensor[0, :, 0, 0], np.array([30, 20, 10])/127.5-1, rtol=1e-6)
    assert tensor.shape == (1, 3, 112, 112) and tensor.dtype == np.float32
    print('Native ArcFace preprocessing inherited correctly')
    types = mounts(Path('/proc/self/mountinfo').read_text())
    for path in ('/dev/shm', '/tmp/cache'):
        assert types[path] == 'tmpfs'
        assert occupancy(path)['total_bytes'] == 4294967296
        print(f'{path}: 4GiB tmpfs confirmed')
    temporary = Path('/tmp/cache/smoke-segment.tmp')
    temporary.write_bytes(b'RAM staging test'*4096)
    assert temporary.read_bytes().startswith(b'RAM staging test')
    temporary.unlink()
    print('Temporary cache write/read succeeded on tmpfs; hardware acceptance remains pending')


if __name__ == '__main__':
    main()
