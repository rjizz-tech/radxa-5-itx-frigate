import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'overlay'), str(ROOT/'tools')]
from radxa_frigate.face import decode_yunet, guarded_face_call, yunet_input, safe_library_embedding
from radxa_frigate.locking import accelerator_slot, AcceleratorBusy
from radxa_frigate.runtime import ModelSession
from radxa_frigate.yolo26 import decode, nms
from deployment import environment
from endurance import diskstats, vmstats
from preflight import media_check
from render_config import render
from validate_outputs import compare


def empty_heads():
    outputs = []
    for stride in (8, 16, 32):
        count = (640//stride)**2
        outputs += [np.ones((1, count, 4), np.float32), np.full((1, count, 80), -20, np.float32)]
    return outputs


class DecoderTests(unittest.TestCase):
    def test_yolo26_scalar_reference_and_coordinate_order(self):
        # Independent scalar calculations using vendor grid/stride equations.
        outputs = empty_heads()
        outputs[1][0, 81, 2] = 3
        outputs[0][0, 81] = [0.25, 0.5, 0.75, 1.5]
        result = decode(outputs)
        expected = np.array([[2, 1/(1+np.exp(-3)), 8/640, 10/640, 24/640, 18/640]], np.float32)
        np.testing.assert_allclose(result[:1], expected, rtol=1e-6)
        self.assertEqual(np.count_nonzero(result[:, 1]), 1)
        compare(result, expected)

    def test_empty_and_invalid_exports(self):
        self.assertFalse(decode(empty_heads()).any())
        with self.assertRaises(ValueError):
            decode([np.zeros((1, 300, 6))])  # Ultralytics e2e export is incompatible.
        outputs = empty_heads()
        outputs[0][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode(outputs)

    def test_class_agnostic_nms_and_limit(self):
        outputs = empty_heads()
        outputs[1][0, 81, 0] = 4
        outputs[1][0, 82, 1] = 3
        outputs[0][0, 81] = [1, 1, 1, 1]
        outputs[0][0, 82] = [2, 1, 0, 1]  # Identical box, different class.
        self.assertEqual(np.count_nonzero(decode(outputs)[:, 1]), 1)
        boxes = np.array([[i*10, 0, i*10+1, 1] for i in range(25)], np.float32)
        self.assertEqual(len(nms(boxes, np.arange(25, dtype=np.float32))), 20)

    def test_yunet_grid_decode_and_raw_bgr_input(self):
        outputs = []
        for channels in (1, 1, 4, 10):
            for stride in (8, 16, 32):
                outputs.append(np.zeros((1, (640//stride)**2, channels), np.float32))
        outputs[0][0, 81, 0] = .81
        outputs[3][0, 81, 0] = 1
        outputs[6][0, 81] = [.5, .5, 0, 0]
        rows = decode_yunet(outputs)
        np.testing.assert_allclose(rows[0, :4], [8, 8, 8, 8])
        self.assertAlmostEqual(float(rows[0, -1]), .9, places=6)
        tensor, (ratio, x, y) = yunet_input(np.full((320, 640, 3), [10, 20, 30], dtype=np.uint8))
        self.assertEqual((ratio, x, y), (1, 0, 160))
        np.testing.assert_array_equal(tensor[0, :, 200, 200], [10, 20, 30])
        self.assertFalse(tensor[:, :, :160].any())


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        model = self.root/'test.axmodel'
        model.write_bytes(b'fake-model-only-for-session-contract-test')
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'schema_version': 1, 'models': {'test': {
            'file': model.name, 'sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'input': {'shape': [1, 2], 'dtype': 'float32'}, 'output_names': ['first', 'second']}}}))
        class Session:
            def __init__(this, path, providers):
                self.assertEqual(providers, ['AXCLRTExecutionProvider'])
            def get_inputs(this):
                return [types.SimpleNamespace(name='actual_input', shape=[1, 2], dtype=np.float32)]
            def get_outputs(this):
                return [types.SimpleNamespace(name='second'), types.SimpleNamespace(name='first')]
            def run(this, output_names, inputs):
                self.assertIsNone(output_names)
                self.assertIn('actual_input', inputs)
                return [np.array([2], np.float32), np.array([1], np.float32)]
        self.provider = types.SimpleNamespace(get_available_providers=lambda: ['AXCLRTExecutionProvider'], InferenceSession=Session)
        self.env_patch = patch.dict(os.environ, {'RADXA_LOCK_DIR': str(self.root/'locks'), 'RADXA_STATE_DIR': str(self.root/'state')})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.directory.cleanup()

    def test_actual_input_name_and_local_output_reordering(self):
        with patch.dict(sys.modules, {'axengine': self.provider}):
            outputs = ModelSession(self.manifest, 'test').run(np.ones((1, 2), np.float32))
        self.assertEqual([x.item() for x in outputs], [1, 2])

    def test_provider_fails_closed_and_checksum_validation(self):
        self.provider.get_available_providers = lambda: ['AXEngineExecutionProvider']
        with patch.dict(sys.modules, {'axengine': self.provider}), self.assertRaisesRegex(RuntimeError, 'PCIe AXCL'):
            ModelSession(self.manifest, 'test').run(np.ones((1, 2), np.float32))
        (self.root/'test.axmodel').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            ModelSession(self.manifest, 'test').run(np.ones((1, 2), np.float32))

    def test_busy_face_admission_is_bounded(self):
        result = []
        def face():
            try:
                with accelerator_slot('face', .03):
                    result.append('incorrectly admitted')
            except AcceleratorBusy:
                result.append('bounded skip')
        with accelerator_slot('object', 1):
            thread = threading.Thread(target=face)
            thread.start()
            thread.join(1)
            self.assertFalse(thread.is_alive())
        self.assertEqual(result, ['bounded skip'])

    def test_face_failures_do_not_escape_video_or_enrollment(self):
        class Processor:
            config = types.SimpleNamespace(face_recognition=types.SimpleNamespace(backend='axcl'))
            @guarded_face_call
            def process_frame(self):
                raise RuntimeError('Simulated disconnected AX-M1')
            @guarded_face_call
            def handle_request(self):
                raise AcceleratorBusy('busy')
        self.assertIsNone(Processor().process_frame())
        self.assertFalse(Processor().handle_request()['success'])
        class Recognizer:
            def align_face(self, *args):
                raise ValueError('bad sample')
        self.assertIsNone(safe_library_embedding(Recognizer(), np.zeros((2, 2, 3))))


class DeploymentTests(unittest.TestCase):
    def test_config_profiles_and_mqtt_schema_port(self):
        for profile, size in [('yolov9', 320), ('yolo26n', 640), ('yolo26s', 640)]:
            config = render(profile, tls=True, mqtt_port=8883)
            self.assertEqual(config['model']['width'], size)
            self.assertIsInstance(config['mqtt']['port'], int)
            self.assertEqual(config['record']['continuous']['days'], 7)
            self.assertEqual(config['cameras']['wyze_test']['detect']['fps'], 5)
            self.assertIn('tls_ca_certs', config['mqtt'])

    def test_secrets_are_literal_and_rtsp_credentials_escaped(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'.env'
            values = ['MEDIA_UUID=known', 'FRIGATE_MQTT_HOST=broker', 'FRIGATE_MQTT_USER=user',
                      'FRIGATE_MQTT_PASSWORD=$literal#secret', 'FRIGATE_RTSP_HOST=camera',
                      'FRIGATE_RTSP_PASSWORD=p@ss:/word', 'FRIGATE_RESTREAM_PASSWORD=s@cret']
            file.write_text('\n'.join(values))
            env = environment(file)
            self.assertEqual(env['FRIGATE_MQTT_PASSWORD'], '$literal#secret')
            self.assertEqual(env['FRIGATE_RTSP_URL_PASSWORD'], 'p%40ss%3A%2Fword')

    def test_wrong_uuid_and_missing_mount_rejected(self):
        target = str(ROOT)
        entry = {'filesystems': [{'target': target, 'uuid': 'wrong', 'options': 'rw,relatime'}]}
        with patch('subprocess.check_output', return_value=json.dumps(entry)):
            with self.assertRaisesRegex(ValueError, 'UUID'):
                media_check({'MEDIA_UUID': 'expected', 'MEDIA_DIR': target})
        with patch('subprocess.check_output', return_value='{"filesystems": []}'):
            with self.assertRaisesRegex(ValueError, 'missing'):
                media_check({'MEDIA_UUID': 'expected', 'MEDIA_DIR': target})

    def test_physical_write_and_swap_counters(self):
        result = diskstats('8 0 sda 0 0 0 0 10 0 1234 20 0 0 0', 'sda')
        self.assertEqual(result['bytes_written'], 1234*512)
        self.assertEqual(vmstats('pswpin 2\npswpout 4\npgmajfault 5')['pswpout'], 4)

    def test_patch_on_verified_upstream_sources(self):
        spec = importlib.util.spec_from_file_location('patch_upstream', ROOT/'docker/patch_upstream.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = ROOT/'.cache/upstream'
        if not all((source/path).is_file() for path in module.EXPECTED):
            self.skipTest('Run tools/research.py to obtain pinned upstream source fixtures')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in module.EXPECTED:
                target = root/path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/path, target)
            module.patch(root, ROOT/'overlay')
            for path in module.EXPECTED:
                ast.parse((root/path).read_text())
            changed = (root/'frigate/data_processing/real_time/face.py').read_text()
            self.assertIn('@guarded_face_call', changed)
            self.assertIn('AxclFaceDetector', changed)
            with self.assertRaisesRegex(ValueError, 'Unsupported upstream'):
                module.patch(root, ROOT/'overlay')


if __name__ == '__main__':
    unittest.main()
