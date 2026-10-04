"""Deployment safety tests without board hardware or Docker."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import deploy
from deployment import environment


class DeploymentTests(unittest.TestCase):
    def test_image_switch_preserves_literal_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'.env'
            path.write_text('# settings\nFRIGATE_IMAGE=old\nFRIGATE_RTSP_PASSWORD=a$#b\n')
            deploy.set_image(path, 'radxa-frigate:git-new')
            env = environment(path, require_inputs=False)
            self.assertEqual(env['FRIGATE_IMAGE'], 'radxa-frigate:git-new')
            self.assertEqual(env['FRIGATE_RTSP_PASSWORD'], 'a$#b')
            self.assertIn('# settings', path.read_text())
            if os.name != 'nt':
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_systemd_rejects_unsafe_path_and_unbound_mount(self):
        for project, media in (('/home/a project', '/srv/frigate/media'),
                               ('/home/%n', '/srv/frigate/media'),
                               ('/opt/frigate', '/other/media')):
            with self.assertRaises(ValueError):
                deploy.service_text(Path(project), media)

    def test_dirty_checkout_rejected(self):
        with patch.object(deploy, 'run', return_value=subprocess.CompletedProcess([], 0, ' M Dockerfile\n')):
            with self.assertRaisesRegex(ValueError, 'Commit or stash'):
                deploy.revision()

    def exercise_update(self, fail_on=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root/'config'
            config_dir.mkdir()
            config = config_dir/'config.yml'
            contents = 'detectors:\n  ax_m1:\n    axcl_model: yolov9\nface_recognition:\n  enabled: false\n'
            config.write_text(contents)
            events = []

            def command(*args, **kwargs):
                events.append(args)
                if fail_on and fail_on(args):
                    raise subprocess.CalledProcessError(1, args)
                return subprocess.CompletedProcess(args, 0)

            original_exists = Path.exists

            def exists(path):
                if str(path).replace('\\', '/') == '/etc/systemd/system/frigate.service':
                    return True
                return original_exists(path)

            with patch.object(deploy, 'ROOT', root), \
                 patch.object(deploy, 'environment', return_value={'CONFIG_DIR': str(config_dir)}), \
                 patch.object(deploy, 'service_text', return_value='unit'), \
                 patch.object(deploy, 'revision', return_value='abc123'), \
                 patch.object(deploy, 'media_check'), \
                 patch.object(deploy, 'check', return_value={'failures': []}), \
                 patch.object(deploy, 'run', side_effect=command), \
                 patch.object(deploy, 'set_image', side_effect=lambda *args: events.append(('switch-image',))), \
                 patch.object(deploy, 'wait_healthy', side_effect=lambda *args: events.append(('healthy',))), \
                 patch.object(Path, 'exists', exists), \
                 patch.object(Path, 'write_text', side_effect=lambda *args: events.append(('write-unit',))):
                try:
                    deploy.deploy('update')
                except subprocess.CalledProcessError:
                    if fail_on is None:
                        raise
            self.assertEqual(config.read_text(), contents)
            self.assertFalse(any('fetch_models.py' in str(arg) for event in events for arg in event))
            return events

    def test_build_failure_leaves_service_and_image_alone(self):
        events = self.exercise_update(lambda args: 'build' in args)
        self.assertNotIn(('switch-image',), events)
        self.assertFalse(any(event[:2] == ('systemctl', 'stop') for event in events))
        self.assertFalse(any('backup.py' in str(arg) for event in events for arg in event))

    def test_backup_failure_prevents_activation(self):
        events = self.exercise_update(lambda args: any('backup.py' in str(arg) for arg in args))
        self.assertNotIn(('switch-image',), events)
        self.assertFalse(any(event[:2] == ('systemctl', 'stop') for event in events))

    def test_success_backs_up_before_switch_and_waits_for_health(self):
        events = self.exercise_update()
        backup = next(i for i, event in enumerate(events) if any('backup.py' in str(arg) for arg in event))
        switch = events.index(('switch-image',))
        restart = events.index(('systemctl', 'restart', 'frigate.service'))
        self.assertLess(backup, switch)
        self.assertLess(switch, restart)
        self.assertEqual(events[-1], ('healthy',))


if __name__ == '__main__':
    unittest.main()
