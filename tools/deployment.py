"""Shared deployment inputs; .env is data, never executed as a shell script."""
import os
from pathlib import Path
import re
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def environment(path=None, require_inputs=True):
    values = dict(os.environ)
    source = path or ROOT/'.env'
    lines = source.read_text().splitlines() if source.exists() else []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition('=')
        if not separator or not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            raise ValueError('Invalid .env assignment; use KEY=literal-value')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value  # Literal: no $ expansion, escapes or inline comments.
    for key in ('MEDIA_UUID', 'FRIGATE_MQTT_HOST', 'FRIGATE_MQTT_USER', 'FRIGATE_MQTT_PASSWORD',
                'FRIGATE_RTSP_HOST', 'FRIGATE_RTSP_PASSWORD', 'FRIGATE_RESTREAM_PASSWORD'):
        if require_inputs and (not values.get(key) or values[key].startswith('REPLACE_')):
            raise ValueError(f'Deployment input missing: {key}')
        if not require_inputs:
            values.setdefault(key, 'unused-during-stop')
    values['FRIGATE_RTSP_URL_USER'] = quote(values.get('FRIGATE_RTSP_USER', 'viewer'), safe='')
    values['FRIGATE_RTSP_URL_PASSWORD'] = quote(values['FRIGATE_RTSP_PASSWORD'], safe='')
    values['FRIGATE_RESTREAM_URL_PASSWORD'] = quote(values['FRIGATE_RESTREAM_PASSWORD'], safe='')
    return values
