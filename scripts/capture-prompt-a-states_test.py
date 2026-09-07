"""Adversarial controls for the external capture wrapper; no GPU or model."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp', prefix='capture-control-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.oracle = self.root / 'oracle'
        self.oracle.mkdir()
        self.binary = self.root / 'trace'
        self.binary.write_text('''#!/usr/bin/env python3
import json, os, shutil
from pathlib import Path
out = Path(os.environ['DS4_ORACLE_STATE_OUT'])
mode = os.environ['CONTROL_MODE']
if mode == 'retained':
    for p in Path(os.environ['CONTROL_FIXTURE']).glob('state-*.json'):
        shutil.copyfile(p, out / p.name)
    raise SystemExit(1)
for i in range(9 if mode == 'missing' else 10):
    step = i + (10 if mode == 'wrong_names' else 0)
    data = {'accepted_decode_count': i, 'session_position': 20+i,
            'vocab': 129280, 'argmax': 0, 'top20': [], 'logits': [0.0]*129280}
    if mode == 'position_gap' and i == 5: data['session_position'] += 1
    if mode == 'short': data['logits'] = [0.0]
    if mode == 'nonfinite': data['logits'][0] = float('nan')
    if mode == 'duplicate_steps': data['accepted_decode_count'] = 0
    text = 'garbage' if mode == 'invalid_json' else json.dumps(data)
    (out / ('state-%02d.json' % step)).write_text(text)
raise SystemExit(1)
''')
        self.binary.chmod(0o700)
        self.runner = Path(os.environ['CAPTURE_RUNNER']).resolve()

    def check(self, mode, valid=False):
        result = subprocess.run(['sh', str(self.runner), str(self.binary), str(self.root/'out'), str(self.oracle)],
                                env=dict(os.environ, CONTROL_MODE=mode), capture_output=True, text=True)
        if valid:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, f'{mode} incorrectly accepted: {result.stdout}')

    def test_complete_expected_red(self): self.check('valid', valid=True)
    def test_position_gap(self): self.check('position_gap')
    def test_retained(self):
        fixture = os.environ.get('CAPTURE_FIXTURE')
        if not fixture: self.skipTest('CAPTURE_FIXTURE not set')
        os.environ['CONTROL_FIXTURE'] = fixture
        self.check('retained', valid=True)
    def test_missing(self): self.check('missing')
    def test_wrong_names(self): self.check('wrong_names')
    def test_invalid_json(self): self.check('invalid_json')
    def test_short_logits(self): self.check('short')
    def test_nonfinite(self): self.check('nonfinite')
    def test_duplicate_steps(self): self.check('duplicate_steps')


if __name__ == '__main__': unittest.main()
