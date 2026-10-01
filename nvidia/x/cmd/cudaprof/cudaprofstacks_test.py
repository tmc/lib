"""CPU-only tests; the fake wrapper never loads CUDA."""
import ctypes
import importlib.util
import json
import os
import pathlib
import tempfile
import threading
import time
import unittest

SOURCE = pathlib.Path(__file__).with_name('cudaprofstacks.py')


class StackTests(unittest.TestCase):
    def setUp(self):
        scratch = pathlib.Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='cudaprof-python-stacks-', dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.clear)
        old = dict(os.environ)
        self.addCleanup(os.environ.update, old)
        spec = importlib.util.spec_from_file_location('cudaprofstacks_tested', SOURCE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        directory = pathlib.Path(self.tmp.name)
        (directory / 'stacks').mkdir()
        generation = 1234
        worker = directory / ('process-%d-%d' % (os.getpid(), generation))
        worker.mkdir()
        (worker / 'handshake.json').write_text(json.dumps({'process': os.getpid(), 'session': 'test', 'generation': generation, 'transport_ready': True}))
        os.environ.update(CUDAPROF_PYTHON_STACKS='1', CUDAPROF_SESSION='test', CUDAPROF_STACK_DIR=str(directory / 'stacks'))
        self.pending = threading.local()
        self.events = []
        def push(kind, identifier):
            if not hasattr(self.pending, 'ids'):
                self.pending.ids = []
            self.pending.ids.append(identifier)
            self.events.append(('push', kind, identifier))
            return 0
        def pop(kind, output):
            identifier = self.pending.ids.pop()
            ctypes.cast(output, ctypes.POINTER(ctypes.c_uint64))[0] = identifier
            self.events.append(('pop', kind, identifier))
            return 0
        self.mod._agent = (push, pop)

    def records(self):
        records = [json.loads(line) for path in pathlib.Path(self.tmp.name, 'stacks').glob('python-*.jsonl') for line in path.read_text().splitlines()]
        definitions = {r['stack_id']: r['frames'] for r in records if r.get('type') == 'definition'}
        return [dict(r, frames=definitions[r['stack_id']]) for r in records if r.get('type') == 'reference']

    def test_nested_exception(self):
        with self.assertRaisesRegex(ValueError, 'workload'):
            with self.mod.scope('outer'):
                with self.mod.submission('inner'):
                    raise ValueError('workload')
        records = self.records()
        self.assertEqual([r['attribution'] for r in records], ['submission', 'scope_entry'])
        self.assertTrue(all(r['status'] == 'attached' for r in records))
        self.assertTrue(all(r['generation'] == 1234 for r in records))
        self.assertEqual(self.events[1][2], self.events[2][2])
        self.assertEqual(self.events[0][2], self.events[3][2])
        self.assertEqual(records[0]['frames'][0]['function'], 'test_nested_exception')

    def test_disabled(self):
        os.environ.pop('CUDAPROF_PYTHON_STACKS')
        with self.mod.scope('disabled'):
            pass
        self.assertEqual(self.events, [])
        self.assertEqual(self.records(), [])

    def test_failed_push(self):
        self.mod._agent = (lambda kind, identifier: 9, lambda *args: self.fail('pop after failed push'))
        with self.mod.scope('failed'):
            pass
        self.assertEqual(self.records()[0]['status'], 'push_failed')

    def test_failed_pop(self):
        self.mod._agent = (lambda *args: 0, lambda *args: 9)
        with self.mod.scope('failed'):
            pass
        self.assertEqual(self.records()[0]['status'], 'unavailable')

    def test_missing_identity(self):
        os.environ['CUDAPROF_SESSION'] = 'different'
        with self.mod.scope('unbound'):
            pass
        self.assertEqual(self.events, [])
        self.assertEqual(self.records()[0]['status'], 'unavailable')

    def test_threads(self):
        def work():
            with self.mod.submission('thread'):
                pass
        workers = [threading.Thread(target=work) for _ in range(3)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        records = self.records()
        self.assertEqual(len(records), 3)
        self.assertEqual(len({r['external_id'] for r in records}), 3)
        self.assertTrue(all(r['status'] == 'attached' for r in records))

    def test_sampler(self):
        stop = self.mod.start_sampler(0.001)
        time.sleep(0.01)
        stop()
        paths = list(pathlib.Path(self.tmp.name, "stacks").glob("python-wall-*.jsonl"))
        self.assertEqual(len(paths), 1)
        rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all(r["provenance"] == "wall_stack_snapshot" for r in rows))
        self.assertTrue(all(r["generation"] == 1234 for r in rows))
        self.assertTrue(pathlib.Path(str(paths[0]) + ".coverage.json").exists())

    def test_child_reset(self):
        self.mod._reset_child()
        self.assertEqual(next(self.mod._ids), 1)
        self.assertEqual(self.mod._stacks, {})
        self.assertIsNone(self.mod._agent)
        self.assertIsNone(self.mod._writer)

    def test_bound(self):
        self.mod._limit = 512
        with self.mod.scope('bounded'):
            pass
        self.assertEqual(self.records(), [])
        self.assertTrue(list(pathlib.Path(self.tmp.name, 'stacks').glob('*.coverage.json')))

    def tearDown(self):
        if self.mod._writer is not None:
            os.close(self.mod._writer)


if __name__ == '__main__':
    unittest.main()
