"""Opt-in Python stacks for cudaprof; no interpreter hooks or dependencies.

Use ``with scope('forward'):`` around a synchronous region, or
``with submission('launch'):`` immediately around one synchronous CUDA call.
A scope is not an exact submission stack. Neither propagates to worker threads.
``start_sampler()`` records wall stack snapshots, including blocked threads;
these are not CPU execution samples. All collection requires capture --stacks.
"""
import contextlib
import atexit
import ctypes
import itertools
import glob
import json
import os
import sys
import threading
import time

_lock = threading.RLock()
_pid = os.getpid()
_generation = time.monotonic_ns()
_writer = None
_agent = None
_ids = itertools.count(1)
_stacks = {}
_limit = 8 * 1024 * 1024
_bytes = 0
_saturated = False
_failure = None
_attempted = _recorded = _dropped = _truncated = 0


def _reset_child():
    global _lock, _pid, _generation, _writer, _agent, _ids, _bytes, _saturated, _attempted, _recorded, _dropped, _truncated, _stacks, _failure
    if _writer is not None:
        try:
            os.close(_writer)
        except OSError:
            pass
    _lock = threading.RLock()
    _pid, _generation = os.getpid(), time.monotonic_ns()
    _writer = _agent = None
    _ids = itertools.count(1)
    _stacks = {}
    _bytes, _saturated = 0, False
    _failure = None
    _attempted = _recorded = _dropped = _truncated = 0


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_child)


def _enabled():
    return _failure is None and os.environ.get('CUDAPROF_PYTHON_STACKS') == '1' and bool(os.environ.get('CUDAPROF_STACK_DIR'))


def _observer_failed(error):
    global _failure, _writer
    if _failure is not None:
        return
    _failure = str(error)
    if _writer is not None:
        try:
            os.close(_writer)
        except OSError:
            pass
        _writer = None
    # The failed journal cannot promise a terminal coverage record. Retain a
    # separate unavailable receipt when storage permits, otherwise diagnose
    # through stderr without risking another exception in the workload.
    try:
        path = os.path.join(os.environ['CUDAPROF_STACK_DIR'], 'python-%d-%d.coverage.json' % (_pid, _generation))
        with open(path, 'w') as out:
            json.dump({'status': 'unavailable', 'reason': _failure}, out)
    except OSError:
        pass
    try:
        sys.stderr.write('cudaprof: Python stack observer unavailable: %s\n' % _failure)
    except (OSError, ValueError):
        pass


def _emit(record):
    global _writer, _bytes, _saturated
    if _saturated or _failure is not None:
        return False
    try:
        if _writer is None:
            path = os.path.join(os.environ['CUDAPROF_STACK_DIR'], 'python-%d-%d.jsonl' % (_pid, _generation))
            _writer = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        data = (json.dumps(record, separators=(',', ':'), ensure_ascii=True) + '\n').encode()
        if _bytes + len(data) > _limit - 512:
            _saturated = True
            receipt = os.path.join(os.environ['CUDAPROF_STACK_DIR'], 'python-%d-%d.coverage.json' % (_pid, _generation))
            with open(receipt, 'w') as out:
                json.dump({'status': 'truncated', 'reason': 'journal byte limit'}, out)
            return False
        if os.write(_writer, data) != len(data):
            raise OSError('short stack journal write')
        _bytes += len(data)
        return True
    except OSError as error:
        _observer_failed(error)
        return False


def _identity():
    directory = os.path.dirname(os.environ['CUDAPROF_STACK_DIR'])
    candidates = []
    for path in glob.glob(os.path.join(directory, 'process-%d-*' % os.getpid(), 'handshake.json')):
        try:
            with open(path) as inp:
                record = json.load(inp)
            if (record.get('session') == os.environ.get('CUDAPROF_SESSION')
                    and record.get('process') == os.getpid()
                    and record.get('transport_ready') is True
                    and isinstance(record.get('generation'), int)):
                candidates.append(record)
        except (OSError, ValueError):
            continue
    return candidates[0] if len(candidates) == 1 else None


def _native():
    global _agent
    if _agent is None:
        lib = ctypes.CDLL(None)
        push, pop = lib.cudaprof_push_external, lib.cudaprof_pop_external
        push.argtypes, push.restype = [ctypes.c_uint32, ctypes.c_uint64], ctypes.c_int
        pop.argtypes, pop.restype = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int
        _agent = (push, pop)
    return _agent


def _chain(frame):
    result = []
    while frame is not None and len(result) < 64:
        result.append((frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno))
        frame = frame.f_back
    return result, frame is not None


@contextlib.contextmanager
def _region(name, provenance):
    global _attempted, _recorded, _dropped, _truncated
    if not _enabled():
        yield
        return
    pid, tid = os.getpid(), threading.get_native_id()
    kind = 5  # CUPTI CUSTOM2; dedicated Python attribution namespace.
    frames, truncated = _chain(sys._getframe(2))
    identity = _identity()
    record = {'version': 1, 'session': os.environ.get('CUDAPROF_SESSION', ''),
              'generation': identity['generation'] if identity else 0,
              'process': str(identity['generation'] if identity else _generation), 'pid': pid,
              'language': 'python', 'attribution': provenance,
              'external_kind': kind,
              'frames': [{'function': fn[:4096], 'file': file[:4096], 'line': line}
                         for file, fn, line in frames],
              'truncated': truncated, 'status': 'unavailable'}
    with _lock:
        _attempted += 1
        _truncated += int(truncated)
        identifier = (1 << 62) | next(_ids)
        record['external_id'] = identifier
        pushed = False
        try:
            push, pop = _native()
            pushed = identity is not None and push(kind, identifier) == 0
            record['status'] = 'attached' if pushed else ('push_failed' if identity else 'unavailable')
        except (AttributeError, OSError):
            pass
    try:
        yield
    finally:
        if pushed:
            if os.getpid() != pid or threading.get_native_id() != tid:
                record['status'] = 'unavailable'
            else:
                value = ctypes.c_uint64()
                try:
                    result = pop(kind, ctypes.byref(value))
                except (AttributeError, OSError):
                    result = -1
                if result != 0 or value.value != identifier:
                    record['status'] = 'unavailable'
        if os.getpid() == pid:
            with _lock:
                key = (record['generation'], tuple((f['file'], f['function'], f['line']) for f in record['frames']))
                if key not in _stacks:
                    stack_id = len(_stacks) + 1
                    definition = {k: record[k] for k in ('version', 'process', 'session', 'generation', 'pid', 'language', 'frames')}
                    definition.update(type='definition', stack_id=stack_id, truncated=truncated)
                    if _emit(definition):
                        _stacks[key] = stack_id
                if key in _stacks:
                    record.update(type='reference', stack_id=_stacks[key])
                    record.pop('frames')
                    emitted = _emit(record)
                else:
                    emitted = False
                if emitted and record['status'] == 'attached':
                    _recorded += 1
                else:
                    _dropped += 1


def scope(name):
    """Capture the caller at scope entry; launches inherit the explicit region."""
    return _region(name, 'scope_entry')


def submission(name):
    """Capture immediately around one synchronous submission on this thread."""
    return _region(name, 'submission')


def start_sampler(interval=0.01):
    """Start wall stack snapshots and return a stop function. Minimum 1 ms."""
    if interval < 0.001:
        raise ValueError('stack sampling interval must be at least 0.001 seconds')
    stop = threading.Event()
    if not _enabled():
        return stop.set
    pid = os.getpid()
    identity = _identity()
    path = os.path.join(os.environ['CUDAPROF_STACK_DIR'], 'python-wall-%d-%d.jsonl' % (pid, time.monotonic_ns()))
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        _observer_failed(error)
        return stop.set
    counts = {'recorded': 0, 'dropped': 0, 'bytes': 0}
    def sample():
        try:
            while not stop.wait(interval) and os.getpid() == pid:
                now = time.monotonic_ns()
                native = {t.ident: t.native_id for t in threading.enumerate()}
                for ident, frame in sys._current_frames().items():
                    if ident == threading.get_ident():
                        continue
                    frames, truncated = _chain(frame)
                    del frame
                    data = {'version': 1, 'type': 'sample', 'process': str(identity['generation'] if identity else _generation),
                            'language': 'python', 'pid': pid, 'tid': native.get(ident, 0),
                            'session': os.environ.get('CUDAPROF_SESSION', ''),
                            'generation': identity['generation'] if identity else 0,
                            'time': now, 'clock': 'monotonic',
                            'status': 'clock_unavailable', 'provenance': 'wall_stack_snapshot',
                            'frames': [{'function': fn[:4096], 'file': file[:4096], 'line': line}
                                       for file, fn, line in frames], 'truncated': truncated}
                    encoded = (json.dumps(data, separators=(',', ':')) + '\n').encode()
                    if counts['bytes'] + len(encoded) > _limit - 512:
                        counts['dropped'] += 1
                        stop.set()
                        break
                    if os.write(fd, encoded) != len(encoded):
                        raise OSError('short stack sample write')
                    counts['bytes'] += len(encoded)
                    counts['recorded'] += 1
        except OSError as error:
            counts['dropped'] += 1
            _observer_failed(error)
        finally:
            coverage = {'version': 1, 'type': 'coverage', 'process': str(identity['generation'] if identity else _generation),
                        'language': 'python', 'pid': pid, 'session': os.environ.get('CUDAPROF_SESSION', ''),
                        'generation': identity['generation'] if identity else 0,
                        'clock': 'monotonic', 'status': 'clock_unavailable',
                        'provenance': 'wall_stack_snapshot', 'attempted': counts['recorded'] + counts['dropped'],
                        'recorded': counts['recorded'], 'dropped': counts['dropped'], 'truncated_count': 0, 'final': True}
            try:
                data = (json.dumps(coverage, separators=(',', ':')) + '\n').encode()
                if os.write(fd, data) != len(data):
                    raise OSError('short stack coverage write')
                with open(path + '.coverage.json', 'w') as out:
                    json.dump(coverage, out)
            except OSError as error:
                _observer_failed(error)
            finally:
                try:
                    os.close(fd)
                except OSError as error:
                    _observer_failed(error)
    worker = threading.Thread(target=sample, name='cudaprof-wall-stacks', daemon=True)
    worker.start()
    def finish():
        if os.getpid() != pid:
            return
        stop.set()
        worker.join()
    atexit.register(finish)
    return finish


def _finish():
    if not _enabled() or _writer is None:
        return
    identity = _identity()
    record = {'version': 1, 'type': 'coverage', 'process': str(identity['generation'] if identity else _generation),
              'pid': os.getpid(), 'session': os.environ.get('CUDAPROF_SESSION', ''),
              'generation': identity['generation'] if identity else 0, 'language': 'python',
              'attempted': _attempted, 'recorded': _recorded, 'dropped': _dropped,
              'truncated_count': _truncated, 'final': True}
    with _lock:
        # The normal byte cap reserves space for this terminal receipt.
        try:
            data = (json.dumps(record, separators=(',', ':')) + '\n').encode()
            if os.write(_writer, data) != len(data):
                raise OSError('short stack coverage write')
        except OSError as error:
            _observer_failed(error)


atexit.register(_finish)
