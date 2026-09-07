import pathlib
import sys
import numpy as np
import jax
import ml_dtypes
from jax.extend import backend
# Preserve the i64 runtime indices consumed by the compiled primitives.
jax.config.update("jax_enable_x64", True)
client = backend.get_backend('cpu')
device = client.local_devices()[0]
root = pathlib.Path(sys.argv[1])
for name in ('causal_dynamic', 'update', 'matmul', 'rmsnorm', 'tiny', 'softmax', 'attention', 'attention_f16', 'attention_bf16', 'attention_masked', 'attention_masked_f16', 'attention_masked_bf16'):
    executable = client.compile_and_load((root / (name + '.mlir')).read_text(), [device])
    if name == 'causal_dynamic':
        for offset in (-(2**63), -1, 0, 1, 2**63-1):
            result = executable.execute([jax.device_put(np.int64(offset), device)])
            want = np.array([[k-q <= offset for k in range(3)] for q in range(2)], dtype=np.bool_)
            np.testing.assert_array_equal(np.asarray(result[0]), want)
        print(name, 'CPU runtime offsets including int64 extremes match reference')
        continue
    if name == 'update':
        for start in (-1, 1, 99):
            operand = np.array([1,2,3,4], dtype=np.int64)
            update = np.array([9,8], dtype=np.int64)
            args = [operand,update,np.int64(start)]
            result = executable.execute([jax.device_put(a, device) for a in args])
            want = operand.copy()
            adjusted = min(max(start,0),2)
            want[adjusted:adjusted+2] = update
            np.testing.assert_array_equal(np.asarray(result[0]), want)
        print(name, 'CPU update and clamp semantics match reference')
        continue
    if name == 'matmul':
        args = [np.arange(6, dtype=np.float32).reshape(2,3), np.arange(12, dtype=np.float32).reshape(3,4)]
        want = args[0] @ args[1]
    elif name == 'rmsnorm':
        args = [np.array([[1,2,3],[4,5,6]], dtype=np.float32), np.array([1,2,3], dtype=np.float32)]
        want = args[0] / np.sqrt(np.mean(args[0]**2, axis=1, keepdims=True) + np.float32(1e-8)) * args[1]
    elif name == 'softmax':
        args = [np.array([[1000,1001,1002],[-1000,-1001,-1002]], dtype=np.float32)]
        exp = np.exp(args[0] - args[0].max(axis=-1, keepdims=True))
        want = exp / exp.sum(axis=-1, keepdims=True)
    elif name.startswith('attention'):
        rng = np.random.default_rng(911271)
        args = [rng.normal(size=shape).astype(np.float32) for shape in ((2,3,4),(2,5,4),(2,5,6))]
        if name.endswith(('_f16', '_bf16')):
            dtype = np.float16 if name.endswith('_f16') else ml_dtypes.bfloat16
            args = [a.astype(dtype) for a in args]
        work = [a.astype(np.float32) for a in args]
        scores = (work[0] @ work[1].swapaxes(-1,-2)) / np.float32(2)
        exp = np.exp(scores-scores.max(axis=-1,keepdims=True))
        want = (exp/exp.sum(axis=-1,keepdims=True)) @ work[2]
        if 'masked' in name:
            padding = np.array([[[1,1,0,0,0]], [[0,0,0,0,0]]], dtype=np.bool_)
            args.append(padding)
            want = np.zeros((2,3,6), dtype=np.float32)
            # Gather eligible keys, independently of the emitter's select/
            # reduction decomposition. Empty rows retain their zero result.
            for batch in range(2):
                for query in range(3):
                    keys = [k for k in range(5) if k <= query and padding[batch,0,k]]
                    if keys:
                        row = scores[batch,query,keys]
                        weights = np.exp(row-row.max())
                        want[batch,query] = (weights/weights.sum()) @ work[2][batch,keys]
    else:
        args = []
        want = np.float32(1e-8)
    inputs = [jax.device_put(a, device) for a in args]
    result = executable.execute(inputs)
    got = np.asarray(result[0])
    # Compare against f32 arithmetic on the quantized input, allowing one
    # reduced-precision rounding step at the output.
    rtol = 0.004 if name.endswith('_bf16') else 0.0005 if name.endswith('_f16') else 1e-6
    np.testing.assert_allclose(got.astype(np.float32), want, rtol=rtol, atol=1e-7 if name.endswith(("_f16", "_bf16")) else 0)
    print(name, 'CPU execution matches numpy')

print('verified 12 CPU programs')
