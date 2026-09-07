import pathlib
import sys
import numpy as np
import jax
import ml_dtypes
from jax.extend import backend
client = backend.get_backend('cpu')
device = client.local_devices()[0]
root = pathlib.Path(sys.argv[1])
for name in ('matmul', 'rmsnorm', 'tiny', 'softmax', 'attention', 'attention_f16', 'attention_bf16'):
    executable = client.compile_and_load((root / (name + '.mlir')).read_text(), [device])
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
        if name != 'attention':
            dtype = np.float16 if name.endswith('_f16') else ml_dtypes.bfloat16
            args = [a.astype(dtype) for a in args]
        work = [a.astype(np.float32) for a in args]
        scores = (work[0] @ work[1].swapaxes(-1,-2)) / np.float32(2)
        exp = np.exp(scores-scores.max(axis=-1,keepdims=True))
        want = (exp/exp.sum(axis=-1,keepdims=True)) @ work[2]
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

print('verified 7 CPU programs')
