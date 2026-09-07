import pathlib
import sys
import numpy as np
import jax
import ml_dtypes
from jax.extend import backend

jax.config.update("jax_enable_x64", True)
client = backend.get_backend("cpu")
device = client.local_devices()[0]
root = pathlib.Path(sys.argv[1])


def execute(name, args):
    executable = client.compile_and_load((root / (name + ".mlir")).read_text(), [device])
    return [np.asarray(x) for x in executable.execute([jax.device_put(a, device) for a in args])]


def rotary(x, cosine, sine):
    # Reference one pair at a time, independent of graph slices/concatenation.
    out = x.astype(np.float32).copy()
    r = cosine.shape[-1]
    cosine = np.broadcast_to(cosine, x.shape[:-1] + (r,))
    sine = np.broadcast_to(sine, cosine.shape)
    for coord in np.ndindex(x.shape[:-1]):
        for i in range(r):
            a, b = np.float32(x[coord+(i,)]), np.float32(x[coord+(i+r,)])
            c, s = cosine[coord+(i,)], sine[coord+(i,)]
            out[coord+(i,)] = a*c-b*s
            out[coord+(i+r,)] = a*s+b*c
    return out


for name, dtype in (("f32", np.float32), ("f16", np.float16), ("bf16", ml_dtypes.bfloat16)):
    rng = np.random.default_rng(8713)
    x = rng.normal(size=(2, 4, 3, 6)).astype(dtype)
    x.flat[:7] = [-80, -20, -4, 0, 4, 20, 80]
    up = rng.normal(size=x.shape).astype(dtype)
    angles = np.arange(12, dtype=np.float32).reshape(2, 1, 3, 2) * np.float32(.17)
    cosine, sine = np.cos(angles), np.sin(angles)
    got = execute(name, [x, up, cosine, sine])
    # Use float64 direct formula as a reference, not the graph's sign split.
    work = x.astype(np.float64)
    activation = work / (1 + np.exp(-work))
    wants = [activation.astype(dtype), (activation * up.astype(np.float64)).astype(dtype), rotary(x, cosine, sine).astype(dtype)]
    rtol = .004 if name == "bf16" else .0005 if name == "f16" else 2e-6
    for i, (actual, want) in enumerate(zip(got, wants)):
        np.testing.assert_allclose(actual.astype(np.float32), want.astype(np.float32), rtol=rtol, atol=1e-7 if i == 2 else 0)
    # Partial rotation preserves tail features exactly, including stored precision.
    np.testing.assert_array_equal(got[2][..., 4:], x[..., 4:])
    print(name, "SiLU, SwiGLU and partial rotary match references")

for name in ("gather", "gather_scalar", "gather_vector"):
    operand = np.arange(4 if name == "gather_vector" else 24, dtype=np.float32)
    if name != "gather_vector":
        operand = operand.reshape(4, 2, 3)
    if name == "gather_scalar":
        indices = np.int64(2)
    else:
        indices = np.array([[3, 0, 3], [-2**31, 2, 2**31-1]], dtype=np.int32) if name == "gather_vector" else np.array([[3, 0, 3], [-2**63, 2, 2**63-1]], dtype=np.int64)
    got = execute(name, [operand, indices])[0]
    np.testing.assert_array_equal(got, operand[np.clip(indices, 0, 3)])
    print(name, "row selection and clamping match reference")

x = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4) / 8
angles = np.arange(6, dtype=np.float32).reshape(3, 2) / 3
cosine, sine = np.cos(angles), np.sin(angles)
got = execute("rotary_full", [x, cosine, sine])[0]
np.testing.assert_allclose(got, rotary(x, cosine, sine), rtol=2e-6, atol=1e-7)
print("rotary_full broadcasts sequence tables across batch and heads")
scalar = np.float32(3.25)
np.testing.assert_array_equal(execute("slice_scalar", [scalar])[0], scalar)
x = np.arange(36, dtype=np.float32).reshape(2, 3, 6)
part = x[:, ::2, 1::2]
np.testing.assert_array_equal(execute("slice_concat", [x])[0], np.concatenate([part, part], axis=1))
print("scalar slice, strided slice and concatenate match references")
print("verified 9 model operation programs")
