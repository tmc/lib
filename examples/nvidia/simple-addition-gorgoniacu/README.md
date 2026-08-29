# simple-addition-gorgoniacu

Vector addition on the GPU with [gorgonia.org/cu](https://pkg.go.dev/gorgonia.org/cu),
the most widely used third-party CUDA Driver API binding for Go. It is a
line-for-line counterpart of `../simple-addition`, which uses the purego-based
bindings in `github.com/tmc/lib/nvidia`.

`cu` is cgo: `cuda.h` and `nvrtc.h` must be on the include path and
`libcuda`/`libnvrtc` on the link path, so this example is a separate module and
is excluded from the `github.com/tmc/lib` build.

    go run .

Notable differences from the purego bindings:

  - Sizes and dimensions are `int`/`int64` rather than the C-exact `uint`/`uint32`.
  - Kernel arguments are a `[]unsafe.Pointer`, not a pointer to an array.
  - `nvrtc.Program.GetPTX` returns a `string`; `cu.LoadData` NUL-terminates it.
  - Contexts are thread-affine, and `CUContext.Lock` bundles
    `runtime.LockOSThread` with making the context current.
