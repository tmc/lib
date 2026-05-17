# Package Spec: nbd

Package nbd provides a pure-Go Network Block Device protocol and Linux device helper.

- Import path: `github.com/tmc/lib/nbd`
- No cgo and no libnbd ABI binding.
- Semantic references only: libnbd, nbdkit, qemu-nbd, and github.com/Merovius/nbd.

## Public API

- `type Device interface { io.ReaderAt; io.WriterAt; Size() uint64 }`
- `type Syncer interface { Sync() error }`
- `type Trimmer interface { Trim(off, length uint64) error }`
- `type Export struct { Name string; Device Device; ReadOnly bool }`
- `func Serve(ctx context.Context, rw io.ReadWriter, exp Export) error`
- `func Dial(ctx context.Context, rw io.ReadWriter, exportName string) (*Client, error)`
- Linux only: `OpenKernelDevice` and small /dev/nbdX ioctl helpers.

## Protocol Surface

- Newstyle fixed handshake with `NBD_OPT_EXPORT_NAME`.
- Simple replies.
- Read, write, disconnect, flush, and trim request handling.
- Cheap tests exercise request framing and client/server behavior over `net.Pipe`.

## Manual Gates

Kernel-device gates require `NBD_TEST_KERNEL=1`, root privileges, a loaded `nbd` module, and an unused `/dev/nbdX`.
