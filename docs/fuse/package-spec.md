# Package Spec: fuse

Package fuse provides a pure-Go FUSE protocol and Linux device helper.

- Import path: `github.com/tmc/lib/fuse`
- No cgo and no libfuse ABI binding.
- Semantic references only: libfuse, go-fuse, bazil.org/fuse, virtiofsd, and QEMU virtio-fs.

## Public API

- `type FileSystem interface` with Lookup, Getattr, Setattr, Open, Read, Write, Readdir, Flush, Fsync, Release.
- `type Frame`, `InHeader`, `OutHeader`, `Attr`, `Entry`, and `Dirent` model the low-level FUSE protocol.
- `func Serve(ctx context.Context, rw io.ReadWriter, fs FileSystem) error`
- `func HandleFrame(ctx context.Context, fs FileSystem, frame Frame) ([]byte, error)`
- Linux only: `OpenDevice` for /dev/fuse.
- `MemoryFS` is bounded to 1 GiB per file to keep test helpers from turning hostile offsets into large allocations.

## Manual Gates

Kernel-device gates require `FUSE_TEST_KERNEL=1` and /dev/fuse access.
