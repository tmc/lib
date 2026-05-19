# Package Spec: virtiofs

Package virtiofs provides pure-Go virtio-fs request helpers over the FUSE protocol.

- Import path: `github.com/tmc/lib/virtiofs`
- No cgo and no virtiofsd/libfuse ABI binding.
- Depends on `github.com/tmc/lib/fuse` for FUSE frames and filesystem semantics.
- Semantic references only: Linux virtio-fs UAPI, virtiofsd, QEMU virtio-fs, libfuse, and the generated fuse package.

## Public API

- `type Config` with Tag, NumQueues, and QueueSize.
- `type Transport interface` exchanges encoded FUSE request and response bytes.
- `type Device` performs typed FUSE frame round trips over a Transport.
- `type Server` handles encoded FUSE requests against a `fuse.FileSystem`.
- `type MemoryTransport` is the deterministic fake transport for tests.

## Manual Gates

VM gates require `VIRTIOFS_TEST_VM=1` and an external VMM-backed `RunVMGate` hook.
