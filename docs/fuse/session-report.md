# Session Report: fuse

Status: pure-Go protocol/UAPI package rendered.

## Boundaries

- No cgo.
- No libfuse ABI binding.
- libfuse, go-fuse, bazil.org/fuse, virtiofsd, and QEMU are semantic references only.
- /dev/fuse checks remain explicit manual gates.
