# Session Report: nbd

Status: pure-Go protocol/UAPI package rendered.

## Semantic Sources

- libnbd: client lifecycle, option negotiation, and error semantics.
- nbdkit: server export behavior and plugin boundary.
- qemu-nbd: interoperability behavior for common request types.
- github.com/Merovius/nbd: pure-Go API shape around io.ReaderAt/io.WriterAt devices.

## Boundaries

- No cgo.
- No libnbd ABI binding.
- C and C-adjacent projects are semantic references only.
- Linux kernel device checks are manual gates, not default tests.

## Completion Rule

Do not call this package complete until NotebookLM review and local non-hardware gates agree. Kernel /dev/nbdX gates remain explicit manual gates.
