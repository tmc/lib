# Test Inventory: nbd

## Cheap Gates

- Plan schema validation.
- Generated file surface assertions.
- Protocol request frame round-trip.
- Magic constant golden bytes.
- No-zeroes handshake compliance.
- Oversized request rejection before allocation.
- Large client I/O chunking at the protocol payload limit.
- Server offset and length bounds validation before device calls.
- Concurrent Client.ReadAt under the race detector.
- Serve context cancellation on idle connections.
- Client/server read, write, trim, disconnect over net.Pipe and MemoryDevice.
- Non-Linux OpenKernelDevice stub returns ErrUnsupported.
- Linux ioctl request numbers cross-compile in generated tests.

## Manual Gates

- `kernel`: set NBD_TEST_KERNEL=1 and run as root on a host with the nbd module loaded.
- Optional `NBD_TEST_DEVICE=/dev/nbd0` selects the kernel device node.
