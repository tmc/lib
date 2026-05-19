# Test Inventory: fuse

## Cheap Gates

- Plan schema validation.
- Generated file surface assertions.
- Frame and response round-trip tests.
- In-memory filesystem lookup/read/write tests.
- In-memory filesystem rejects hostile growth offsets.
- Server request/response over net.Pipe.
- Non-Linux OpenDevice stub returns ErrUnsupported.
- Linux OpenDevice error-path compile test.

## Manual Gates

- `kernel`: set FUSE_TEST_KERNEL=1 and run on a host with /dev/fuse access.
