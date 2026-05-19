# Manual Gates: fuse

## kernel

Opens /dev/fuse and exercises Linux kernel FUSE setup. This gate is never run by default.

- Environment: `FUSE_TEST_KERNEL=1`
- Optional device: `FUSE_TEST_DEVICE=/dev/fuse`
- Command: `FUSE_TEST_KERNEL=1 go test -run '^TestIntegration' -count=1 ./fuse`
- Generated helper: `RequireKernelGate(t *testing.T)`
- Generated smoke: opens the selected /dev/fuse node and verifies a valid file descriptor.
- Harness hook: set `RunKernelGate` from a handwritten external test to replace the generated smoke with a mount lifecycle check.
- Skip reason: set FUSE_TEST_KERNEL=1 and run on a host with /dev/fuse access
