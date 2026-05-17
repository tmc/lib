# Manual Gates: nbd

## kernel

Opens /dev/nbdX and exercises Linux kernel NBD ioctls. This gate is never run by default.

- Environment: `NBD_TEST_KERNEL=1`
- Optional device: `NBD_TEST_DEVICE=/dev/nbd0`
- Command: `NBD_TEST_KERNEL=1 go test -run '^TestIntegration' -count=1 ./nbd`
- Generated helper: `RequireKernelGate(t *testing.T)`
- Generated smoke: opens the selected /dev/nbdX node and verifies a valid file descriptor.
- Harness hook: set `RunKernelGate` from a handwritten external test to replace the generated smoke with a deeper root/kernel check.
- Skip reason: set NBD_TEST_KERNEL=1 and run as root on a host with the nbd module loaded
