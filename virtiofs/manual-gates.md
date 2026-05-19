# Manual Gates: virtiofs

## vm

Exercises virtio-fs through a real VMM/guest transport. This gate is never run by default.

- Environment: `VIRTIOFS_TEST_VM=1`
- Command: `VIRTIOFS_TEST_VM=1 go test -run '^TestIntegration' -count=1 ./virtiofs`
- Generated helper: `RequireVMGate(t *testing.T)`
- Harness hook: set `RunVMGate` from a handwritten external test.
- Fail-closed behavior: with `VIRTIOFS_TEST_VM=1` and no hook, the generated test fails with `manual integration body is not generated`.
- Skip reason: set VIRTIOFS_TEST_VM=1 and provide a VM-backed RunVMGate hook.
