# Manual Gates: kvm

## hardware

Opens /dev/kvm and runs host KVM capability or VM/vCPU smoke checks. This gate is never run by default.

- Environment: `KVM_TEST_HARDWARE=1`
- Command: `KVM_TEST_HARDWARE=1 go test -run '^TestIntegration' -count=1 ./kvm`
- Generated helper: `RequireHardwareGate(t *testing.T)`
- Generated smoke: the default UAPI gate opens the device and checks API version `12`.
- Harness hook: set `RunHardwareGate` from a handwritten external test to replace the generated smoke with a deeper hardware check.
- Skip reason: set KVM_TEST_HARDWARE=1 to open /dev/kvm and run hardware integration tests
