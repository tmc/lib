# Session Report: kvm

Status: artifact-only package plan rendered.

## Semantic Sources

- `Documentation/virt/kvm/api.rst` (docs): KVM userspace API lifecycle and ioctl semantics
- KVM has no dynamic library loader; ioctl requests are kernel UAPI calls.
- The generated surface includes VGICv3, ARM vCPU init, one-reg, device-attr, and IRQ-line ioctls.
- Default tests must not open /dev/kvm.
- Hardware tests require KVM_TEST_HARDWARE=1 and a handwritten harness hook.

## Default Gates

- Validate package plan before rendering.
- Run generated cheap tests only by default.
- Treat hardware tests as manual gates.

## Completion Rule

Do not call this package complete until artifacts, NotebookLM review, local non-hardware gates, and manual gate notes agree.
