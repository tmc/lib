# Package Spec: kvm

Package kvm provides Go bindings for the Linux KVM userspace API.

- Import path: `github.com/tmc/lib/kvm`
- ABI package: `internal/uapi`
- Public API note: KVM is a Linux UAPI ioctl interface, not a shared C library. The public package is generated from semantic UAPI intent while hardware use remains manual-gated.

## Parse Inputs

- Headers: `include/uapi/linux/kvm.h`

## Semantic Sources

- `Documentation/virt/kvm/api.rst` (docs): KVM userspace API lifecycle and ioctl semantics
- KVM has no dynamic library loader; ioctl requests are kernel UAPI calls.
- The generated surface includes VGICv3, ARM vCPU init, one-reg, device-attr, and IRQ-line ioctls.
- Default tests must not open /dev/kvm.
- Hardware tests require KVM_TEST_HARDWARE=1 and a handwritten harness hook.

## Linux UAPI

- Device path: `/dev/kvm`
- IO type: `0xAE`
- Resources: `4`
- Requests: `15`
- Linux methods: `22`

## Resources

- `Device` wraps `fd` and is owned.
- `VM` wraps `fd` and is owned.
- `VCPU` wraps `fd` and is owned.

## Operations

- `Open`: Open opens /dev/kvm. calls backend `open`.
- `APIVersion`: APIVersion returns the kernel KVM API version. calls backend `apiVersion`.
- `CheckExtension`: CheckExtension reports the value of a KVM_CAP_* extension. calls backend `checkExtension`.
- `CreateVM`: CreateVM creates a VM. calls backend `createVM`.
- `MapMemory`: MapMemory maps data as a userspace memory slot and keeps it live until the VM is closed or the slot is replaced. calls backend `mapMemory`.
- `SetUserMemoryRegion`: SetUserMemoryRegion sets or removes a userspace memory slot. calls backend `setUserMemoryRegion`.
- `CreateDevice`: CreateDevice creates a VM device such as an ARM VGIC. calls backend `createDevice`.
- `IRQLine`: IRQLine sets the level of a guest interrupt line. calls backend `irqLine`.
- `ARMPreferredTarget`: ARMPreferredTarget fills init with KVM's preferred ARM vCPU target. calls backend `armPreferredTarget`.
- `CreateVCPU`: CreateVCPU creates a vCPU and maps its shared kvm_run page. calls backend `createVCPU`.
- `ARMVCPUInit`: ARMVCPUInit initializes an ARM vCPU with the supplied target and features. calls backend `armVCPUInit`.
- `GetOneReg`: GetOneReg reads one vCPU register into addr. calls backend `getOneReg`.
- `SetOneReg`: SetOneReg writes one vCPU register from addr. calls backend `setOneReg`.
- `Run`: Run enters the guest until KVM exits to userspace. calls backend `run`; blocking, context policy `not-supported`; manual gate `hardware`.
- `SetAttr`: SetAttr sets an attribute on a KVM device descriptor. calls backend `setAttr`.
- `GetAttr`: GetAttr gets an attribute from a KVM device descriptor. calls backend `getAttr`.

## Adapters

No real-provider adapters declared.

## Unsafe Policy

No public unsafe exposure is allowed.

## Element Sets

No element sets declared.
