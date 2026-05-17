//go:build !linux

package kvm

// Open opens /dev/kvm.
func Open() (*Device, error) {
	return nil, ErrUnsupported
}

// OpenPath opens a KVM device node.
func OpenPath(string) (*Device, error) {
	return nil, ErrUnsupported
}

// Close closes the KVM device descriptor.
func (d *Device) Close() error {
	return ErrUnsupported
}

// APIVersion returns the kernel KVM API version.
func (d *Device) APIVersion() (int, error) {
	return 0, ErrUnsupported
}

// CheckExtension reports the value of a KVM_CAP_* extension.
func (d *Device) CheckExtension(Capability) (int, error) {
	return 0, ErrUnsupported
}

// CreateVM creates a VM.
func (d *Device) CreateVM(MachineType) (*VM, error) {
	return nil, ErrUnsupported
}

// MapMemory maps data as a userspace memory slot.
func (v *VM) MapMemory(uint32, uint64, []byte, MemoryFlag) error {
	return ErrUnsupported
}

// Close closes the VM descriptor.
func (v *VM) Close() error {
	return ErrUnsupported
}

// SetUserMemoryRegion sets or removes a userspace memory slot.
func (v *VM) SetUserMemoryRegion(*UserspaceMemoryRegion) error {
	return ErrUnsupported
}

// CreateDevice creates a VM device such as an ARM VGIC.
func (v *VM) CreateDevice(*CreateDevice) (*DeviceFD, error) {
	return nil, ErrUnsupported
}

// IRQLine sets the level of a guest interrupt line.
func (v *VM) IRQLine(*IRQLevel) error {
	return ErrUnsupported
}

// ARMPreferredTarget fills init with KVM's preferred ARM vCPU target.
func (v *VM) ARMPreferredTarget(*VCPUInit) error {
	return ErrUnsupported
}

// CreateVCPU creates a vCPU.
func (v *VM) CreateVCPU(int) (*VCPU, error) {
	return nil, ErrUnsupported
}

// ARMVCPUInit initializes an ARM vCPU with the supplied target and features.
func (v *VCPU) ARMVCPUInit(*VCPUInit) error {
	return ErrUnsupported
}

// GetOneReg reads one vCPU register into addr.
func (v *VCPU) GetOneReg(OneReg) error {
	return ErrUnsupported
}

// SetOneReg writes one vCPU register from addr.
func (v *VCPU) SetOneReg(OneReg) error {
	return ErrUnsupported
}

// Close closes the vCPU descriptor.
func (v *VCPU) Close() error {
	return ErrUnsupported
}

// Run enters the guest until KVM exits to userspace.
func (v *VCPU) Run() (Exit, error) {
	return Exit{}, ErrUnsupported
}

// Exit returns the latest exit metadata from the mapped kvm_run page.
func (v *VCPU) Exit() Exit {
	return Exit{}
}

// Close closes the KVM device descriptor.
func (d *DeviceFD) Close() error {
	return ErrUnsupported
}

// SetAttr sets an attribute on a KVM device descriptor.
func (d *DeviceFD) SetAttr(*DeviceAttr) error {
	return ErrUnsupported
}

// GetAttr gets an attribute from a KVM device descriptor.
func (d *DeviceFD) GetAttr(*DeviceAttr) error {
	return ErrUnsupported
}
