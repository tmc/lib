package kvm

// Capability is a KVM_CAP_* extension identifier.
type Capability int

// MachineType is a KVM_CREATE_VM machine type.
type MachineType uintptr

// DeviceType is a KVM_DEV_TYPE_* value.
type DeviceType uint32

// DeviceAttrGroup is a KVM_DEV_* attribute group.
type DeviceAttrGroup uint32

// MemoryFlag is a KVM_MEM_* userspace memory slot flag.
type MemoryFlag uint32

// UserspaceMemoryRegion is struct kvm_userspace_memory_region.
//
// Most callers should use VM.MapMemory instead. Direct use requires the caller to keep the memory at UserspaceAddr live for as long as KVM may access the slot.
type UserspaceMemoryRegion struct {
	Slot          uint32
	Flags         MemoryFlag
	GuestPhysAddr uint64
	MemorySize    uint64
	UserspaceAddr uint64
}

// CreateDevice is struct kvm_create_device.
type CreateDevice struct {
	Type  DeviceType
	FD    uint32
	Flags uint32
}

// DeviceAttr is struct kvm_device_attr.
type DeviceAttr struct {
	Flags uint32
	Group DeviceAttrGroup
	Attr  uint64
	Addr  uint64
}

// OneReg is struct kvm_one_reg.
type OneReg struct {
	ID   uint64
	Addr uint64
}

// IRQLevel is struct kvm_irq_level.
type IRQLevel struct {
	IRQ   uint32
	Level uint32
}

// VCPUInit is struct kvm_vcpu_init on arm64.
type VCPUInit struct {
	Target   uint32
	Features [7]uint32
}

// ExitReason is a KVM_EXIT_* reason.
type ExitReason uint32

// Exit describes the latest KVM userspace exit.
type Exit struct {
	Reason ExitReason
	// Raw is the mutable mapped kvm_run page. Its contents may change after the next Run call and become invalid after VCPU.Close.
	Raw []byte
}

const (
	// MachineDefault asks KVM to create the default VM type.
	MachineDefault MachineType = 0
)

const (
	CapIRQChip      Capability = 0
	CapHLT          Capability = 1
	CapUserMemory   Capability = 3
	CapSetTSSAddr   Capability = 4
	CapVCPUEvents   Capability = 41
	CapIRQFD        Capability = 32
	CapIOEventFD    Capability = 36
	CapOneReg       Capability = 70
	CapVMAttributes Capability = 101
)

const (
	DeviceARMVGICV3 DeviceType = 7
)

const (
	DeviceAttrARMVGICAddr       DeviceAttrGroup = 0
	DeviceAttrARMVGICDistRegs   DeviceAttrGroup = 1
	DeviceAttrARMVGICCtrl       DeviceAttrGroup = 4
	DeviceAttrARMVGICRedistRegs DeviceAttrGroup = 5
	DeviceAttrARMVGICCPUSysRegs DeviceAttrGroup = 6
)

const (
	ARMVGICCtrlInit         uint64 = 0
	ARMVGICV3AddrTypeDist   uint64 = 2
	ARMVGICV3AddrTypeRedist uint64 = 3
)

const (
	ARMVCPUFeaturePSCI02         uint32 = 2
	ARMVCPUFeaturePtrauthAddress uint32 = 5
	ARMVCPUFeaturePtrauthGeneric uint32 = 6
)

const (
	ARMIRQTypeCPU   uint32 = 0
	ARMIRQTypeSPI   uint32 = 1
	ARMIRQTypePPI   uint32 = 2
	ARMIRQTypeShift uint32 = 24
)

const (
	RegARMCore      uint64 = 0x1000_0000
	RegARM64        uint64 = 0x6000_0000_0000_0000
	RegARM64SysReg  uint64 = 0x1300_0000
	RegSizeU32      uint64 = 0x0020_0000_0000_0000
	RegSizeU64      uint64 = 0x0030_0000_0000_0000
	RegSizeU128     uint64 = 0x0040_0000_0000_0000
	RegARMTimerCNT  uint64 = RegARM64 | RegSizeU64 | RegARM64SysReg | (3 << 14) | (3 << 11) | (14 << 7) | (3 << 3) | 2
	RegARMTimerCVAL uint64 = RegARM64 | RegSizeU64 | RegARM64SysReg | (3 << 14) | (3 << 11) | (14 << 7) | (0 << 3) | 2
	RegMPIDREL1     uint64 = RegARM64 | RegSizeU64 | RegARM64SysReg | (3 << 14) | (0 << 11) | (0 << 7) | (0 << 3) | 5
)

const (
	MemLogDirtyPages MemoryFlag = 1 << 0
	MemReadonly      MemoryFlag = 1 << 1
	MemGuestMemFD    MemoryFlag = 1 << 2
)

const (
	ExitUnknown       ExitReason = 0
	ExitException     ExitReason = 1
	ExitIO            ExitReason = 2
	ExitHypercall     ExitReason = 3
	ExitDebug         ExitReason = 4
	ExitHLT           ExitReason = 5
	ExitMMIO          ExitReason = 6
	ExitIRQWindowOpen ExitReason = 7
	ExitShutdown      ExitReason = 8
	ExitFailEntry     ExitReason = 9
	ExitInternalError ExitReason = 17
	ExitSystemEvent   ExitReason = 24
	ExitMemoryFault   ExitReason = 39
)
