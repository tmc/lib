//go:build linux

package kvm

import (
	"fmt"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

const defaultPath = "/dev/kvm"

// Open opens /dev/kvm.
func Open() (*Device, error) {
	return OpenPath(defaultPath)
}

// OpenPath opens a KVM device node.
func OpenPath(path string) (*Device, error) {
	f, err := os.OpenFile(path, os.O_RDWR|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("kvm open %s: %w", path, err)
	}
	return &Device{file: f}, nil
}

// Close closes the KVM device descriptor.
func (d *Device) Close() error {
	if d == nil || d.file == nil {
		return ErrClosed
	}
	f := d.file
	d.file = nil
	if err := f.Close(); err != nil {
		return fmt.Errorf("kvm close device: %w", err)
	}
	return nil
}

// APIVersion returns the kernel KVM API version.
func (d *Device) APIVersion() (int, error) {
	if d == nil || d.file == nil {
		return 0, ErrClosed
	}
	n, err := ioctlRet(int(d.file.Fd()), requestGetAPIVersion, 0)
	if err != nil {
		return 0, fmt.Errorf("kvm get api version: %w", err)
	}
	return n, nil
}

// CheckExtension reports the value of a KVM_CAP_* extension.
func (d *Device) CheckExtension(cap Capability) (int, error) {
	if d == nil || d.file == nil {
		return 0, ErrClosed
	}
	n, err := ioctlRet(int(d.file.Fd()), requestCheckExtension, uintptr(cap))
	if err != nil {
		return 0, fmt.Errorf("kvm check extension %d: %w", cap, err)
	}
	return n, nil
}

// CreateVM creates a VM.
func (d *Device) CreateVM(machineType MachineType) (*VM, error) {
	if d == nil || d.file == nil {
		return nil, ErrClosed
	}
	fd, err := ioctlRet(int(d.file.Fd()), requestCreateVM, uintptr(machineType))
	if err != nil {
		return nil, fmt.Errorf("kvm create vm: %w", err)
	}
	size, err := ioctlRet(int(d.file.Fd()), requestGetVCPUMMapSize, 0)
	if err != nil {
		_ = syscall.Close(fd)
		return nil, fmt.Errorf("kvm get vcpu mmap size: %w", err)
	}
	return &VM{
		file:     os.NewFile(uintptr(fd), "kvm-vm"),
		mmapSize: size,
		regions:  make(map[uint32][]byte),
		pins:     make(map[uint32]*runtime.Pinner),
	}, nil
}

// Close closes the VM descriptor.
func (v *VM) Close() error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	f := v.file
	v.file = nil
	v.mu.Lock()
	for _, pin := range v.pins {
		pin.Unpin()
	}
	v.regions = nil
	v.pins = nil
	v.mu.Unlock()
	if err := f.Close(); err != nil {
		return fmt.Errorf("kvm close vm: %w", err)
	}
	return nil
}

// MapMemory maps data as a userspace memory slot and keeps it live until the VM is closed or the slot is replaced. Passing an empty slice removes the slot. Callers must not reuse, resize, or discard the mapped slice backing storage while KVM can access the slot.
func (v *VM) MapMemory(slot uint32, guestPhysAddr uint64, data []byte, flags MemoryFlag) error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	region := UserspaceMemoryRegion{Slot: slot, Flags: flags, GuestPhysAddr: guestPhysAddr}
	var pin *runtime.Pinner
	if len(data) != 0 {
		pin = new(runtime.Pinner)
		pin.Pin(&data[0])
		region.MemorySize = uint64(len(data))
		region.UserspaceAddr = uint64(uintptr(unsafe.Pointer(&data[0])))
	}
	if err := v.SetUserMemoryRegion(&region); err != nil {
		if pin != nil {
			pin.Unpin()
		}
		return err
	}
	if len(data) == 0 {
		v.mu.Lock()
		if old := v.pins[slot]; old != nil {
			old.Unpin()
		}
		delete(v.regions, slot)
		delete(v.pins, slot)
		v.mu.Unlock()
		return nil
	}
	v.mu.Lock()
	if v.regions == nil {
		v.regions = make(map[uint32][]byte)
	}
	if v.pins == nil {
		v.pins = make(map[uint32]*runtime.Pinner)
	}
	if old := v.pins[slot]; old != nil {
		old.Unpin()
	}
	v.regions[slot] = data
	v.pins[slot] = pin
	v.mu.Unlock()
	return nil
}

// SetUserMemoryRegion sets or removes a userspace memory slot.
func (v *VM) SetUserMemoryRegion(region *UserspaceMemoryRegion) error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	if err := ioctl(int(v.file.Fd()), requestSetUserMemoryRegion, uintptr(unsafe.Pointer(region))); err != nil {
		return fmt.Errorf("kvm set user memory region: %w", err)
	}
	return nil
}

// CreateDevice creates a VM device such as an ARM VGIC.
func (v *VM) CreateDevice(args *CreateDevice) (*DeviceFD, error) {
	if v == nil || v.file == nil {
		return nil, ErrClosed
	}
	if args == nil {
		return nil, ErrInvalidArgument
	}
	if err := ioctl(int(v.file.Fd()), requestCreateDevice, uintptr(unsafe.Pointer(args))); err != nil {
		return nil, fmt.Errorf("kvm create device: %w", err)
	}
	return &DeviceFD{file: os.NewFile(uintptr(args.FD), "kvm-device")}, nil
}

// IRQLine sets the level of a guest interrupt line.
func (v *VM) IRQLine(level *IRQLevel) error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	if level == nil {
		return ErrInvalidArgument
	}
	if err := ioctl(int(v.file.Fd()), requestIRQLine, uintptr(unsafe.Pointer(level))); err != nil {
		return fmt.Errorf("kvm irq line: %w", err)
	}
	return nil
}

// CreateVCPU creates a vCPU and maps its shared kvm_run page.
func (v *VM) CreateVCPU(id int) (*VCPU, error) {
	if v == nil || v.file == nil {
		return nil, ErrClosed
	}
	fd, err := ioctlRet(int(v.file.Fd()), requestCreateVCPU, uintptr(id))
	if err != nil {
		return nil, fmt.Errorf("kvm create vcpu %d: %w", id, err)
	}
	run, err := syscall.Mmap(fd, 0, v.mmapSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		_ = syscall.Close(fd)
		return nil, fmt.Errorf("kvm mmap vcpu run: %w", err)
	}
	return &VCPU{file: os.NewFile(uintptr(fd), "kvm-vcpu"), run: run}, nil
}

// ARMPreferredTarget fills init with KVM's preferred ARM vCPU target.
func (v *VM) ARMPreferredTarget(init *VCPUInit) error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	if init == nil {
		return ErrInvalidArgument
	}
	if err := ioctl(int(v.file.Fd()), requestARMPreferredTarget, uintptr(unsafe.Pointer(init))); err != nil {
		return fmt.Errorf("kvm arm preferred target: %w", err)
	}
	return nil
}

// ARMVCPUInit initializes an ARM vCPU with the supplied target and features.
func (v *VCPU) ARMVCPUInit(init *VCPUInit) error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	if init == nil {
		return ErrInvalidArgument
	}
	if err := ioctl(int(v.file.Fd()), requestARMVCPUInit, uintptr(unsafe.Pointer(init))); err != nil {
		return fmt.Errorf("kvm arm vcpu init: %w", err)
	}
	return nil
}

// GetOneReg reads one vCPU register into addr.
func (v *VCPU) GetOneReg(reg OneReg) error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	if err := ioctl(int(v.file.Fd()), requestGetOneReg, uintptr(unsafe.Pointer(&reg))); err != nil {
		return fmt.Errorf("kvm get one reg %#x: %w", reg.ID, err)
	}
	return nil
}

// SetOneReg writes one vCPU register from addr.
func (v *VCPU) SetOneReg(reg OneReg) error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	if err := ioctl(int(v.file.Fd()), requestSetOneReg, uintptr(unsafe.Pointer(&reg))); err != nil {
		return fmt.Errorf("kvm set one reg %#x: %w", reg.ID, err)
	}
	return nil
}

// Close unmaps the kvm_run page and closes the vCPU descriptor.
func (v *VCPU) Close() error {
	if v == nil || v.file == nil {
		return ErrClosed
	}
	var err error
	if v.run != nil {
		if e := syscall.Munmap(v.run); e != nil {
			err = fmt.Errorf("kvm munmap vcpu run: %w", e)
		}
		v.run = nil
	}
	f := v.file
	v.file = nil
	if e := f.Close(); e != nil && err == nil {
		err = fmt.Errorf("kvm close vcpu: %w", e)
	}
	return err
}

// Run enters the guest until KVM exits to userspace.
func (v *VCPU) Run() (Exit, error) {
	if v == nil || v.file == nil {
		return Exit{}, ErrClosed
	}
	if err := ioctl(int(v.file.Fd()), requestRun, 0); err != nil {
		return Exit{}, fmt.Errorf("kvm run vcpu: %w", err)
	}
	return v.Exit(), nil
}

// Exit returns the latest exit metadata from the mapped kvm_run page.
func (v *VCPU) Exit() Exit {
	if v == nil || len(v.run) < runHeaderSize {
		return Exit{}
	}
	h := (*runHeader)(unsafe.Pointer(&v.run[0]))
	return Exit{Reason: ExitReason(h.ExitReason), Raw: v.run}
}

// Close closes the KVM device descriptor.
func (d *DeviceFD) Close() error {
	if d == nil || d.file == nil {
		return ErrClosed
	}
	f := d.file
	d.file = nil
	if err := f.Close(); err != nil {
		return fmt.Errorf("kvm close device fd: %w", err)
	}
	return nil
}

// SetAttr sets an attribute on a KVM device descriptor.
func (d *DeviceFD) SetAttr(attr *DeviceAttr) error {
	if d == nil || d.file == nil {
		return ErrClosed
	}
	if attr == nil {
		return ErrInvalidArgument
	}
	if err := ioctl(int(d.file.Fd()), requestSetDeviceAttr, uintptr(unsafe.Pointer(attr))); err != nil {
		return fmt.Errorf("kvm set device attr: %w", err)
	}
	return nil
}

// GetAttr gets an attribute from a KVM device descriptor.
func (d *DeviceFD) GetAttr(attr *DeviceAttr) error {
	if d == nil || d.file == nil {
		return ErrClosed
	}
	if attr == nil {
		return ErrInvalidArgument
	}
	if err := ioctl(int(d.file.Fd()), requestGetDeviceAttr, uintptr(unsafe.Pointer(attr))); err != nil {
		return fmt.Errorf("kvm get device attr: %w", err)
	}
	return nil
}
