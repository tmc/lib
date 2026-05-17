//go:build linux

package kvm

import (
	"testing"
	"unsafe"
)

func TestIOCTLRequests(t *testing.T) {
	if got, want := requestGetAPIVersion, uintptr(0xAE00); got != want {
		t.Fatalf("KVM_GET_API_VERSION = %#x, want %#x", got, want)
	}
	if got, want := requestCreateVM, uintptr(0xAE01); got != want {
		t.Fatalf("KVM_CREATE_VM = %#x, want %#x", got, want)
	}
	if got, want := requestCreateVCPU, uintptr(0xAE41); got != want {
		t.Fatalf("KVM_CREATE_VCPU = %#x, want %#x", got, want)
	}
	for name, tc := range map[string]struct{ got, want uintptr }{
		"KVM_SET_USER_MEMORY_REGION": {requestSetUserMemoryRegion, 0x4020ae46},
		"KVM_CREATE_DEVICE":          {requestCreateDevice, 0xc00caee0},
		"KVM_SET_DEVICE_ATTR":        {requestSetDeviceAttr, 0x4018aee1},
		"KVM_GET_DEVICE_ATTR":        {requestGetDeviceAttr, 0x4018aee2},
		"KVM_ARM_PREFERRED_TARGET":   {requestARMPreferredTarget, 0x8020aeaf},
		"KVM_ARM_VCPU_INIT":          {requestARMVCPUInit, 0x4020aeae},
		"KVM_GET_ONE_REG":            {requestGetOneReg, 0x4010aeab},
		"KVM_SET_ONE_REG":            {requestSetOneReg, 0x4010aeac},
		"KVM_RUN":                    {requestRun, 0xae80},
		"KVM_IRQ_LINE":               {requestIRQLine, 0x4008ae61},
	} {
		if tc.got != tc.want {
			t.Fatalf("%s = %#x, want %#x", name, tc.got, tc.want)
		}
	}
}

func TestStructLayouts(t *testing.T) {
	for name, tc := range map[string]struct{ got, want uintptr }{
		"UserspaceMemoryRegion": {unsafe.Sizeof(UserspaceMemoryRegion{}), 32},
		"CreateDevice":          {unsafe.Sizeof(CreateDevice{}), 12},
		"DeviceAttr":            {unsafe.Sizeof(DeviceAttr{}), 24},
		"OneReg":                {unsafe.Sizeof(OneReg{}), 16},
		"IRQLevel":              {unsafe.Sizeof(IRQLevel{}), 8},
		"VCPUInit":              {unsafe.Sizeof(VCPUInit{}), 32},
	} {
		if tc.got != tc.want {
			t.Fatalf("%s size = %d, want %d", name, tc.got, tc.want)
		}
	}
}

func TestRunHeaderLayout(t *testing.T) {
	if got := unsafe.Offsetof(runHeader{}.ExitReason); got != 8 {
		t.Fatalf("exit_reason offset = %d, want 8", got)
	}
}
