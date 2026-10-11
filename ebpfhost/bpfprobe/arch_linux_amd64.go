//go:build linux && amd64

package bpfprobe

const (
	sysPerfEventOpen         = 298
	returnRegisterOffset     = 80
	stackPointerOffset       = 152
	instructionPointerOffset = 128
	stackArgumentOffset      = 8
)

var argumentRegisterOffsets = []int16{112, 104, 96, 88, 72, 64}
