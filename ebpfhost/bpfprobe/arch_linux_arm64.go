//go:build linux && arm64

package bpfprobe

// Linux UAPI struct user_pt_regs contains regs[31], sp, pc and pstate.
// AAPCS64 passes eight integer arguments in x0..x7 and the rest at sp.
const (
	sysPerfEventOpen         = 241
	returnRegisterOffset     = 0
	stackPointerOffset       = 248
	instructionPointerOffset = 256
	stackArgumentOffset      = 0
)

var argumentRegisterOffsets = []int16{0, 8, 16, 24, 32, 40, 48, 56}
