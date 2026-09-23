package cudabpfdev

// Op identifies a version-1 device instruction.
type Op uint8

const (
	Const       Op = iota + 1 // Dst = Imm.
	Move                      // Dst = A.
	Add                       // Dst = A + B modulo 2^32.
	And                       // Dst = A & B.
	Lane                      // Dst = hardware lane ID, a varying value.
	Ballot                    // Dst = active mask of lanes whose A is nonzero.
	Leader                    // Dst = A from the lowest active lane, broadcast.
	JumpNonzero               // Branch to Target if A != 0, otherwise fall through.
	Repeat                    // Repeat [Target, PC] Imm times in total.
	MapAdd                    // Add B to map slot A once per active warp.
	Exit                      // Return A; no side effect.
)

// Instruction encodes one operation. Registers are indexed by Dst, A and B;
// unused fields must be zero. Target is an absolute instruction index. Imm is
// an unsigned constant or a bounded loop count. Unknown opcodes are rejected.
type Instruction struct {
	Op        Op
	Dst, A, B uint8
	Imm       uint32
	Target    int
}

// Program is the versioned device contract. Registers is in [1,32]. Slots is
// the fixed map capacity; a runtime key outside it must not write memory.
// StagingBytes is requested shared storage, not an inferred hardware limit.
// Code must not be mutated concurrently with verification or evaluation.
type Program struct {
	Version      uint16
	Registers    uint8
	Slots        uint32
	StagingBytes uint32
	Code         []Instruction
}

// Budget bounds verifier work and the declared instrumentation resources.
// MaxSteps bounds dynamic bytecode steps per warp, including loop repetition.
// Registers and shared bytes here are explicit caller allowances, not hardware
// facts or compiler allocation measurements. Injection must additionally check
// the target's measured limits and the compiler's final resource report.
type Budget struct {
	MaxInstructions int
	MaxSteps        uint64
	MaxRegisters    uint32
	MaxSharedBytes  uint32
	MaxSlots        uint32
}

// Analysis reports the conservative instruction and register bounds. Registers
// includes declared virtual registers and one loop counter per Repeat. It is
// not a native compiler's register count or an occupancy estimate.
type Analysis struct {
	MaxSteps  uint64
	Registers uint32
	Loops     int
}

// VerifyError locates a rejected instruction; PC -1 denotes a program header.
type VerifyError struct {
	PC     int
	Reason string
}

func (e *VerifyError) Error() string { return errorText(e.PC, e.Reason) }
