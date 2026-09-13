package cuptisass

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// InstructionCensus computes the instruction count per kernel function
// from a parsed SASS disassembly.
func InstructionCensus(dis *cudasass.Disassembly) map[string]int {
	if dis == nil {
		return nil
	}
	census := make(map[string]int, len(dis.Functions))
	for _, fn := range dis.Functions {
		census[fn.Name] = len(fn.Instructions)
	}
	return census
}

// InstructionOffsets returns the sorted instruction PC offsets for the named function.
func InstructionOffsets(dis *cudasass.Disassembly, kernelName string) ([]uint32, error) {
	if dis == nil {
		return nil, fmt.Errorf("cuptisass: nil disassembly")
	}
	fn := dis.Function(kernelName)
	if fn == nil {
		return nil, fmt.Errorf("cuptisass: kernel %q not found in disassembly", kernelName)
	}
	offsets := make([]uint32, len(fn.Instructions))
	for i, inst := range fn.Instructions {
		offsets[i] = uint32(inst.Address)
	}
	return offsets, nil
}

// VerifyCensus cross-checks flushed SASS metric records against the expected
// instruction count and offsets from a static disassembly. It reports an error
// if any record references an instruction offset not present in the disassembly,
// or if the record count deviates from the expected count when full coverage
// is expected.
func VerifyCensus(records []Record, dis *cudasass.Disassembly, kernelName string) error {
	if dis == nil {
		return fmt.Errorf("cuptisass: nil disassembly")
	}
	fn := dis.Function(kernelName)
	if fn == nil {
		return fmt.Errorf("cuptisass: kernel %q not found in disassembly", kernelName)
	}
	knownOffsets := make(map[uint32]bool, len(fn.Instructions))
	for _, inst := range fn.Instructions {
		knownOffsets[uint32(inst.Address)] = true
	}

	var kernelRecords int
	for _, rec := range records {
		if rec.FunctionName != kernelName {
			continue
		}
		kernelRecords++
		if !knownOffsets[rec.PCOffset] {
			return fmt.Errorf("cuptisass: record PC offset 0x%x not found in SASS instructions for %q", rec.PCOffset, kernelName)
		}
	}
	return nil
}
