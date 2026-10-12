package cudamega

import "fmt"

// Invocation is the checked storage required for one scalar program invocation.
// ScratchWords and StartOrderWords count uint32 and uint64 words, respectively.
// Their byte sizes fit int. An Invocation does not allocate storage or admit a
// device launch; the executor still checks kernel residency and ownership.
type Invocation struct {
	Elements        uint32
	ScratchWords    int
	StartOrderWords int
}

// Invocation checks elements against the program's scalar extent contract and
// returns its required storage sizes, before any allocation or native call.
// A nonzero task Extent must equal elements. A nonzero Grid.MaxElements bounds
// elements; zero leaves that bound unset. This method supports the current
// uniform scalar layout, whose scratch stride is elements.
//
// CPU Run and Reference use this check. Generated CUDA currently checks the
// maximum but does not enforce positive task Extent; CUDA exact-extent parity
// requires the wider operand and storage binding.
func (p *Program) Invocation(elements uint64) (Invocation, error) {
	if p == nil {
		return Invocation{}, fmt.Errorf("nil program")
	}
	if p.grid.MaxElements != 0 && elements > uint64(p.grid.MaxElements) {
		return Invocation{}, fmt.Errorf("extent %d exceeds maximum %d", elements, p.grid.MaxElements)
	}
	for _, s := range p.steps {
		if s.Operation.Kind != Add && s.Operation.Kind != Multiply {
			return Invocation{}, fmt.Errorf("task %q does not have a scalar operation", s.Task.ID)
		}
		if s.Task.Extent != 0 && s.Task.Extent != elements {
			return Invocation{}, fmt.Errorf("task %q declares extent %d, invocation has %d", s.Task.ID, s.Task.Extent, elements)
		}
	}
	return invocationSizes(uint64(len(p.steps)), elements, uint64(p.grid.CTAs), uint64(^uint(0)>>1))
}

func invocationSizes(tasks, elements, ctas, maxInt uint64) (Invocation, error) {
	scratch, err := ProgramScratchBytes(tasks, elements)
	if err != nil {
		return Invocation{}, err
	}
	order, err := StartOrderWords(tasks, ctas)
	if err != nil {
		return Invocation{}, err
	}
	if scratch > maxInt || order > maxInt/8 {
		return Invocation{}, fmt.Errorf("finite program buffer size exceeds int range")
	}
	return Invocation{Elements: uint32(elements), ScratchWords: int(scratch / 4), StartOrderWords: int(order)}, nil
}
