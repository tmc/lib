package cudamega

import (
	"fmt"
	"math"
)

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
// elements is the input buffer length, not a scratch stride. Declared task
// extents are fixed; undeclared extents use elements. Each consumed source
// must cover the exact task extent. Grid.MaxElements bounds input and task
// extents and must be nonzero. Storage returns the individual scratch ranges.
// CPU Run and Reference use this check; generated CUDA uses the same exact
// task extents and refuses short sources before selection.
func (p *Program) Invocation(elements uint64) (Invocation, error) {
	if p == nil {
		return Invocation{}, fmt.Errorf("nil program")
	}
	if p.grid.MaxElements == 0 {
		return Invocation{}, fmt.Errorf("program requires a nonzero maximum element extent")
	}
	if elements > uint64(p.grid.MaxElements) {
		return Invocation{}, fmt.Errorf("extent %d exceeds maximum %d", elements, p.grid.MaxElements)
	}
	for _, s := range p.steps {
		if s.Operation.Body == nil && s.Operation.Kind != Add && s.Operation.Kind != Multiply {
			return Invocation{}, fmt.Errorf("task %q does not have a scalar operation", s.Task.ID)
		}
	}
	if !p.bound {
		return invocationSizes(uint64(len(p.steps)), elements, uint64(p.grid.CTAs), uint64(^uint(0)>>1))
	}
	if elements == 0 || elements > math.MaxUint32 {
		return Invocation{}, fmt.Errorf("invalid finite program extent")
	}
	_, scratch, err := p.storage(elements)
	if err != nil {
		return Invocation{}, err
	}
	order, err := StartOrderWords(uint64(len(p.steps)), uint64(p.grid.CTAs))
	if err != nil || order > uint64(^uint(0)>>1)/8 {
		return Invocation{}, fmt.Errorf("finite program start-order size exceeds int range")
	}
	return Invocation{Elements: uint32(elements), ScratchWords: scratch, StartOrderWords: int(order)}, nil
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
