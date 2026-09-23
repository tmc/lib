package cudabpfdev

import (
	"context"
	"fmt"
	"math/bits"
)

// Result is the host interpreter's snapshot at program exit. Values holds each
// active lane's returned value. Map holds this invocation's counters, updated
// once per active warp. Misses counts out-of-range map keys without writing.
// Steps counts warp instruction steps, not GPU cycles or lane instructions.
type Result struct {
	Values        [32]uint32
	Map           []uint32
	Misses, Steps uint64
}

// Evaluate interprets a verified program for one active warp as a host oracle.
// Registers and map counters are private to each call. The result is published
// only at exit; cancellation or verification failure returns no snapshot.
// An empty active mask performs no instructions or side effects. This is not
// a GPU execution path and supplies no performance or injection qualification.
func Evaluate(ctx context.Context, p Program, b Budget, active uint32) (Result, error) {
	a, err := Verify(p, b)
	if err != nil {
		return Result{}, err
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	// The oracle has its own allocation ceiling, independent of a caller's
	// device-map allowance. It must not allocate gigabytes to test a program.
	if p.Slots > 1<<20 {
		return Result{}, fmt.Errorf("cudabpfdev: host oracle map exceeds 1048576 slots")
	}
	r := Result{Map: make([]uint32, p.Slots)}
	if active == 0 {
		return r, nil
	}
	leader := bits.TrailingZeros32(active)
	var regs [32][32]uint32
	counts := make([]uint32, len(p.Code))
	for pc := 0; ; {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		if r.Steps >= a.MaxSteps {
			return Result{}, fmt.Errorf("cudabpfdev: interpreter exceeded verified bound")
		}
		r.Steps++
		in := p.Code[pc]
		next := pc + 1
		switch in.Op {
		case Exit:
			for lane := 0; lane < 32; lane++ {
				if active&(uint32(1)<<lane) != 0 {
					r.Values[lane] = regs[lane][in.A]
				}
			}
			return r, nil
		case JumpNonzero:
			if regs[leader][in.A] != 0 {
				next = in.Target
			}
		case Repeat:
			counts[pc]++
			if counts[pc] < in.Imm {
				next = in.Target
			} else {
				counts[pc] = 0
			}
		case MapAdd:
			key, value := regs[leader][in.A], regs[leader][in.B]
			if key >= p.Slots {
				r.Misses++
			} else {
				r.Map[key] += value
			}
		default:
			var aggregate uint32
			if in.Op == Leader {
				aggregate = regs[leader][in.A]
			}
			if in.Op == Ballot {
				for lane := 0; lane < 32; lane++ {
					if active&(uint32(1)<<lane) != 0 && regs[lane][in.A] != 0 {
						aggregate |= uint32(1) << lane
					}
				}
			}
			for lane := 0; lane < 32; lane++ {
				if active&(uint32(1)<<lane) == 0 {
					continue
				}
				v := &regs[lane]
				switch in.Op {
				case Const:
					v[in.Dst] = in.Imm
				case Move:
					v[in.Dst] = v[in.A]
				case Add:
					v[in.Dst] = v[in.A] + v[in.B]
				case And:
					v[in.Dst] = v[in.A] & v[in.B]
				case Lane:
					v[in.Dst] = uint32(lane)
				case Ballot, Leader:
					v[in.Dst] = aggregate
				}
			}
		}
		pc = next
	}
}
