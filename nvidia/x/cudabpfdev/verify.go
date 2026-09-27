package cudabpfdev

import "fmt"

func errorText(pc int, reason string) string {
	return fmt.Sprintf("cudabpfdev: instruction %d: %s", pc, reason)
}

type loop struct {
	first, last int
	count       uint32
}

// Verify checks the instruction contract, single-entry loop structure,
// resource bounds and warp uniformity at a control-flow fixed point. It rejects
// unreachable instructions and reports the instruction that violates a rule.
func Verify(p Program, b Budget) (Analysis, error) {
	var result Analysis
	fail := func(pc int, reason string) (Analysis, error) { return Analysis{}, &VerifyError{pc, reason} }
	n := len(p.Code)
	if p.Version != 1 {
		return fail(-1, "unsupported bytecode version")
	}
	if p.Registers == 0 || p.Registers > 32 || n == 0 || b.MaxInstructions <= 0 || n > b.MaxInstructions || n > 4096 {
		return fail(-1, "invalid register or instruction count")
	}
	if b.MaxSteps == 0 || p.Slots == 0 || p.Slots > b.MaxSlots || p.StagingBytes > b.MaxSharedBytes {
		return fail(-1, "map, shared memory or step budget exceeded")
	}
	edges := make([][]int, n)
	var loops []loop
	for pc, in := range p.Code {
		var dst, a, c, imm, target bool
		switch in.Op {
		case Const:
			dst, imm = true, true
		case Move, Ballot, Leader:
			dst, a = true, true
		case Add, And:
			dst, a, c = true, true, true
		case Lane:
			dst = true
		case JumpNonzero:
			a, target = true, true
		case Repeat:
			imm, target = true, true
		case MapAdd:
			a, c = true, true
		case Exit:
			a = true
		default:
			return fail(pc, "unsupported opcode")
		}
		if (!dst && in.Dst != 0) || (!a && in.A != 0) || (!c && in.B != 0) || (!imm && in.Imm != 0) || (!target && in.Target != 0) {
			return fail(pc, "nonzero unused instruction field")
		}
		if (dst && in.Dst >= p.Registers) || (a && in.A >= p.Registers) || (c && in.B >= p.Registers) {
			return fail(pc, "register out of range")
		}
		if target && (in.Target < 0 || in.Target >= n) {
			return fail(pc, "branch target out of range")
		}
		if in.Op == JumpNonzero && in.Target <= pc {
			return fail(pc, "unbounded backward branch")
		}
		if in.Op == Repeat {
			if in.Imm == 0 || in.Target >= pc {
				return fail(pc, "invalid bounded loop")
			}
			loops = append(loops, loop{in.Target, pc, in.Imm})
		}
		if in.Op != Exit {
			if pc+1 >= n {
				return fail(pc, "execution falls off program")
			}
			edges[pc] = append(edges[pc], pc+1)
			if target {
				edges[pc] = append(edges[pc], in.Target)
			}
		}
	}
	result.Loops = len(loops)
	result.Registers = uint32(p.Registers) + uint32(len(loops))
	if result.Registers > b.MaxRegisters {
		return fail(-1, "register budget exceeded")
	}
	for i, l := range loops {
		for _, r := range loops[i+1:] {
			if l.first < r.first && r.first <= l.last && l.last < r.last || r.first < l.first && l.first <= r.last && r.last < l.last {
				return fail(l.last, "overlapping loop regions")
			}
		}
		for pc, next := range edges {
			for _, to := range next {
				inside := pc >= l.first && pc <= l.last
				enters := to >= l.first && to <= l.last
				if !inside && enters && to != l.first {
					return fail(pc, "irreducible entry into loop body")
				}
				if inside && !enters && !(pc == l.last && to == pc+1) {
					return fail(pc, "branch crosses loop boundary")
				}
			}
		}
	}
	// Count each instruction at the product of all enclosing loop bounds.
	// Checked against the budget before multiplying, so huge nested bounds
	// cannot wrap and turn an unbounded request into a small one.
	for pc := range p.Code {
		steps := uint64(1)
		for _, l := range loops {
			if pc >= l.first && pc <= l.last {
				if steps > b.MaxSteps/uint64(l.count) {
					return fail(pc, "step budget exceeded")
				}
				steps *= uint64(l.count)
			}
		}
		if steps > b.MaxSteps-result.MaxSteps {
			return fail(pc, "step budget exceeded")
		}
		result.MaxSteps += steps
	}
	// Each bit means the register may vary between active lanes. All registers
	// are zero on entry. Joins union varying bits; transfer is monotone. At most
	// 32 bits can be added per instruction, bounding fixed-point work.
	state := make([]uint32, n)
	reached := make([]bool, n)
	queued := make([]bool, n)
	queue := []int{0}
	reached[0] = true
	queued[0] = true
	for len(queue) > 0 {
		pc := queue[0]
		queue = queue[1:]
		queued[pc] = false
		in := p.Code[pc]
		out := state[pc]
		vary := false
		writes := true
		switch in.Op {
		case Const, Ballot, Leader:
		case Lane:
			vary = true
		case Move:
			vary = out&(uint32(1)<<in.A) != 0
		case Add, And:
			vary = out&((uint32(1)<<in.A)|(uint32(1)<<in.B)) != 0
		default:
			writes = false
		}
		if writes {
			out &^= uint32(1) << in.Dst
			if vary {
				out |= uint32(1) << in.Dst
			}
		}
		for _, to := range edges[pc] {
			merged := state[to] | out
			if !reached[to] || merged != state[to] {
				state[to] = merged
				reached[to] = true
				if !queued[to] {
					queue = append(queue, to)
					queued[to] = true
				}
			}
		}
	}
	for pc, in := range p.Code {
		if !reached[pc] {
			return fail(pc, "unreachable instruction")
		}
		if in.Op == JumpNonzero && state[pc]&(uint32(1)<<in.A) != 0 {
			return fail(pc, "lane-varying branch condition")
		}
		if in.Op == MapAdd {
			keyVary := state[pc]&(uint32(1)<<in.A) != 0
			valVary := state[pc]&(uint32(1)<<in.B) != 0
			if keyVary {
				return fail(pc, "warp-uniformity violation: lane-varying map key")
			}
			if valVary {
				return fail(pc, "warp-uniformity violation: lane-varying map value")
			}
		}
	}
	return result, nil
}
