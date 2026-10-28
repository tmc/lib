package qualify

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

// ptxSim interprets, for one thread, the subset of PTX that this package
// emits, so emitted code can be checked against the Go references without
// a GPU. Registers hold raw bits; global memory and Tensor Memory are maps
// of 32-bit words.
type ptxSim struct {
	regs   map[string]uint64
	params map[string]uint64
	mem    map[uint64]uint32 // global memory, by byte address of each word
	tmem   map[uint32]uint32 // Tensor Memory, by (lane<<16 | column)
	out    map[uint64]uint16
	rbits  []uint32

	timer   func() uint64 // %globaltimer
	tryWait func() bool   // mbarrier.try_wait result
	sleeps  []uint64      // nanosleep arguments
}

func (s *ptxSim) val(op string) uint64 {
	op = strings.TrimSpace(op)
	if v, ok := s.regs[op]; ok {
		return v
	}
	if op == "%globaltimer" {
		return s.timer()
	}
	if strings.HasPrefix(op, "%") {
		panic("unset register " + op)
	}
	if strings.HasPrefix(op, "0f") {
		v, err := strconv.ParseUint(op[2:], 16, 32)
		if err != nil {
			panic("bad float operand " + op)
		}
		return v
	}
	v, err := strconv.ParseUint(op, 0, 64)
	if err != nil {
		panic("bad operand " + op)
	}
	return v
}

func (s *ptxSim) f32(op string) float32 { return math.Float32frombits(uint32(s.val(op))) }

func (s *ptxSim) setF32(r string, v float32) { s.regs[r] = uint64(math.Float32bits(v)) }

func (s *ptxSim) setBool(r string, b bool) {
	s.regs[r] = 0
	if b {
		s.regs[r] = 1
	}
}

func addr(op string) string { return strings.TrimSpace(strings.Trim(op, "[]")) }

// funcBody returns the body of the PTX function or kernel named name.
func funcBody(t *testing.T, ptx, name string) string {
	t.Helper()
	i := strings.Index(ptx, " "+name+"(")
	if i < 0 {
		t.Fatalf("no function %s in PTX", name)
	}
	j := strings.Index(ptx[i:], "\n{\n")
	k := strings.Index(ptx[i:], "\n}\n")
	if j < 0 || k < j {
		t.Fatalf("function %s has no body", name)
	}
	return ptx[i+j+3 : i+k]
}

func (s *ptxSim) run(t *testing.T, ptx string) {
	t.Helper()
	if s.regs == nil {
		s.regs = map[string]uint64{}
	}
	stmts := ptxStatements(ptx)
	labels := map[string]int{}
	for i, st := range stmts {
		if j := strings.Index(st.text, ":"); j > 0 && strings.IndexFunc(st.text[:j], notIdent) < 0 {
			labels[st.text[:j]] = i
		}
	}
	for pc := 0; pc < len(stmts); pc++ {
		text := stmts[pc].text
		mn, ops := splitInstruction(text)
		if mn == "" {
			continue
		}
		if i := strings.Index(text, "@"); i >= 0 && i < strings.Index(text, mn) {
			p := strings.Fields(text[i+1:])[0]
			want := uint64(1)
			if strings.HasPrefix(p, "!") {
				p, want = p[1:], 0
			}
			if s.regs[p] != want {
				continue
			}
		}
		u32 := func(v uint64) uint64 { return v & 0xffffffff }
		switch mn {
		case "ld.param.u64", "ld.param.u32", "ld.param.b32", "ld.param.b64":
			v, ok := s.params[addr(ops[1])]
			if !ok {
				t.Fatalf("ptxSim: unset parameter %s", ops[1])
			}
			s.regs[ops[0]] = v
		case "st.param.b32", "st.param.b64":
			s.params[addr(ops[0])] = s.val(ops[1])
		case "mov.u32", "mov.b32", "mov.f32", "mov.u64":
			s.regs[ops[0]] = s.val(ops[1])
		case "mad.lo.u32":
			s.regs[ops[0]] = u32(s.val(ops[1])*s.val(ops[2]) + s.val(ops[3]))
		case "shr.u32":
			s.regs[ops[0]] = s.val(ops[1]) >> s.val(ops[2])
		case "shl.b32":
			s.regs[ops[0]] = u32(s.val(ops[1]) << s.val(ops[2]))
		case "mul.lo.u32":
			s.regs[ops[0]] = u32(s.val(ops[1]) * s.val(ops[2]))
		case "mul.hi.u32":
			s.regs[ops[0]] = (s.val(ops[1]) * s.val(ops[2])) >> 32
		case "mul.wide.u32":
			s.regs[ops[0]] = s.val(ops[1]) * s.val(ops[2])
		case "add.u32":
			s.regs[ops[0]] = u32(s.val(ops[1]) + s.val(ops[2]))
		case "add.u64":
			s.regs[ops[0]] = s.val(ops[1]) + s.val(ops[2])
		case "sub.u64":
			s.regs[ops[0]] = s.val(ops[1]) - s.val(ops[2])
		case "xor.b32":
			s.regs[ops[0]] = s.val(ops[1]) ^ s.val(ops[2])
		case "div.u32":
			s.regs[ops[0]] = s.val(ops[1]) / s.val(ops[2])
		case "rem.u32":
			s.regs[ops[0]] = s.val(ops[1]) % s.val(ops[2])
		case "setp.ge.u32":
			s.setBool(ops[0], s.val(ops[1]) >= s.val(ops[2]))
		case "setp.le.u64":
			s.setBool(ops[0], s.val(ops[1]) <= s.val(ops[2]))
		case "setp.gt.f32":
			s.setBool(ops[0], s.f32(ops[1]) > s.f32(ops[2]))
		case "add.rn.f32":
			s.setF32(ops[0], s.f32(ops[1])+s.f32(ops[2]))
		case "sub.rn.f32":
			s.setF32(ops[0], s.f32(ops[1])-s.f32(ops[2]))
		case "mul.rn.f32":
			s.setF32(ops[0], s.f32(ops[1])*s.f32(ops[2]))
		case "abs.f32":
			s.setF32(ops[0], float32(math.Abs(float64(s.f32(ops[1])))))
		case "testp.finite.f32":
			v := float64(s.f32(ops[1]))
			s.setBool(ops[0], !math.IsNaN(v) && !math.IsInf(v, 0))
		case "bra":
			pc = labels[ops[0]] - 1
		case "ret":
			return
		case "nanosleep.u32":
			s.sleeps = append(s.sleeps, s.val(ops[0]))
		case "mbarrier.try_wait.parity.shared::cta.b64":
			s.setBool(ops[0], s.tryWait())
		case "tcgen05.wait::ld.sync.aligned":
		case "tcgen05.ld.sync.aligned.32x32b.x16.b32", "tcgen05.ld.sync.aligned.32x32b.x1.b32":
			ta := uint32(s.val(addr(ops[1])))
			for i, r := range strings.Split(strings.Trim(ops[0], "{}"), ",") {
				s.regs[strings.TrimSpace(r)] = uint64(s.tmem[ta+uint32(i)])
			}
		case "ld.global.v4.f32":
			a := s.val(addr(ops[1]))
			for i, r := range strings.Split(strings.Trim(ops[0], "{}"), ",") {
				s.regs[strings.TrimSpace(r)] = uint64(s.mem[a+4*uint64(i)])
			}
		case "ld.global.u32":
			a := addr(ops[1])
			off := uint64(0)
			if i := strings.IndexByte(a, '+'); i > 0 {
				off, _ = strconv.ParseUint(a[i+1:], 10, 64)
				a = a[:i]
			}
			s.regs[ops[0]] = uint64(s.mem[s.val(a)+off])
		case "atom.global.add.u32":
			a := s.val(addr(ops[1]))
			s.regs[ops[0]] = uint64(s.mem[a])
			s.mem[a] += uint32(s.val(ops[2]))
		case "st.global.v4.b32", "st.global.v2.b32", "st.global.b64":
			a := addr(ops[0])
			off := uint64(0)
			if i := strings.IndexByte(a, '+'); i > 0 {
				off, _ = strconv.ParseUint(a[i+1:], 10, 64)
				a = a[:i]
			}
			base := s.val(a) + off
			if mn == "st.global.b64" {
				v := s.val(ops[1])
				s.mem[base], s.mem[base+4] = uint32(v), uint32(v>>32)
				break
			}
			for i, r := range strings.Split(strings.Trim(ops[1], "{}"), ",") {
				s.mem[base+4*uint64(i)] = uint32(s.val(r))
			}
		case "cvt.rs.satfinite.e2m1x4.f32":
			var v [4]float32
			for i, r := range strings.Split(strings.Trim(ops[1], "{}"), ",") {
				v[i] = s.f32(r)
			}
			rb := uint32(s.val(ops[2]))
			s.rbits = append(s.rbits, rb)
			s.regs[ops[0]] = uint64(StochasticRoundE2M1x4(v, rb))
		case "st.global.b16":
			s.out[s.val(addr(ops[0]))] = uint16(s.val(ops[1]))
		default:
			t.Fatalf("ptxSim: unsupported instruction %q", text)
		}
	}
}
