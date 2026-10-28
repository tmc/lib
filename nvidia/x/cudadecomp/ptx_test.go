package cudadecomp

import (
	"encoding/binary"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// ptxThread interprets the instructions of the merge kernel for one thread.
// It implements only the opcodes [EmitBytePlaneMergePTX] uses and fails on
// any other, so a change to the emitted code that it cannot follow is an
// error rather than a silent pass.
type ptxThread struct {
	regs   map[string]uint64
	params map[string]uint64
	mem    map[uint64][]byte // allocation base -> bytes
	ctaid  uint64
	ntid   uint64
	tid    uint64
}

func (th *ptxThread) addr(a uint64) ([]byte, error) {
	for base, b := range th.mem {
		if a >= base && a < base+uint64(len(b)) {
			return b[a-base:], nil
		}
	}
	return nil, fmt.Errorf("address %#x unmapped", a)
}

func (th *ptxThread) value(op string) (uint64, error) {
	switch op {
	case "%ctaid.x":
		return th.ctaid, nil
	case "%ntid.x":
		return th.ntid, nil
	case "%tid.x":
		return th.tid, nil
	}
	if strings.HasPrefix(op, "%") {
		v, ok := th.regs[op]
		if !ok {
			return 0, fmt.Errorf("register %s read before written", op)
		}
		return v, nil
	}
	return strconv.ParseUint(op, 0, 64)
}

func memOperand(op string) string {
	return strings.TrimSuffix(strings.TrimPrefix(op, "["), "]")
}

func (th *ptxThread) run(k *cudaptx.Kernel) error {
	labels := map[string]int{}
	for _, l := range k.Labels {
		labels[l.Name] = l.Instruction
	}
	for pc := 0; pc < len(k.Instructions); pc++ {
		in := k.Instructions[pc]
		ops := cudaptx.SplitOperands(in.Operands)
		if in.Predicate != "" {
			p := strings.TrimPrefix(in.Predicate, "@")
			neg := strings.HasPrefix(p, "!")
			v, err := th.value(strings.TrimPrefix(p, "!"))
			if err != nil {
				return err
			}
			if (v != 0) == neg {
				continue
			}
		}
		get := func(i int) uint64 {
			v, err := th.value(ops[i])
			if err != nil {
				panic(err)
			}
			return v
		}
		switch in.Opcode {
		case "ld.param.u64", "ld.param.u32":
			v, ok := th.params[memOperand(ops[1])]
			if !ok {
				return fmt.Errorf("line %d: unknown param %s", in.Line, ops[1])
			}
			th.regs[ops[0]] = v
		case "mov.u32", "cvta.to.global.u64":
			th.regs[ops[0]] = get(1)
		case "mad.lo.s32":
			th.regs[ops[0]] = uint64(uint32(get(1)*get(2) + get(3)))
		case "setp.ge.u32":
			th.regs[ops[0]] = 0
			if uint32(get(1)) >= uint32(get(2)) {
				th.regs[ops[0]] = 1
			}
		case "bra":
			target, ok := labels[ops[0]]
			if !ok {
				return fmt.Errorf("line %d: unknown label %s", in.Line, ops[0])
			}
			pc = target - 1
		case "mul.wide.u32":
			th.regs[ops[0]] = uint64(uint32(get(1))) * uint64(uint32(get(2)))
		case "add.s64":
			th.regs[ops[0]] = get(1) + get(2)
		case "ld.global.u32":
			b, err := th.addr(th.regs[memOperand(ops[1])])
			if err != nil {
				return err
			}
			th.regs[ops[0]] = uint64(binary.LittleEndian.Uint32(b))
		case "prmt.b32":
			th.regs[ops[0]] = uint64(prmt(uint32(get(1)), uint32(get(2)), uint32(get(3))))
		case "st.global.v2.u32":
			b, err := th.addr(th.regs[memOperand(ops[0])])
			if err != nil {
				return err
			}
			vs := strings.Split(strings.Trim(ops[1], "{}"), ",")
			for i, v := range vs {
				x, err := th.value(strings.TrimSpace(v))
				if err != nil {
					return err
				}
				binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
			}
		case "ret":
			return nil
		default:
			return fmt.Errorf("line %d: interpreter has no %s", in.Line, in.Opcode)
		}
	}
	return nil
}

// prmt is the default mode of prmt.b32 from the PTX ISA: byte k of the
// result is byte (c >> 4k) & 7 of the pair {b, a}, a supplying bytes 0-3.
func prmt(a, b, c uint32) uint32 {
	x := uint64(b)<<32 | uint64(a)
	var d uint32
	for k := range 4 {
		sel := c >> (4 * k) & 0xf
		byt := uint32(x >> (8 * (sel & 7)))
		if sel&8 != 0 { // sign-replicate mode
			byt = 0
			if x>>(8*(sel&7))&0x80 != 0 {
				byt = 0xff
			}
		}
		d |= (byt & 0xff) << (8 * k)
	}
	return d
}

// mergeOnHost runs every thread of the merge kernel in ptx over w.
func mergeOnHost(t *testing.T, ptx, name string, w []uint16) []uint16 {
	t.Helper()
	m, err := cudaptx.ParseString(ptx)
	if err != nil {
		t.Fatal(err)
	}
	k := m.Kernel(name)
	if k == nil {
		t.Fatalf("no kernel %s", name)
	}
	planes := SplitBytePlanes(w)
	n := len(w) / 4
	out := make([]byte, 2*len(w)+16) // 16 guard bytes
	const hi, lo, dst = 0x1000_0000, 0x2000_0000, 0x3000_0000
	mem := map[uint64][]byte{hi: planes[:len(w)], lo: planes[len(w):], dst: out}
	const block = 32
	for cta := range (n + block - 1) / block {
		for tid := range block {
			th := &ptxThread{
				regs:   map[string]uint64{},
				params: map[string]uint64{name + "_hi": hi, name + "_lo": lo, name + "_out": dst, name + "_n": uint64(n)},
				mem:    mem,
				ctaid:  uint64(cta),
				ntid:   block,
				tid:    uint64(tid),
			}
			if err := th.run(k); err != nil {
				t.Fatalf("thread %d.%d: %v", cta, tid, err)
			}
		}
	}
	for i, b := range out[2*len(w):] {
		if b != 0 {
			t.Fatalf("guard byte %d written", i)
		}
	}
	got := make([]uint16, len(w))
	for i := range got {
		got[i] = binary.LittleEndian.Uint16(out[2*i:])
	}
	return got
}

func TestBytePlaneMergePTX(t *testing.T) {
	for _, target := range []string{"sm_90a", "sm_121a"} {
		ptx, err := EmitBytePlaneMergePTX("merge_planes", target)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(ptx, ".target "+target) {
			t.Errorf("%s: no .target line", target)
		}
		m, err := cudaptx.ParseString(ptx)
		if err != nil {
			t.Fatal(err)
		}
		if err := m.AssertOpcodes("merge_planes", []string{"prmt", "st"}, []string{"ld.local", "st.local"}); err != nil {
			t.Error(err)
		}
		w := weights(4*100, false, 31) // 100 threads, 4 idle in the last block
		if got := mergeOnHost(t, ptx, "merge_planes", w); !slices.Equal(got, w) {
			t.Errorf("%s: merged weights differ", target)
		}
	}
}

// TestBytePlaneMergePTXMutants checks that the interpreter sees a wrong
// selector or swapped operands.
func TestBytePlaneMergePTXMutants(t *testing.T) {
	ptx, err := EmitBytePlaneMergePTX("merge_planes", "sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	w := weights(64, false, 32)
	for _, m := range []struct{ old, new string }{
		{"0x5140", "0x4051"},
		{"0x7362", "0x5140"},
		{"%r7, %r6, 0x5140", "%r6, %r7, 0x5140"},
	} {
		bad := strings.Replace(ptx, m.old, m.new, 1)
		if bad == ptx {
			t.Fatalf("mutant %q not applied", m.old)
		}
		if got := mergeOnHost(t, bad, "merge_planes", w); slices.Equal(got, w) {
			t.Errorf("mutant %q -> %q not detected", m.old, m.new)
		}
	}
}

func TestPrmt(t *testing.T) {
	tests := []struct{ a, b, c, want uint32 }{
		{0x33221100, 0x77665544, 0x3210, 0x33221100},
		{0x33221100, 0x77665544, 0x7654, 0x77665544},
		{0x33221100, 0x77665544, 0x5140, 0x55114400},
		{0x000000f0, 0, 0x0008, 0xf0f0f0ff},
	}
	for _, tt := range tests {
		if got := prmt(tt.a, tt.b, tt.c); got != tt.want {
			t.Errorf("prmt(%#x, %#x, %#x) = %#x, want %#x", tt.a, tt.b, tt.c, got, tt.want)
		}
	}
}

func TestEmitBytePlaneMergePTXErrors(t *testing.T) {
	if _, err := EmitBytePlaneMergePTX("bad name", "sm_90a"); err == nil {
		t.Error("bad name accepted")
	}
	if _, err := EmitBytePlaneMergePTX("k", "sm_75"); err == nil {
		t.Error("sm_75 accepted")
	}
}
