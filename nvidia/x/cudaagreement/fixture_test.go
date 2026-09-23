package cudaagreement_test

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// The fixture is an integer chain so that agreement has a bit-exact answer:
// t1 = x*w, t2 = t1+x, y = t2^w over int32. Nothing here rounds, so a
// difference between two arms is a difference in what ran, not in how it
// rounded.
const (
	fixtureElements = 1024
	fixtureBlock    = 256
	fixtureGrid     = fixtureElements / fixtureBlock
)

// binaryPTX emits an elementwise int32 kernel out[i] = a[i] op b[i].
func binaryPTX(target, op string) string {
	return fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry apply(.param .u64 a, .param .u64 b, .param .u64 out) {
 .reg .pred p;
 .reg .b32 tid, ntid, ctaid, idx, va, vb, r;
 .reg .b64 pa, pb, po, off;
 mov.u32 tid, %%tid.x;
 mov.u32 ntid, %%ntid.x;
 mov.u32 ctaid, %%ctaid.x;
 mad.lo.s32 idx, ctaid, ntid, tid;
 setp.ge.u32 p, idx, %d;
 @p bra done;
 ld.param.u64 pa, [a];
 ld.param.u64 pb, [b];
 ld.param.u64 po, [out];
 mul.wide.u32 off, idx, 4;
 add.u64 pa, pa, off;
 add.u64 pb, pb, off;
 add.u64 po, po, off;
 ld.global.s32 va, [pa];
 ld.global.s32 vb, [pb];
 %s r, va, vb;
 st.global.s32 [po], r;
done:
 ret;
}
`, target, fixtureElements, op)
}

// fixturePlan builds the three-step integer chain for target.
func fixturePlan(t *testing.T, target string) *cudaagreement.Plan {
	t.Helper()
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "a", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "b", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	steps := []struct {
		name          string
		op            string
		lhs, rhs, out string
	}{
		{"mul", "mul.lo.s32", "x", "w", "t1"},
		{"add", "add.s32", "t1", "x", "t2"},
		{"xor", "xor.b32", "t2", "w", "y"},
	}
	plan := &cudaagreement.Plan{Name: "intchain"}
	for _, s := range steps {
		art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(binaryPTX(target, s.op)), "apply", schema, target)
		if err != nil {
			t.Fatal(err)
		}
		plan.Steps = append(plan.Steps, cudaagreement.Step{
			Name:     s.name,
			Artifact: art,
			Inputs:   []string{s.lhs, s.rhs},
			Output:   s.out,
			Grid:     fixtureGrid,
			Block:    fixtureBlock,
		})
	}
	return plan
}

// fixtureInputs returns x and w for one round.
func fixtureInputs(round int) (x, w []int32) {
	x = make([]int32, fixtureElements)
	w = make([]int32, fixtureElements)
	for i := range x {
		x[i] = int32(i-512+round) * 7
		w[i] = int32(i%13-6) + int32(round)
	}
	return x, w
}

// fixtureReference evaluates the chain on the host in int32.
func fixtureReference(x, w []int32) []int32 {
	y := make([]int32, len(x))
	for i := range x {
		t1 := x[i] * w[i]
		t2 := t1 + x[i]
		y[i] = t2 ^ w[i]
	}
	return y
}

// bytesOf renders an int32 readback as the bytes the comparison sees.
func bytesOf(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, n := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(n))
	}
	return b
}

// fixturePlanWithOp is fixturePlan with the final step's operation replaced,
// for the control that shows the comparison failing on device output.
func fixturePlanWithOp(t *testing.T, target, op string) *cudaagreement.Plan {
	t.Helper()
	p := fixturePlan(t, target)
	last := p.Steps[len(p.Steps)-1]
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(binaryPTX(target, op)), "apply", last.Artifact.Schema, target)
	if err != nil {
		t.Fatal(err)
	}
	p.Steps[len(p.Steps)-1].Artifact = art
	return p
}
