package cudaagreement_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamega"
)

// The mega fixture is a unary uint32 chain, t1 = x+3, t2 = t1*5, y = t2+7,
// because a persistent-worker task maps one input to one output by an
// operation cudamega defines. Each step's own artifact is a PTX kernel
// computing the same thing, which is what the launch and graph arms run and
// what the Digest in every arm's record names.
const megaElements = 4096

type unaryStep struct {
	name    string
	ptxOp   string // PTX instruction applied to the element and the scalar
	op      cudamega.Operation
	in, out string
}

var megaChain = []unaryStep{
	{"add3", "add.u32", cudamega.Operation{Kind: cudamega.Add, Scalar: 3}, "x", "t1"},
	{"mul5", "mul.lo.u32", cudamega.Operation{Kind: cudamega.Multiply, Scalar: 5}, "t1", "t2"},
	{"add7", "add.u32", cudamega.Operation{Kind: cudamega.Add, Scalar: 7}, "t2", "y"},
}

// unaryPTX emits out[i] = a[i] op scalar over uint32.
func unaryPTX(target, op string, scalar uint32) string {
	return fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry apply(.param .u64 a, .param .u64 out) {
 .reg .pred p;
 .reg .b32 tid, ntid, ctaid, idx, va, r;
 .reg .b64 pa, po, off;
 mov.u32 tid, %%tid.x;
 mov.u32 ntid, %%ntid.x;
 mov.u32 ctaid, %%ctaid.x;
 mad.lo.s32 idx, ctaid, ntid, tid;
 setp.ge.u32 p, idx, %d;
 @p bra done;
 ld.param.u64 pa, [a];
 ld.param.u64 po, [out];
 mul.wide.u32 off, idx, 4;
 add.u64 pa, pa, off;
 add.u64 po, po, off;
 ld.global.u32 va, [pa];
 %s r, va, %d;
 st.global.u32 [po], r;
done:
 ret;
}
`, target, megaElements, op, scalar)
}

// megaPlan builds the unary chain for target, with each step's operation.
func megaPlan(t *testing.T, target string) (*cudaagreement.Plan, map[string]cudamega.Operation) {
	t.Helper()
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "a", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	plan := &cudaagreement.Plan{Name: "megachain"}
	ops := make(map[string]cudamega.Operation)
	for _, s := range megaChain {
		art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(unaryPTX(target, s.ptxOp, s.op.Scalar)), "apply", schema, target)
		if err != nil {
			t.Fatal(err)
		}
		plan.Steps = append(plan.Steps, cudaagreement.Step{
			Name:     s.name,
			Artifact: art,
			Inputs:   []string{s.in},
			Output:   s.out,
			Grid:     megaElements / fixtureBlock,
			Block:    fixtureBlock,
		})
		ops[s.name] = s.op
	}
	return plan, ops
}

// megaReference evaluates the chain on the host.
func megaReference(x []uint32) []uint32 {
	y := make([]uint32, len(x))
	for i, v := range x {
		y[i] = (v+3)*5 + 7
	}
	return y
}

// megaBuffers binds every fixture buffer to a distinct fake address at the
// fixture size, for host tests that never dereference them.
func megaBuffers() map[string]cudaagreement.Buffer {
	bufs := make(map[string]cudaagreement.Buffer)
	for i, name := range []string{"x", "t1", "t2", "y"} {
		bufs[name] = cudaagreement.Buffer{Ptr: driver.DevicePtr(0x10000 * (i + 1)), Bytes: 4 * megaElements}
	}
	return bufs
}

// megaGrid is the worker geometry: small enough to be resident on any
// supported device, wide enough that every task is split across CTAs.
var megaGrid = cudamega.Grid{CTAs: 8, ThreadsPerCTA: 128}
