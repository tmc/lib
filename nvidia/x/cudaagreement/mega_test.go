package cudaagreement_test

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// TestMegaUnits checks how the mega executor divides a plan. A run is what
// one persistent launch performs, and the worker feeds each task only its
// predecessor's output, so a step reading anything else must start a run of
// its own rather than silently read the predecessor.
func TestMegaUnits(t *testing.T) {
	tests := []struct {
		name string
		edit func(*cudaagreement.Plan, map[string]cudamega.Operation, map[string]cudaagreement.Buffer, *cudaagreement.MegaExecutor)
		want []string
	}{
		{"whole chain", nil, []string{"add3,mul5,add7"}},
		{
			"no operation for the middle step",
			func(_ *cudaagreement.Plan, ops map[string]cudamega.Operation, _ map[string]cudaagreement.Buffer, _ *cudaagreement.MegaExecutor) {
				delete(ops, "mul5")
			},
			[]string{"add3", "mul5 declined: no persistent-worker operation for step", "add7"},
		},
		{
			"input is not the previous output",
			func(p *cudaagreement.Plan, _ map[string]cudamega.Operation, _ map[string]cudaagreement.Buffer, _ *cudaagreement.MegaExecutor) {
				p.Steps[2].Inputs = []string{"x"}
			},
			[]string{"add3,mul5", "add7"},
		},
		{
			"size changes",
			func(_ *cudaagreement.Plan, _ map[string]cudamega.Operation, b map[string]cudaagreement.Buffer, _ *cudaagreement.MegaExecutor) {
				y := b["y"]
				y.Bytes /= 2
				b["y"] = y
			},
			[]string{"add3,mul5", "add7 declined: input is 16384 bytes and output 8192; a worker task maps elements one to one"},
		},
		{
			"two inputs",
			func(p *cudaagreement.Plan, _ map[string]cudamega.Operation, _ map[string]cudaagreement.Buffer, _ *cudaagreement.MegaExecutor) {
				p.Steps[0].Inputs = []string{"x", "x"}
			},
			[]string{"add3 declined: step takes 2 inputs; a worker task takes one", "mul5,add7"},
		},
		{
			"unbound buffer",
			func(_ *cudaagreement.Plan, _ map[string]cudamega.Operation, b map[string]cudaagreement.Buffer, _ *cudaagreement.MegaExecutor) {
				delete(b, "t2")
			},
			[]string{"add3 ", "mul5 declined: step buffers are not bound", "add7 declined: step buffers are not bound"},
		},
		{
			"partial word",
			func(_ *cudaagreement.Plan, _ map[string]cudamega.Operation, b map[string]cudaagreement.Buffer, _ *cudaagreement.MegaExecutor) {
				for _, n := range []string{"x", "t1"} {
					v := b[n]
					v.Bytes = 6
					b[n] = v
				}
			},
			[]string{"add3 declined: buffers are 6 bytes, not whole uint32 words", "mul5 declined: input is 6 bytes and output 16384; a worker task maps elements one to one", "add7"},
		},
		{
			"forced decline",
			func(_ *cudaagreement.Plan, _ map[string]cudamega.Operation, _ map[string]cudaagreement.Buffer, e *cudaagreement.MegaExecutor) {
				e.Decline = func(s cudaagreement.Step) error {
					if s.Name == "add7" {
						return errors.New("forced")
					}
					return nil
				}
			},
			[]string{"add3,mul5", "add7 declined: forced"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, ops := megaPlan(t, "sm_121a")
			bufs := megaBuffers()
			e := &cudaagreement.MegaExecutor{Operations: ops}
			if tt.edit != nil {
				tt.edit(plan, ops, bufs, e)
			}
			got := cudaagreement.MegaUnits(e, plan, bufs)
			want := slices.Clone(tt.want)
			for i := range want {
				want[i] = strings.TrimSpace(want[i])
			}
			if !slices.Equal(got, want) {
				t.Fatalf("units = %q, want %q", got, want)
			}
		})
	}
}

// TestCheckStartOrder feeds the start-order check the buffers a worker could
// leave behind. It is what a Mega record is made from, so each way the device
// could have run a task other than as the program dealt it must be refused.
func TestCheckStartOrder(t *testing.T) {
	// Two tasks on a 4-CTA grid: a owned by CTAs [0,2), b by [2,4).
	steps := []cudamega.Step{
		{Task: cudataskgraph.Task{ID: "a"}, Owner: cudamega.Owner{Lo: 0, Hi: 2}},
		{Task: cudataskgraph.Task{ID: "b"}, Owner: cudamega.Owner{Lo: 2, Hi: 4}},
	}
	// Layout: ord[0] is the ticket counter, ord[1+cta*tasks+task] the ticket.
	good := []uint64{4, 1, 0, 2, 0, 0, 3, 0, 4}
	tests := []struct {
		name string
		edit func([]uint64) []uint64
		want string
	}{
		{"as dealt", nil, ""},
		{"short buffer", func(o []uint64) []uint64 { return o[:5] }, "start order has 5 words, want 9"},
		{"counter", func(o []uint64) []uint64 { o[0] = 3; return o }, "dealt 3 start tickets, want 4"},
		{"owner never started", func(o []uint64) []uint64 { o[3] = 0; o[0] = 4; return o }, "task a on CTA 1 has ticket 0"},
		{"non-owner started", func(o []uint64) []uint64 { o[2] = 5; return o }, "task b on CTA 0 has ticket 5"},
		{"ticket reused", func(o []uint64) []uint64 { o[8] = 3; return o }, "ticket 3, out of range or reused"},
		{"ticket out of range", func(o []uint64) []uint64 { o[8] = 9; return o }, "ticket 9, out of range or reused"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ord := slices.Clone(good)
			if tt.edit != nil {
				ord = tt.edit(ord)
			}
			err := cudaagreement.CheckStartOrder(steps, 4, ord)
			switch {
			case tt.want == "" && err != nil:
				t.Fatalf("checkStartOrder = %v, want nil", err)
			case tt.want != "" && (err == nil || !strings.Contains(err.Error(), tt.want)):
				t.Fatalf("checkStartOrder = %v, want error containing %q", err, tt.want)
			}
		})
	}
}

// TestVerifyMegaGeometry checks the geometry rule: a Mega step records the
// worker's geometry and program, which differ from the artifact launch's, and
// that is not a disagreement. Geometry still separates two arms that both
// launched the artifact.
func TestVerifyMegaGeometry(t *testing.T) {
	out := []byte{1, 2, 3, 4}
	launch := record("launch", cudaplan.BackendLaunch, cudaplan.BackendLaunch, out, "add3", "mul5")
	mega := record("mega", cudaplan.BackendMega, cudaplan.BackendMega, out, "add3", "mul5")
	for i := range mega.Record.Steps {
		mega.Record.Steps[i].Program = "worker-digest"
		mega.Record.Steps[i].EntryPoint = "dag_worker"
		mega.Record.Steps[i].Grid, mega.Record.Steps[i].Block = 48, 128
	}
	if err := cudaagreement.Verify([]cudaagreement.Arm{launch, mega}); err != nil {
		t.Fatalf("Verify over a mega arm: %v", err)
	}
	if !strings.Contains(mega.Record.Steps[0].String(), "program=worker-digest") {
		t.Fatalf("step record %q does not name its program", mega.Record.Steps[0])
	}
	// The artifact digest is still compared: a mega arm whose steps name
	// another artifact ran other work.
	mega.Record.Steps[1].Digest = "other"
	if err := cudaagreement.Verify([]cudaagreement.Arm{launch, mega}); err == nil || !strings.Contains(err.Error(), "used artifact") {
		t.Fatalf("Verify = %v, want an artifact mismatch", err)
	}
}

// TestMegaProgram builds the program the native test launches, so that a
// refusal by cudamega shows up here rather than as a declined run on the
// device.
func TestMegaProgram(t *testing.T) {
	plan, ops := megaPlan(t, "sm_121a")
	e := &cudaagreement.MegaExecutor{Operations: ops, Target: cudamega.SM121a, Grid: megaGrid}
	progs, err := cudaagreement.MegaPrograms(e, plan, megaBuffers())
	if err != nil {
		t.Fatal(err)
	}
	if len(progs) != 1 {
		t.Fatalf("%d programs, want 1", len(progs))
	}
	steps := progs[0].Steps()
	for k, s := range steps {
		if s.Task.ID != plan.Steps[k].Name || s.Operation != ops[s.Task.ID] || s.Owner.Count() != megaGrid.CTAs {
			t.Fatalf("program step %d = %+v, want %s owned by all %d CTAs", k, s, plan.Steps[k].Name, megaGrid.CTAs)
		}
	}
	src, err := progs[0].Source()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(src, `extern "C" __global__ void dag_worker(`) {
		t.Fatal("program source has no dag_worker entry")
	}
}
