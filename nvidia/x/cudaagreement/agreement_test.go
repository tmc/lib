package cudaagreement_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// record builds an arm whose steps all executed on executed, for the gate
// tests that do not need a device.
func record(arm string, requested, executed cudaplan.TargetBackend, out []byte, steps ...string) cudaagreement.Arm {
	r := &cudaagreement.Record{Arm: arm, Requested: requested}
	for _, s := range steps {
		sr := cudaagreement.StepRecord{
			Step: s, Requested: requested, Executed: executed,
			Digest: "digest-" + s, EntryPoint: "apply", Grid: 4, Block: 256,
			Output: cudaagreement.Buffer{Ptr: 0x1000, Bytes: len(out)},
		}
		if executed != requested {
			sr.Declined = "forced"
		}
		r.Steps = append(r.Steps, sr)
	}
	return cudaagreement.Arm{Record: r, Output: out}
}

func TestVerify(t *testing.T) {
	ok := []byte{1, 2, 3, 4}
	launch := record("launch", cudaplan.BackendLaunch, cudaplan.BackendLaunch, ok, "mul", "add")
	graph := record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, ok, "mul", "add")

	tests := []struct {
		name string
		arms []cudaagreement.Arm
		want string // substring of the required error, empty for success
	}{
		{"agreeing arms", []cudaagreement.Arm{launch, graph}, ""},
		{"one arm", []cudaagreement.Arm{launch}, "need at least 2"},
		{
			"one byte differs",
			[]cudaagreement.Arm{launch, record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, []byte{1, 2, 4, 4}, "mul", "add")},
			"differ at output byte 2",
		},
		{
			"declined step",
			[]cudaagreement.Arm{launch, record("graph", cudaplan.BackendGraph, cudaplan.BackendLaunch, ok, "mul", "add")},
			"did not execute on Graph",
		},
		{
			"empty outputs",
			[]cudaagreement.Arm{
				record("launch", cudaplan.BackendLaunch, cudaplan.BackendLaunch, nil, "mul", "add"),
				record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, nil, "mul", "add"),
			},
			"produced no output bytes",
		},
		{
			"different steps",
			[]cudaagreement.Arm{launch, record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, ok, "mul", "sub")},
			"step 1 is add",
		},
		{
			"different artifact",
			[]cudaagreement.Arm{launch, func() cudaagreement.Arm {
				a := record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, ok, "mul", "add")
				a.Record.Steps[0].Digest = "other"
				return a
			}()},
			"used artifact",
		},
		{
			"different geometry",
			[]cudaagreement.Arm{launch, func() cudaagreement.Arm {
				a := record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, ok, "mul", "add")
				a.Record.Steps[1].Grid = 8
				return a
			}()},
			"ran 4x256",
		},
		{"no record", []cudaagreement.Arm{launch, {Output: ok}}, "no execution record"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := cudaagreement.Verify(tt.arms)
			switch {
			case tt.want == "" && err != nil:
				t.Fatalf("Verify = %v, want nil", err)
			case tt.want != "" && err == nil:
				t.Fatalf("Verify = nil, want error containing %q", tt.want)
			case tt.want != "" && !strings.Contains(err.Error(), tt.want):
				t.Fatalf("Verify = %v, want error containing %q", err, tt.want)
			}
		})
	}
}

// TestAgreeSeesEveryBit walks a single bit through the output so that no
// position is one the comparison happens not to look at.
func TestAgreeSeesEveryBit(t *testing.T) {
	base := []byte{0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77}
	a := record("launch", cudaplan.BackendLaunch, cudaplan.BackendLaunch, base, "mul")
	for i := range base {
		for bit := range 8 {
			mutant := append([]byte(nil), base...)
			mutant[i] ^= 1 << bit
			b := record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, mutant, "mul")
			if err := cudaagreement.Agree([]cudaagreement.Arm{a, b}); err == nil {
				t.Fatalf("Agree accepted a flip of byte %d bit %d", i, bit)
			}
		}
	}
	same := record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, append([]byte(nil), base...), "mul")
	if err := cudaagreement.Agree([]cudaagreement.Arm{a, same}); err != nil {
		t.Fatalf("Agree on identical outputs: %v", err)
	}
}

func TestPlanValidate(t *testing.T) {
	good := fixturePlan(t, "sm_121a")
	if err := good.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	tests := []struct {
		name string
		edit func(*cudaagreement.Plan)
		want string
	}{
		{"no steps", func(p *cudaagreement.Plan) { p.Steps = nil }, "no steps"},
		{"no name", func(p *cudaagreement.Plan) { p.Steps[0].Name = "" }, "has no name"},
		{"no artifact", func(p *cudaagreement.Plan) { p.Steps[1].Artifact = nil }, "has no artifact"},
		{"no output", func(p *cudaagreement.Plan) { p.Steps[1].Output = "" }, "has no output"},
		{"zero grid", func(p *cudaagreement.Plan) { p.Steps[2].Grid = 0 }, "geometry is 0x256"},
		{"extra input", func(p *cudaagreement.Plan) { p.Steps[0].Inputs = []string{"x", "w", "x"} }, "takes 3 parameters, want 4"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := fixturePlan(t, "sm_121a")
			tt.edit(p)
			err := p.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate = %v, want error containing %q", err, tt.want)
			}
		})
	}
}

func TestRecordExecutedOn(t *testing.T) {
	r := record("graph", cudaplan.BackendGraph, cudaplan.BackendGraph, []byte{1}, "mul", "add").Record
	if err := r.ExecutedOn(cudaplan.BackendGraph); err != nil {
		t.Fatalf("ExecutedOn: %v", err)
	}
	if err := r.ExecutedOn(cudaplan.BackendLaunch); err == nil {
		t.Fatal("ExecutedOn(Launch) accepted a record of Graph steps")
	}
	r.Steps[0].Executed = cudaplan.BackendLaunch
	r.Steps[0].Declined = "forced"
	if err := r.ExecutedOn(cudaplan.BackendGraph); err == nil {
		t.Fatal("ExecutedOn accepted a declined step")
	}
	if d := r.Declined(); len(d) != 1 || d[0].Step != "mul" {
		t.Fatalf("Declined = %v, want one entry for mul", d)
	}
	if err := (*cudaagreement.Record)(nil).ExecutedOn(cudaplan.BackendGraph); err == nil {
		t.Fatal("ExecutedOn accepted a nil record")
	}
}
