package cudaplan

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func policyArtifact(t *testing.T, target string) *cudakernel.Artifact {
	t.Helper()
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX,
		[]byte(".version 8.0\n.target "+target+"\n.address_size 64\n.visible .entry policy() { ret; }\n"),
		"policy", cudakernel.Schema{}, target)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestBackendPolicyRegions(t *testing.T) {
	base := BackendRequest{Kind: cudair.KindUnary, Region: Region{1, 2}, StepIndex: 1, StepCount: 3}
	got, err := SelectBackend(base)
	if err != nil || got.Region != base.Region || got.StepIndex != 1 || got.Backend != BackendLaunch {
		t.Fatalf("ordinary decision=%+v error=%v", got, err)
	}
	for _, test := range []struct {
		name string
		edit func(*BackendRequest)
	}{
		{"zero count", func(r *BackendRequest) { r.StepCount = 0 }},
		{"negative count", func(r *BackendRequest) { r.StepCount = -1 }},
		{"negative start", func(r *BackendRequest) { r.Region.Start = -1 }},
		{"empty", func(r *BackendRequest) { r.Region.End = r.Region.Start }},
		{"out of range", func(r *BackendRequest) { r.Region.End = 4 }},
		{"step before region", func(r *BackendRequest) { r.StepIndex = 0 }},
		{"step after region", func(r *BackendRequest) { r.StepIndex = 2 }},
		{"ordinary multi step", func(r *BackendRequest) { r.Region = Region{0, 3} }},
		{"finite partial region", func(r *BackendRequest) {
			r.FiniteMegaRequested, r.FiniteMegaAdmitted, r.Artifact = true, true, policyArtifact(t, "sm_90")
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			r := base
			test.edit(&r)
			if d, err := SelectBackend(r); err == nil || d.Backend != 0 {
				t.Fatalf("invalid candidate admitted: %+v, %v", d, err)
			}
		})
	}
	r := base
	r.Region = Region{0, 3}
	r.FiniteMegaRequested, r.FiniteMegaAdmitted, r.Artifact = true, true, policyArtifact(t, "sm_90")
	if d, err := SelectBackend(r); err != nil || d.Backend != BackendMega || d.Region != r.Region {
		t.Fatalf("whole finite region refused: %+v, %v", d, err)
	}
}

func TestBackendPolicyTargets(t *testing.T) {
	r := BackendRequest{Kind: cudair.KindUnary, Region: Region{0, 1}, StepCount: 1,
		DeviceTarget: "sm_90", Artifact: policyArtifact(t, "sm_90")}
	d, err := SelectBackend(r)
	if err != nil || !d.TargetChecked || d.DeviceTarget != "sm_90" {
		t.Fatalf("compatible declared target: %+v, %v", d, err)
	}
	r.DeviceTarget = "sm_121"
	r.Region = Region{7, 8}
	if d.DeviceTarget != "sm_90" || d.Region != (Region{0, 1}) {
		t.Fatal("decision shares mutable request facts")
	}
	r.Region = Region{0, 1}
	r.DeviceTarget = "sm_90"
	r.Artifact = policyArtifact(t, "sm_121")
	if _, err := SelectBackend(r); err == nil {
		t.Fatal("incompatible target admitted")
	}
	r.DeviceTarget = ""
	if d, err := SelectBackend(r); err != nil || d.TargetChecked || d.DeviceTarget != "" {
		t.Fatalf("unknown device invented compatibility: %+v, %v", d, err)
	}
	r.DeviceTarget = "not-a-target"
	if _, err := SelectBackend(r); err == nil {
		t.Fatal("invalid supplied device target admitted")
	}
	r.DeviceTarget, r.Artifact = "sm_90", nil
	if d, err := SelectBackend(r); err != nil || d.TargetChecked {
		t.Fatalf("no artifact invented a compatibility check: %+v, %v", d, err)
	}
}

func TestBackendPolicyOrdinaryFallback(t *testing.T) {
	// Worker capability alone does not request worker dispatch. The ordinary
	// candidate remains one step and its recorded decision says Launch.
	r := BackendRequest{Kind: cudair.KindUnary, Region: Region{1, 2}, StepIndex: 1, StepCount: 2,
		FiniteMegaAdmitted: true, Artifact: policyArtifact(t, "sm_90"), DeviceTarget: "sm_90"}
	d, err := SelectBackend(r)
	if err != nil || d.Backend != BackendLaunch || !strings.Contains(d.Reason, "ordinary") || !d.TargetChecked {
		t.Fatalf("ordinary fallback decision=%+v error=%v", d, err)
	}
}

func TestCompileBackendPolicyRegions(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	a := g.AddUnary("neg", x)
	y := g.AddUnary("relu", a)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	for _, mega := range []bool{false, true} {
		p, err := Compile(snap, WithFiniteMega(mega))
		if err != nil {
			t.Fatal(err)
		}
		ds := p.BackendDecisions()
		if len(ds) != 2 {
			t.Fatalf("decisions=%+v", ds)
		}
		for i, d := range ds {
			region, backend := (Region{i, i + 1}), BackendLaunch
			if mega {
				region, backend = Region{0, 2}, BackendMega
			}
			if d.Region != region || d.StepIndex != i || d.Backend != backend || d.DeviceTarget != "" || d.TargetChecked {
				t.Fatalf("mega=%v step=%d decision=%+v", mega, i, d)
			}
		}
		ds[0].Region.End = 100
		if p.BackendDecisions()[0].Region.End == 100 {
			t.Fatal("plan decisions expose owned slice")
		}
		if mega && p.megaLowering.Len() != 2 {
			t.Fatal("selected whole region did not reach lowering")
		}
		if err := p.Close(); err != nil {
			t.Fatal(err)
		}
	}
	var p *ExecutablePlan
	if p.BackendDecisions() != nil {
		t.Fatal("nil plan has decisions")
	}
}
