package cudapdl_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapdl"
)

func TestPDLRequiresOwnerDeclarations(t *testing.T) {
	for _, tc := range []struct {
		name               string
		producer, consumer cudapdl.KernelDesc
	}{
		{"line comments", cudapdl.KernelDesc{Source: "// griddepcontrol.launch_dependents"}, cudapdl.KernelDesc{Source: "// griddepcontrol.wait"}},
		{"block comments", cudapdl.KernelDesc{Source: "/* griddepcontrol.launch_dependents; */"}, cudapdl.KernelDesc{Source: "/* griddepcontrol.wait; */"}},
		{"quoted strings", cudapdl.KernelDesc{Source: `const char *s = "griddepcontrol.launch_dependents";`}, cudapdl.KernelDesc{Source: `const char *s = "griddepcontrol.wait";`}},
		{"PTX text", cudapdl.KernelDesc{Source: "griddepcontrol.launch_dependents;"}, cudapdl.KernelDesc{Source: "griddepcontrol.wait;"}},
		{"binary strings", cudapdl.KernelDesc{Artifact: &cudakernel.Artifact{Kind: cudakernel.KindCUBIN, Target: "sm_90", Raw: []byte("griddepcontrol.launch_dependents")}}, cudapdl.KernelDesc{Artifact: &cudakernel.Artifact{Kind: cudakernel.KindCUBIN, Target: "sm_90", Raw: []byte("griddepcontrol.wait")}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := cudapdl.CheckEligibility("sm_90", tc.producer, tc.consumer); got.Eligible || got.Policy != cudapdl.PolicyStandardFallback || got.FallbackReason == "" {
				t.Fatalf("undeclared pair admitted: %+v", got)
			}
		})
	}
	producer := cudapdl.KernelDesc{HasLaunchDependents: true}
	consumer := cudapdl.KernelDesc{HasWait: true}
	if got := cudapdl.CheckEligibility("sm_90", producer, consumer); !got.Eligible || got.Policy != cudapdl.PolicyProgrammatic {
		t.Fatalf("declared planning pair refused: %+v", got)
	}
	producer.HasLaunchDependents = false
	producer.Source = "// griddepcontrol.launch_dependents"
	if got := cudapdl.CheckEligibility("sm_90", producer, consumer); got.Eligible {
		t.Fatalf("comment replaced missing producer declaration: %+v", got)
	}
	producer.HasLaunchDependents = true
	consumer.HasWait = false
	consumer.Source = "// griddepcontrol.wait"
	if got := cudapdl.CheckEligibility("sm_90", producer, consumer); got.Eligible {
		t.Fatalf("comment replaced missing consumer declaration: %+v", got)
	}
}

func TestPDLArtifactTargetIdentity(t *testing.T) {
	for _, tc := range []struct {
		name, target       string
		producer, consumer *cudakernel.Artifact
		want               bool
	}{
		{"planning without artifacts", "sm_90", nil, nil, true},
		{"missing planning target", "", nil, nil, false},
		{"matching", "sm_90", &cudakernel.Artifact{Target: "sm_90"}, &cudakernel.Artifact{Target: "sm_90"}, true},
		{"canonical alias", "compute_90", &cudakernel.Artifact{Target: "sm_90"}, &cudakernel.Artifact{Target: "compute_90"}, true},
		{"normalized", " SM_90 ", &cudakernel.Artifact{Target: " sm_90 "}, nil, true},
		{"inferred matching", "", &cudakernel.Artifact{Target: "sm_90"}, &cudakernel.Artifact{Target: "sm_90"}, true},
		{"inferred from consumer", "", nil, &cudakernel.Artifact{Target: "sm_90"}, true},
		{"legacy producer", "sm_90", &cudakernel.Artifact{Target: "sm_80"}, nil, false},
		{"legacy consumer", "sm_90", nil, &cudakernel.Artifact{Target: "sm_80"}, false},
		{"other producer", "sm_90", &cudakernel.Artifact{Target: "sm_100"}, nil, false},
		{"other consumer", "sm_90", nil, &cudakernel.Artifact{Target: "sm_100"}, false},
		{"suffix mismatch", "sm_90", &cudakernel.Artifact{Target: "sm_90a"}, nil, false},
		{"missing producer target", "sm_90", &cudakernel.Artifact{}, nil, false},
		{"missing consumer target", "sm_90", nil, &cudakernel.Artifact{}, false},
		{"invalid producer target", "sm_90", &cudakernel.Artifact{Target: "invalid"}, nil, false},
		{"conflicting inferred targets", "", &cudakernel.Artifact{Target: "sm_90"}, &cudakernel.Artifact{Target: "sm_100"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			producer := cudapdl.KernelDesc{HasLaunchDependents: true, Artifact: tc.producer}
			consumer := cudapdl.KernelDesc{HasWait: true, Artifact: tc.consumer}
			got := cudapdl.CheckEligibility(tc.target, producer, consumer)
			if got.Eligible != tc.want {
				t.Fatalf("eligible=%v, want %v: %+v", got.Eligible, tc.want, got)
			}
			if !tc.want && (got.Policy != cudapdl.PolicyStandardFallback || got.FallbackReason == "") {
				t.Fatalf("refusal omitted policy or reason: %+v", got)
			}
		})
	}
}
