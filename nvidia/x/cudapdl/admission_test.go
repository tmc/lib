package cudapdl_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapdl"
)

func mustConstructPTXArtifactWithRaw(t *testing.T, entry, target string, raw []byte) *cudakernel.Artifact {
	t.Helper()
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, raw, entry, cudakernel.Schema{}, target)
	if err != nil {
		t.Fatalf("NewArtifact(%s, %s): %v", entry, target, err)
	}
	return art
}

func TestPDLRequiresOwnerDeclarations(t *testing.T) {
	prodRawWithIntrinsic := mustConstructPTXArtifactWithRaw(t, "prod", "sm_90",
		[]byte(".version 8.0\n.target sm_90\n// griddepcontrol.launch_dependents\n.visible .entry prod()\n{\n\tret;\n}\n"))
	consRawWithIntrinsic := mustConstructPTXArtifactWithRaw(t, "cons", "sm_90",
		[]byte(".version 8.0\n.target sm_90\n// griddepcontrol.wait\n.visible .entry cons()\n{\n\tret;\n}\n"))

	for _, tc := range []struct {
		name               string
		producer, consumer cudapdl.KernelDesc
	}{
		{"line comments", cudapdl.KernelDesc{Source: "// griddepcontrol.launch_dependents"}, cudapdl.KernelDesc{Source: "// griddepcontrol.wait"}},
		{"block comments", cudapdl.KernelDesc{Source: "/* griddepcontrol.launch_dependents; */"}, cudapdl.KernelDesc{Source: "/* griddepcontrol.wait; */"}},
		{"quoted strings", cudapdl.KernelDesc{Source: `const char *s = "griddepcontrol.launch_dependents";`}, cudapdl.KernelDesc{Source: `const char *s = "griddepcontrol.wait";`}},
		{"PTX text", cudapdl.KernelDesc{Source: "griddepcontrol.launch_dependents;"}, cudapdl.KernelDesc{Source: "griddepcontrol.wait;"}},
		{"binary strings", cudapdl.KernelDesc{Artifact: prodRawWithIntrinsic}, cudapdl.KernelDesc{Artifact: consRawWithIntrinsic}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := cudapdl.CheckEligibility("sm_90", tc.producer, tc.consumer); got.Eligible || got.Policy != cudapdl.PolicyStandardFallback || got.FallbackReason == "" {
				t.Fatalf("undeclared pair admitted: %+v", got)
			}
		})
	}
	producer := testProducerDesc("prod")
	consumer := testConsumerDesc("cons")
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

	// Trigger/wait boolean declarations without phase effect/resource contracts
	// must also be refused even if Source contains griddepcontrol intrinsics.
	declOnlyProd := cudapdl.KernelDesc{
		Name:                "decl_only_prod",
		Source:              "griddepcontrol.launch_dependents;",
		HasLaunchDependents: true,
	}
	declOnlyCons := cudapdl.KernelDesc{
		Name:    "decl_only_cons",
		Source:  "griddepcontrol.wait;",
		HasWait: true,
	}
	if got := cudapdl.CheckEligibility("sm_90", declOnlyProd, declOnlyCons); got.Eligible {
		t.Fatalf("declaration-only pair without effects/resources admitted: %+v", got)
	}
}

func TestPDLArtifactTargetIdentity(t *testing.T) {
	artSM90 := mustConstructPTXArtifact(t, "k", "sm_90")
	artCompute90 := mustConstructPTXArtifact(t, "k", "compute_90")
	artSM80 := mustConstructPTXArtifact(t, "k", "sm_80")
	artSM100 := mustConstructPTXArtifact(t, "k", "sm_100")
	artSM90a := mustConstructPTXArtifact(t, "k", "sm_90a")

	for _, tc := range []struct {
		name, target       string
		producer, consumer *cudakernel.Artifact
		want               bool
	}{
		{"planning without artifacts", "sm_90", nil, nil, true},
		{"missing planning target", "", nil, nil, false},
		{"matching", "sm_90", artSM90, artSM90, true},
		{"canonical alias", "compute_90", artSM90, artCompute90, true},
		{"normalized", " SM_90 ", artSM90, nil, true},
		{"inferred matching", "", artSM90, artSM90, true},
		{"inferred from consumer", "", nil, artSM90, true},
		{"legacy producer", "sm_90", artSM80, nil, false},
		{"legacy consumer", "sm_90", nil, artSM80, false},
		{"other producer", "sm_90", artSM100, nil, false},
		{"other consumer", "sm_90", nil, artSM100, false},
		{"suffix mismatch", "sm_90", artSM90a, nil, false},
		{"missing producer target", "sm_90", &cudakernel.Artifact{}, nil, false},
		{"missing consumer target", "sm_90", nil, &cudakernel.Artifact{}, false},
		{"conflicting inferred targets", "", artSM90, artSM100, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			producer := testProducerDesc("prod")
			producer.Artifact = tc.producer
			consumer := testConsumerDesc("cons")
			consumer.Artifact = tc.consumer
			got := cudapdl.CheckEligibility(tc.target, producer, consumer)
			if got.Eligible != tc.want {
				t.Fatalf("eligible=%v, want %v: %+v", got.Eligible, tc.want, got)
			}
			if !tc.want && (got.Policy != cudapdl.PolicyStandardFallback || got.FallbackReason == "") {
				t.Fatalf("refusal omitted policy or reason: %+v", got)
			}
		})
	}

	t.Run("invalid target rejected at artifact construction and target validator", func(t *testing.T) {
		if _, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_90\n.visible .entry k() { ret; }\n"), "k", cudakernel.Schema{}, "invalid"); err == nil {
			t.Fatal("expected NewArtifact to reject invalid target")
		}
		producer := testProducerDesc("prod")
		consumer := testConsumerDesc("cons")
		got := cudapdl.CheckEligibility("invalid", producer, consumer)
		if got.Eligible || got.Policy != cudapdl.PolicyStandardFallback || got.FallbackReason == "" {
			t.Fatalf("expected invalid target fallback, got %+v", got)
		}
	})
}
