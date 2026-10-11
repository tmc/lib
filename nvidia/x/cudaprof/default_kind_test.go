package cudaprof

import (
	"context"
	"github.com/tmc/lib/nvidia/cupti"
	"runtime/pprof"
	"testing"
)

// These assignments retain the exported signatures used by existing callers.
var (
	_ func(context.Context, pprof.LabelSet, func(context.Context)) = Do
	_ func(uint64) ([]string, bool)                                = ExternalStack
	_ func(uint64, []string)                                       = RecordExternalStack
)

func TestDefaultExternalStackNamespace(t *testing.T) {
	ResetExternalStacks()
	t.Cleanup(ResetExternalStacks)
	RecordExternalStack(9, []string{"default"})
	if !RecordExternalStackKind(cupti.ExternalCorrelationKindCustom1, 9, []string{"other"}) {
		t.Fatal("other namespace rejected")
	}
	got, ok := ExternalStack(9)
	if !ok || len(got) != 1 || got[0] != "default" {
		t.Fatalf("default = %v, %v", got, ok)
	}
	got[0] = "changed"
	got, _ = ExternalStackKind(cupti.ExternalCorrelationKindCustom0, 9)
	if got[0] != "default" {
		t.Fatalf("stack alias = %v", got)
	}
}
