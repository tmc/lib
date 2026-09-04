package inferbench

import (
	"strings"
	"testing"
)

func TestCheckAdapterCapabilities(t *testing.T) {
	complete := AdapterCapabilities{
		PersistentCompiledRuntime: true,
		ExplicitPromptTokenIDs:    true,
		GreedyArgmax:              true,
		ContinuousServing:         true,
		QueueIncluded:             true,
		FinalDeviceSynchronize:    true,
		DeviceResidentKV:          true,
		DispatchCounters:          true,
		TierByteCounters:          true,
		CompileAndWarmupExcluded:  true,
		QualifiedNullControl:      true,
		RawSampleArtifacts:        true,
		PeakMemoryMeasured:        true,
	}
	if err := CheckAdapterCapabilities("arm", complete); err != nil {
		t.Fatal(err)
	}
	complete.ExplicitPromptTokenIDs = false
	if err := CheckAdapterCapabilities("arm", complete); err == nil || !strings.Contains(err.Error(), "explicit_prompt_token_ids") {
		t.Fatalf("error = %v", err)
	}
	complete.ExplicitPromptTokenIDs = true
	complete.PeakMemoryMeasured = false
	if err := CheckAdapterCapabilities("arm", complete); err == nil || !strings.Contains(err.Error(), "peak_memory_measured") {
		t.Fatalf("error = %v", err)
	}
}

func TestCheckAdapterOutputCapabilities(t *testing.T) {
	complete := AdapterCapabilities{
		PersistentCompiledRuntime: true, ExplicitPromptTokenIDs: true, GreedyArgmax: true,
		ContinuousServing: true, QueueIncluded: true, FinalDeviceSynchronize: true,
		DeviceResidentKV: true, DispatchCounters: true, TierByteCounters: true,
		CompileAndWarmupExcluded: true, QualifiedNullControl: true, RawSampleArtifacts: true,
		PeakMemoryMeasured: true,
	}
	if err := CheckAdapterOutputCapabilities("arm", ThroughputOutputMode, complete); err != nil {
		t.Fatal(err)
	}
	if err := CheckAdapterOutputCapabilities("arm", ObservedOutputMode, complete); err == nil || !strings.Contains(err.Error(), "full_logits") {
		t.Fatalf("error = %v", err)
	}
	complete.FullLogits = true
	complete.AllLayerPopulatedKV = true
	complete.ObservedOutputIncluded = true
	if err := CheckAdapterOutputCapabilities("arm", ObservedOutputMode, complete); err != nil {
		t.Fatal(err)
	}
}
