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
	}
	if err := CheckAdapterCapabilities("arm", complete); err != nil {
		t.Fatal(err)
	}
	complete.ExplicitPromptTokenIDs = false
	if err := CheckAdapterCapabilities("arm", complete); err == nil || !strings.Contains(err.Error(), "explicit_prompt_token_ids") {
		t.Fatalf("error = %v", err)
	}
}
