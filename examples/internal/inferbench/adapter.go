package inferbench

import "fmt"

// AdapterCapabilities states whether an external runtime can execute the same
// serving workload and expose the evidence required by Campaign.Validate.
type AdapterCapabilities struct {
	PersistentCompiledRuntime bool `json:"persistent_compiled_runtime"`
	ExplicitPromptTokenIDs    bool `json:"explicit_prompt_token_ids"`
	GreedyArgmax              bool `json:"greedy_argmax"`
	ContinuousServing         bool `json:"continuous_serving"`
	QueueIncluded             bool `json:"queue_included"`
	FinalDeviceSynchronize    bool `json:"final_device_synchronize"`
	DeviceResidentKV          bool `json:"device_resident_kv"`
	DispatchCounters          bool `json:"dispatch_counters"`
	TierByteCounters          bool `json:"tier_byte_counters"`
	CompileAndWarmupExcluded  bool `json:"compile_and_warmup_excluded"`
	QualifiedNullControl      bool `json:"qualified_null_control"`
	RawSampleArtifacts        bool `json:"raw_sample_artifacts"`
	FullLogits                bool `json:"full_logits"`
	AllLayerPopulatedKV       bool `json:"all_layer_populated_kv"`
	ObservedOutputIncluded    bool `json:"observed_output_included_in_timing"`
}

// CheckAdapterOutputCapabilities rejects an adapter that cannot perform the
// declared output work inside the measured interval.
func CheckAdapterOutputCapabilities(arm, mode string, c AdapterCapabilities) error {
	if err := CheckAdapterCapabilities(arm, c); err != nil {
		return err
	}
	switch mode {
	case ThroughputOutputMode:
		return nil
	case ObservedOutputMode:
		checks := []struct {
			ok   bool
			name string
		}{
			{c.FullLogits, "full_logits"},
			{c.AllLayerPopulatedKV, "all_layer_populated_kv"},
			{c.ObservedOutputIncluded, "observed_output_included_in_timing"},
		}
		for _, check := range checks {
			if !check.ok {
				return fmt.Errorf("adapter capability %s unavailable", check.name)
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported output mode %q", mode)
	}
}

// AdapterRefusal is a machine-readable explanation that an arm cannot yet
// participate in a matched campaign.
type AdapterRefusal struct {
	SchemaVersion int    `json:"schema_version"`
	State         string `json:"state"`
	Arm           string `json:"arm"`
	Code          string `json:"code"`
	Reason        string `json:"reason"`
}

// CheckAdapterCapabilities returns the first missing matched-work property.
func CheckAdapterCapabilities(arm string, c AdapterCapabilities) error {
	checks := []struct {
		ok   bool
		name string
	}{
		{c.PersistentCompiledRuntime, "persistent_compiled_runtime"},
		{c.ExplicitPromptTokenIDs, "explicit_prompt_token_ids"},
		{c.GreedyArgmax, "greedy_argmax"},
		{c.ContinuousServing, "continuous_serving"},
		{c.QueueIncluded, "queue_included"},
		{c.FinalDeviceSynchronize, "final_device_synchronize"},
		{c.DeviceResidentKV, "device_resident_kv"},
		{c.DispatchCounters, "dispatch_counters"},
		{c.TierByteCounters, "tier_byte_counters"},
		{c.CompileAndWarmupExcluded, "compile_and_warmup_excluded"},
		{c.QualifiedNullControl, "qualified_null_control"},
		{c.RawSampleArtifacts, "raw_sample_artifacts"},
	}
	for _, check := range checks {
		if !check.ok {
			return fmt.Errorf("adapter capability %s unavailable", check.name)
		}
	}
	return nil
}
