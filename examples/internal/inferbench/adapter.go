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
	}
	for _, check := range checks {
		if !check.ok {
			return fmt.Errorf("adapter capability %s unavailable", check.name)
		}
	}
	return nil
}
