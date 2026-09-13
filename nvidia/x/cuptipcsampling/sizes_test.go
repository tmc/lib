package cuptipcsampling

import "testing"

// TestParamSizes pins each declared prefix to the size measured with sizeof(3)
// on the CUDA toolkits installed on the GB10 host. Every toolkit from 12.6
// through 13.4 reports the same values, so a change here means nvidia/cupti was
// regenerated against a header that moved a field and every call site that
// declares one of these sizes needs re-measuring before it is trusted.
func TestParamSizes(t *testing.T) {
	tests := []struct {
		name string
		got  uint
		want uint
	}{
		{"PCSamplingEnableParams", enableParamsSize(), 24},
		{"PCSamplingDisableParams", disableParamsSize(), 24},
		{"PCSamplingStartParams", startParamsSize(), 24},
		{"PCSamplingStopParams", stopParamsSize(), 24},
		{"PCSamplingGetDataParams", getDataParamsSize(), 32},
		{"PCSamplingConfigurationInfoParams", configurationInfoParamsSize(), 40},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Errorf("prefix size = %d, want %d", tt.got, tt.want)
			}
		})
	}
}
