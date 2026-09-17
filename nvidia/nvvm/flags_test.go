package nvvm

import "testing"

func TestValidateOption(t *testing.T) {
	for _, tc := range []struct {
		flag string
		ok   bool
	}{
		{"-g", true},
		{"-gen-lto", true},
		{"-opt=0", true},
		{"-opt=3", true},
		{"-opt=2", false},
		{"-arch=compute_75", true},
		{"-arch=compute_80", true},
		{"-arch=compute_90a", true},
		{"-arch=compute_100f", true},
		{"-arch=sm_90", false},
		{"-arch=compute_x", false},
		{"-ftz=0", true},
		{"-ftz=1", true},
		{"-ftz=2", false},
		{"-prec-sqrt=1", true},
		{"-prec-div=0", true},
		{"-fma=1", true},
		{"-jump-table-density=101", true},
		{"-jump-table-density=0", true},
		{"-jump-table-density=102", false},
		{"-jump-table-density=-1", false},
		{"-g=1", false},
		{"-gen-lto=yes", false},
		{"-opt", false},
		{"", false},
		{"-unknown", false},
	} {
		err := ValidateOption(tc.flag)
		if (err == nil) != tc.ok {
			t.Errorf("ValidateOption(%q) = %v, want ok=%t", tc.flag, err, tc.ok)
		}
	}
}

func TestComputeArch(t *testing.T) {
	if got := ComputeArch(8, 0); got != "compute_80" {
		t.Errorf("ComputeArch(8, 0) = %q, want compute_80", got)
	}
	if got := ComputeArch(10, 0); got != "compute_100" {
		t.Errorf("ComputeArch(10, 0) = %q, want compute_100", got)
	}
}
