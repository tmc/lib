package nvjitlink

import "testing"

func TestValidateOption(t *testing.T) {
	for _, tc := range []struct {
		flag string
		ok   bool
	}{
		{"-arch=sm_90", true},
		{"-arch=compute_100a", true},
		{"-lto", true},
		{"-ptx", true},
		{"-maxrregcount=64", true},
		{"-Xptxas=-v", true},
		{"-arch=sm_x", false},
		{"-maxrregcount=-1", false},
		{"-ptx=yes", false},
		{"-Werror", false},
	} {
		err := ValidateOption(tc.flag)
		if (err == nil) != tc.ok {
			t.Errorf("ValidateOption(%q) = %v, want ok=%t", tc.flag, err, tc.ok)
		}
	}
}
