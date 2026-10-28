package advise

import "testing"

func TestTargetSupports(t *testing.T) {
	tests := []struct {
		arch                 string
		tcgen05, wgmma, mxf4 bool
	}{
		{"sm_80", false, false, false},
		{"sm_86", false, false, false},
		{"sm_89", false, false, false},
		{"sm_90", false, true, false},
		{"sm_90a", false, true, false},
		{"sm_100", true, false, false},
		{"sm_100a", true, false, false},
		{"sm_103a", true, false, false},
		{"sm_110a", true, false, false},
		{"sm_120a", false, false, true},
		{"sm_121a", false, false, true},
		{"", false, false, false},
	}
	for _, tt := range tests {
		got := [3]bool{targetSupports(tt.arch, "tcgen05"), targetSupports(tt.arch, "wgmma"), targetSupports(tt.arch, "mmasyncmxf4")}
		if want := [3]bool{tt.tcgen05, tt.wgmma, tt.mxf4}; got != want {
			t.Errorf("%q: tcgen05, wgmma, mxf4 = %v, want %v", tt.arch, got, want)
		}
	}
}

func TestCubinArch(t *testing.T) {
	header := func(abi byte, flags uint32) []byte {
		b := make([]byte, 64)
		copy(b, "\x7fELF")
		b[4], b[8] = 2, abi
		b[48], b[49], b[50], b[51] = byte(flags), byte(flags>>8), byte(flags>>16), byte(flags>>24)
		return b
	}
	tests := []struct {
		data []byte
		want string
	}{
		// e_flags of cubins nvcc 13.4 wrote, ELF ABI version 8.
		{header(8, 0x6006702), "sm_103"},
		{header(8, 0x6007902), "sm_121"},
		{header(8, 0x6005a04), "sm_90"},
		{header(8, 0x6005004), "sm_80"},
		{header(7, 0x5a005a), "sm_90"},
		{[]byte("not elf"), ""},
	}
	for _, tt := range tests {
		if got := cubinArch(tt.data); got != tt.want {
			t.Errorf("cubinArch(%x) = %q, want %q", tt.data[:min(len(tt.data), 8)], got, tt.want)
		}
	}
}
