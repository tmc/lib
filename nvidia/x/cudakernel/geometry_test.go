package cudakernel

import "testing"

// TestRecordedGeometry pins the geometry of each recorded target, so that
// reading it from cudadevicetable cannot move a value.
func TestRecordedGeometry(t *testing.T) {
	tests := []struct {
		arch                   string
		major, minor           int
		suffix                 string
		smem, optin            uint32
		warps, blocks, quantum uint32
	}{
		{"sm_90", 9, 0, "", 233472, 232448, 64, 32, 256},
		{"sm_90a", 9, 0, "a", 233472, 232448, 64, 32, 256},
		{"sm_100", 10, 0, "", 233472, 232448, 64, 32, 256},
		{"sm_100a", 10, 0, "a", 233472, 232448, 64, 32, 256},
		{"sm_121", 12, 1, "", 102400, 101376, 48, 24, 256},
		{"sm_121a", 12, 1, "a", 102400, 101376, 48, 24, 256},
	}
	for _, tt := range tests {
		info, err := ProbedTargetSM(tt.arch)
		if err != nil {
			t.Errorf("%s: %v", tt.arch, err)
			continue
		}
		got := TargetInfo{Arch: info.Arch, Major: info.Major, Minor: info.Minor, Suffix: info.Suffix,
			MaxSharedMemoryBytes: info.MaxSharedMemoryBytes, MaxOptInSharedBytes: info.MaxOptInSharedBytes,
			WarpsPerSM: info.WarpsPerSM, MaxBlocksPerSM: info.MaxBlocksPerSM, RegisterQuantum: info.RegisterQuantum}
		want := TargetInfo{Arch: tt.arch, Major: tt.major, Minor: tt.minor, Suffix: tt.suffix,
			MaxSharedMemoryBytes: tt.smem, MaxOptInSharedBytes: tt.optin,
			WarpsPerSM: tt.warps, MaxBlocksPerSM: tt.blocks, RegisterQuantum: tt.quantum}
		if got != want {
			t.Errorf("%s:\n got %+v\nwant %+v", tt.arch, got, want)
		}
	}
}
