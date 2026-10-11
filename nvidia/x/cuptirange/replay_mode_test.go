package cuptirange

import "testing"

func TestReplayModeSupport(t *testing.T) {
	for _, tt := range []struct {
		version uint32
		kernel  bool
	}{
		{24, false}, {26, false}, {27, false}, {28, false},
		{130001, false}, {130101, false}, {130200, true}, {130202, true},
		{130301, false}, {130400, true},
		// Unlisted runtimes are refused.
		{130201, false}, {130300, false}, {130302, false}, {130401, true}, {130500, false},
	} {
		if err := ReplayModeSupport(tt.version, UserReplay); (err == nil) != qualifiedLayout(tt.version) {
			t.Fatal(err)
		}
		if err := ReplayModeSupport(tt.version, KernelReplay); (err == nil) != tt.kernel {
			t.Fatalf("version=%d error=%v", tt.version, err)
		}
	}
	if ReplayModeSupport(140000, KernelReplay) == nil {
		t.Fatal("accepted kernel replay on a new major")
	}
	if ReplayModeSupport(130202, ReplayMode(2)) == nil {
		t.Fatal("accepted unknown replay")
	}
}
