package cuptirange

import "testing"

func TestReplayModeSupport(t *testing.T) {
	for _, version := range []uint32{24, 26, 27, 28, 130001, 130101, 130200, 130201, 130202, 130301, 130400} {
		if err := ReplayModeSupport(version, UserReplay); err != nil {
			t.Fatal(err)
		}
		want := version == 130200 || version == 130202
		if err := ReplayModeSupport(version, KernelReplay); (err == nil) != want {
			t.Fatalf("version=%d error=%v", version, err)
		}
	}
	if ReplayModeSupport(130202, ReplayMode(2)) == nil {
		t.Fatal("accepted unknown replay")
	}
}
