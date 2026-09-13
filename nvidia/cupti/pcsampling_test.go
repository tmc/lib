package cupti

import "testing"

func TestPCSamplingReasonLayoutFor(t *testing.T) {
	for _, version := range []uint32{27, 130001, 130200, 130400} {
		layout, ok := PCSamplingReasonLayoutFor(version)
		if !ok {
			t.Fatalf("PCSamplingReasonLayoutFor(%d) rejected validated layout", version)
		}
		if layout.CountParamsSize != 32 || layout.NamesParamsSize != 48 {
			t.Fatalf("PCSamplingReasonLayoutFor(%d) = %+v, want 32/48", version, layout)
		}
	}
	if layout, ok := PCSamplingReasonLayoutFor(130401); ok || layout != (PCSamplingReasonLayout{}) {
		t.Fatalf("PCSamplingReasonLayoutFor(130401) = %+v, %v, want zero, false", layout, ok)
	}
}
