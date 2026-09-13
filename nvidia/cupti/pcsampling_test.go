package cupti

import (
	"testing"
	"unsafe"
)

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

func TestPCSamplingStallReasons(t *testing.T) {
	inv, err := pcSamplingStallReasons(1, 4, 130400,
		func(p *PCSamplingGetNumStallReasonsParams) error {
			if p.Ctx != 1 || p.NumStallReasons == 0 {
				t.Fatalf("count params = %+v", *p)
			}
			*(*uint)(unsafe.Pointer(p.NumStallReasons)) = 2
			return nil
		},
		func(p *PCSamplingGetStallReasonsParams) error {
			indices := unsafe.Slice((*uint32)(unsafe.Pointer(p.StallReasonIndex)), 2)
			pointers := unsafe.Slice((*uintptr)(unsafe.Pointer(p.StallReasons)), 2)
			indices[0], indices[1] = 9, 3
			copy((*[StallReasonStringSize]byte)(unsafe.Pointer(pointers[0]))[:], []byte("late\x00"))
			copy((*[StallReasonStringSize]byte)(unsafe.Pointer(pointers[1]))[:], []byte("early\x00"))
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if inv.CUPTIVersion != 130400 || len(inv.Reasons) != 2 || inv.Reasons[0].Index != 3 || inv.Reasons[0].Name != "early" {
		t.Fatalf("inventory = %+v", inv)
	}
}
