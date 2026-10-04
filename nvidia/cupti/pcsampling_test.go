package cupti

import (
	"strings"
	"testing"
	"unsafe"
)

func TestPCSamplingReasonLayoutFor(t *testing.T) {
	for _, version := range []uint32{24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400} {
		layout, ok := PCSamplingReasonLayoutFor(version)
		if !ok {
			t.Fatalf("PCSamplingReasonLayoutFor(%d) rejected validated layout", version)
		}
		if layout.CountParamsSize != 32 || layout.NamesParamsSize != 48 {
			t.Fatalf("PCSamplingReasonLayoutFor(%d) = %+v, want 32/48", version, layout)
		}
	}
	// A later release in the same major uses the newest measured layout
	// below it.
	for _, version := range []uint32{25, 27, 29, 130002, 130100, 130201, 130401, 130500, 139999} {
		if layout, ok := PCSamplingReasonLayoutFor(version); !ok || layout.CountParamsSize != 32 || layout.NamesParamsSize != 48 {
			t.Fatalf("PCSamplingReasonLayoutFor(%d) = %+v, %v, want 32/48, true", version, layout, ok)
		}
	}
	// A new major, or a release older than any measured one, is refused.
	for _, version := range []uint32{0, 20, 120000, 130000, 140000} {
		if layout, ok := PCSamplingReasonLayoutFor(version); ok || layout != (PCSamplingReasonLayout{}) {
			t.Fatalf("PCSamplingReasonLayoutFor(%d) = %+v, %v, want zero, false", version, layout, ok)
		}
	}
}

// fakeReasonQueries stands in for CUPTI, which receives the output addresses
// as plain integers. The fake checks that each address field names the typed
// storage it was derived from and writes the reasons through that storage.
func fakeReasonQueries(t *testing.T, count uint, reasons []PCSamplingStallReasonInfo) (func(*PCSamplingGetNumStallReasonsParams, *uint) error, func(*PCSamplingGetStallReasonsParams, pcSamplingReasonStorage) error) {
	countQuery := func(p *PCSamplingGetNumStallReasonsParams, n *uint) error {
		if p.Ctx != 1 || p.Size != 32 || p.NumStallReasons != uintptr(unsafe.Pointer(n)) {
			t.Fatalf("count params = %+v, want NumStallReasons %#x", *p, uintptr(unsafe.Pointer(n)))
		}
		*n = count
		return nil
	}
	namesQuery := func(p *PCSamplingGetStallReasonsParams, s pcSamplingReasonStorage) error {
		if p.Ctx != 1 || p.Size != 48 || p.NumStallReasons != count {
			t.Fatalf("names params = %+v, want count %d", *p, count)
		}
		if len(s.indices) != int(count) || len(s.pointers) != int(count) || len(s.names) != int(count) {
			t.Fatalf("storage lengths = %d/%d/%d, want %d", len(s.indices), len(s.pointers), len(s.names), count)
		}
		if p.StallReasonIndex != uintptr(unsafe.Pointer(&s.indices[0])) {
			t.Fatalf("StallReasonIndex = %#x, want %#x", p.StallReasonIndex, uintptr(unsafe.Pointer(&s.indices[0])))
		}
		if p.StallReasons != uintptr(unsafe.Pointer(&s.pointers[0])) {
			t.Fatalf("StallReasons = %#x, want %#x", p.StallReasons, uintptr(unsafe.Pointer(&s.pointers[0])))
		}
		for i := range s.pointers {
			if s.pointers[i] != uintptr(unsafe.Pointer(&s.names[i][0])) {
				t.Fatalf("pointers[%d] = %#x, want %#x", i, s.pointers[i], uintptr(unsafe.Pointer(&s.names[i][0])))
			}
		}
		for i, r := range reasons {
			s.indices[i] = r.Index
			copy(s.names[i][:], r.Name+"\x00")
		}
		return nil
	}
	return countQuery, namesQuery
}

func TestPCSamplingStallReasons(t *testing.T) {
	countQuery, namesQuery := fakeReasonQueries(t, 2, []PCSamplingStallReasonInfo{{9, "late"}, {3, "early"}})
	inv, err := pcSamplingStallReasons(1, 4, 130400, countQuery, namesQuery)
	if err != nil {
		t.Fatal(err)
	}
	want := []PCSamplingStallReasonInfo{{3, "early"}, {9, "late"}}
	if inv.CUPTIVersion != 130400 || len(inv.Reasons) != len(want) || inv.Reasons[0] != want[0] || inv.Reasons[1] != want[1] {
		t.Fatalf("inventory = %+v, want %+v", inv, want)
	}
}

func TestPCSamplingStallReasonsRejects(t *testing.T) {
	tests := []struct {
		name    string
		count   uint
		reasons []PCSamplingStallReasonInfo
		want    string
	}{
		{"over limit", 5, nil, "exceeds limit"},
		{"unterminated", 1, []PCSamplingStallReasonInfo{{1, strings.Repeat("x", StallReasonStringSize)}}, "unterminated"},
		{"empty name", 1, []PCSamplingStallReasonInfo{{1, ""}}, "unterminated"},
		{"duplicate index", 2, []PCSamplingStallReasonInfo{{1, "a"}, {1, "b"}}, "duplicate"},
		{"duplicate name", 2, []PCSamplingStallReasonInfo{{1, "a"}, {2, "a"}}, "duplicate"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			countQuery, namesQuery := fakeReasonQueries(t, tt.count, tt.reasons)
			inv, err := pcSamplingStallReasons(1, 4, 130400, countQuery, namesQuery)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("pcSamplingStallReasons = %+v, %v, want error containing %q", inv, err, tt.want)
			}
		})
	}
}
