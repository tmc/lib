package cuptipcsampling

import (
	"errors"
	"reflect"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func nativePointer(address uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&address))
}

func countTwo(t *testing.T) func(*cupti.PCSamplingGetNumStallReasonsParams) error {
	return func(p *cupti.PCSamplingGetNumStallReasonsParams) error {
		if p.Size != 32 || p.Ctx != 1 || p.PPriv != 0 {
			t.Fatal("invalid count ABI")
		}
		*(*uint)(nativePointer(p.NumStallReasons)) = 2
		return nil
	}
}

func fillNames(t *testing.T, p *cupti.PCSamplingGetStallReasonsParams) [][]byte {
	t.Helper()
	if p.Size != 48 || p.Ctx != 1 || p.NumStallReasons != 2 || p.PPriv != 0 {
		t.Fatal("invalid names ABI")
	}
	indices := unsafe.Slice((*uint32)(nativePointer(p.StallReasonIndex)), 2)
	indices[0], indices[1] = 9, 3
	pointers := unsafe.Slice((*uintptr)(nativePointer(p.StallReasons)), 2)
	names := [][]byte{unsafe.Slice((*byte)(nativePointer(pointers[0])), 128), unsafe.Slice((*byte)(nativePointer(pointers[1])), 128)}
	copy(names[0], "selected_not_issued\x00")
	copy(names[1], "selected\x00")
	return names
}

func TestStallReasons(t *testing.T) {
	for _, version := range []uint32{27, 130001, 130200, 130400} {
		var native [][]byte
		out, err := stallReasons(1, 10, version, countTwo(t), func(p *cupti.PCSamplingGetStallReasonsParams) error { native = fillNames(t, p); return nil })
		want := []StallReason{{Index: 3, Name: "selected"}, {Index: 9, Name: "selected_not_issued"}}
		if err != nil || out.CUPTIVersion != version || !reflect.DeepEqual(out.Reasons, want) {
			t.Fatalf("inventory=%v err=%v", out, err)
		}
		native[0][0] = 'X'
		if out.Reasons[1].Name != "selected_not_issued" {
			t.Fatal("name aliases native output")
		}
	}
}

func TestStallReasonBounds(t *testing.T) {
	for _, tt := range []struct {
		name    string
		ctx     cupti.Context
		limit   int
		version uint32
	}{
		{"nil context", 0, 2, 27}, {"zero limit", 1, 0, 27}, {"negative limit", 1, -1, 27}, {"large limit", 1, 65537, 27}, {"unknown ABI", 1, 2, 130201},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := stallReasons(tt.ctx, tt.limit, tt.version, func(*cupti.PCSamplingGetNumStallReasonsParams) error { t.Fatal("queried invalid request"); return nil }, nil)
			if err == nil {
				t.Fatal("invalid request accepted")
			}
		})
	}
	if _, err := stallReasons(1, 1, 27, countTwo(t), func(*cupti.PCSamplingGetStallReasonsParams) error { t.Fatal("queried oversized result"); return nil }); err == nil {
		t.Fatal("count bound ignored")
	}
	if _, err := stallReasons(1, 10, 27, func(*cupti.PCSamplingGetNumStallReasonsParams) error { return nil }, nil); err == nil {
		t.Fatal("unwritten count accepted")
	}
	out, err := stallReasons(1, 10, 27, func(p *cupti.PCSamplingGetNumStallReasonsParams) error {
		*(*uint)(nativePointer(p.NumStallReasons)) = 0
		return nil
	}, nil)
	if err != nil || out.CUPTIVersion != 27 || len(out.Reasons) != 0 {
		t.Fatal("empty inventory not retained")
	}
}

func TestMalformedReasons(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*cupti.PCSamplingGetStallReasonsParams, [][]byte)
	}{
		{"empty", func(_ *cupti.PCSamplingGetStallReasonsParams, n [][]byte) { n[0][0] = 0 }},
		{"unterminated", func(_ *cupti.PCSamplingGetStallReasonsParams, n [][]byte) {
			for i := range n[0] {
				n[0][i] = 'x'
			}
		}},
		{"duplicate name", func(_ *cupti.PCSamplingGetStallReasonsParams, n [][]byte) { copy(n[0], "selected\x00") }},
		{"duplicate ID", func(p *cupti.PCSamplingGetStallReasonsParams, _ [][]byte) {
			v := unsafe.Slice((*uint32)(nativePointer(p.StallReasonIndex)), 2)
			v[0] = v[1]
		}},
		{"changed count", func(p *cupti.PCSamplingGetStallReasonsParams, _ [][]byte) { p.NumStallReasons = 3 }},
		{"changed table", func(p *cupti.PCSamplingGetStallReasonsParams, _ [][]byte) { p.StallReasons = 0 }},
		{"changed name pointer", func(p *cupti.PCSamplingGetStallReasonsParams, _ [][]byte) {
			unsafe.Slice((*uintptr)(nativePointer(p.StallReasons)), 2)[0] = 0
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out, err := stallReasons(1, 10, 27, countTwo(t), func(p *cupti.PCSamplingGetStallReasonsParams) error {
				n := fillNames(t, p)
				tt.change(p, n)
				return nil
			})
			if err == nil || len(out.Reasons) != 0 {
				t.Fatal("malformed inventory returned")
			}
		})
	}
	want := errors.New("native unsupported")
	if _, err := stallReasons(1, 10, 27, func(*cupti.PCSamplingGetNumStallReasonsParams) error { return want }, nil); !errors.Is(err, want) {
		t.Fatal("lost native count error")
	}
	if _, err := stallReasons(1, 10, 27, countTwo(t), func(*cupti.PCSamplingGetStallReasonsParams) error { return want }); !errors.Is(err, want) {
		t.Fatal("lost native names error")
	}
}
