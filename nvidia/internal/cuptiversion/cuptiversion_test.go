package cuptiversion

import (
	"fmt"
	"testing"
)

var qualified = []uint32{24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400}

func TestLayout(t *testing.T) {
	tests := []struct {
		v      uint32
		want   uint32
		wantOK bool
	}{
		{24, 24, true},
		{130400, 130400, true},
		{130401, 130400, true}, // newer patch
		{130500, 130400, true}, // newer minor
		{139999, 130400, true},
		{130201, 130200, true}, // between qualified releases
		{130100, 130001, true},
		{25, 24, true},
		{29, 28, true},
		{0, 0, false},
		{20, 0, false},     // CUDA 12.2, older than any qualified
		{130000, 0, false}, // older than any qualified 13.x
		{120000, 0, false}, // not a CUDA 12 numbering CUPTI uses
		{140000, 0, false}, // new major
	}
	for _, tt := range tests {
		got, ok := Layout(tt.v, qualified...)
		if got != tt.want || ok != tt.wantOK {
			t.Errorf("Layout(%d) = %d, %v; want %d, %v", tt.v, got, ok, tt.want, tt.wantOK)
		}
	}
}

func TestAdmittedLayout(t *testing.T) {
	for _, v := range []uint32{24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400, 130401} {
		want := v
		if v == 130401 {
			want = 130400
		}
		got, ok := AdmittedLayout(v, qualified...)
		if !ok || got != want {
			t.Fatalf("%d: %d %v", v, got, ok)
		}
	}
	for _, v := range []uint32{0, 20, 25, 27, 29, 999, 120901, 130000, 130002, 130003, 130100, 130201, 130300, 130402, 130500, 139999, 140000} {
		if got, ok := AdmittedLayout(v, qualified...); ok || got != 0 {
			t.Fatalf("admitted %d", v)
		}
	}
	if _, ok := AdmittedLayout(130401, 130301); ok {
		t.Fatal("alias without target")
	}
}

func ExampleAdmittedLayout() {
	layout, ok := AdmittedLayout(130401, 130400)
	fmt.Println(layout, ok)
	_, ok = AdmittedLayout(130500, 130400)
	fmt.Println(ok)
	// Output:
	// 130400 true
	// false
}

func TestAdmittedLayoutOrder(t *testing.T) {
	for _, rows := range [][]uint32{{130400, 24, 130001, 24}, {24, 130001, 24, 130400}} {
		if got, ok := AdmittedLayout(130401, rows...); !ok || got != 130400 {
			t.Fatalf("alias=%d,%v", got, ok)
		}
		if got, ok := AdmittedLayout(24, rows...); !ok || got != 24 {
			t.Fatalf("exact=%d,%v", got, ok)
		}
	}
}
