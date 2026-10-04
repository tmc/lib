package cuptiversion

import "testing"

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
