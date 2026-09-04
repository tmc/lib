package cudakv

import "testing"

func TestCacheAdvanceBounds(t *testing.T) {
	for _, tt := range []struct {
		name    string
		pos     int
		n       int
		closed  bool
		want    int
		wantErr bool
	}{
		{"zero", 3, 0, false, 3, false},
		{"advance", 3, 2, false, 5, false},
		{"exact capacity", 3, 5, false, 8, false},
		{"negative", 3, -1, false, 3, true},
		{"past capacity", 3, 6, false, 3, true},
		{"overflow", 3, int(^uint(0) >> 1), false, 3, true},
		{"full", 8, 1, false, 8, true},
		{"closed", 3, 1, true, 3, true},
		{"closed zero", 3, 0, true, 3, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := &Cache{maxSeqLen: 8, pos: tt.pos}
			if !tt.closed {
				c.layers = []LayerCache{}
			}
			err := c.Advance(tt.n)
			if (err != nil) != tt.wantErr || c.Pos() != tt.want {
				t.Fatalf("Advance(%d): pos=%d err=%v, want pos=%d error=%v", tt.n, c.Pos(), err, tt.want, tt.wantErr)
			}
		})
	}
}
