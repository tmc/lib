package cudalineinfo_test

import (
	"fmt"
	"reflect"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalineinfo"
)

func TestParseUnitCoordinateOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	limit := fmt.Sprintf("#line %d \"a.go\"\n", maxInt)
	oversized := "#line " + strconv.FormatUint(uint64(maxInt)+1, 10) + " \"b.go\""
	malformed := "#line nope \"b.go\""
	type origin struct {
		file string
		line int
	}
	for _, tt := range []struct {
		name, input, source string
		want                map[int]origin
	}{
		{"last coordinate", limit + "x", "x", map[int]origin{1: {"a.go", maxInt}}},
		{"end mapping", limit + "x\ny", "x\ny", map[int]origin{1: {"a.go", maxInt}}},
		{"continuation to limit", fmt.Sprintf("#line %d \"a.go\"\nx\ny\nz", maxInt-1), "x\ny\nz", map[int]origin{1: {"a.go", maxInt - 1}, 2: {"a.go", maxInt}}},
		{"directive reset", limit + "x\ny\n#line 7 \"b.go\"\nz\nw", "x\ny\nz\nw", map[int]origin{1: {"a.go", maxInt}, 3: {"b.go", 7}, 4: {"b.go", 8}}},
		{"immediate reset", limit + "x\n#line 7 \"b.go\"\ny", "x\ny", map[int]origin{1: {"a.go", maxInt}, 2: {"b.go", 7}}},
		{"generated reset", limit + "x\n#line 1 \"boundary.cu\"\ny", "x\ny", map[int]origin{1: {"a.go", maxInt}}},
		{"generated path reset", limit + "x\n#line 1 \"/dir/boundary.cu\"\ny", "x\ny", map[int]origin{1: {"a.go", maxInt}}},
		{"malformed retains mapping", "#line 7 \"a.go\"\nx\n" + malformed + "\ny", "x\n" + malformed + "\ny", map[int]origin{1: {"a.go", 7}, 2: {"a.go", 8}, 3: {"a.go", 9}}},
		{"oversized retains mapping", "#line 7 \"a.go\"\nx\n" + oversized + "\ny", "x\n" + oversized + "\ny", map[int]origin{1: {"a.go", 7}, 2: {"a.go", 8}, 3: {"a.go", 9}}},
		{"malformed after limit", limit + "x\n" + malformed + "\ny", "x\n" + malformed + "\ny", map[int]origin{1: {"a.go", maxInt}}},
		{"oversized after limit", limit + "x\n" + oversized + "\ny", "x\n" + oversized + "\ny", map[int]origin{1: {"a.go", maxInt}}},
		{"oversized without mapping", oversized + "\ny", oversized + "\ny", map[int]origin{}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			u := cudalineinfo.ParseUnit("boundary", tt.input)
			if u.Source != tt.source {
				t.Fatalf("Source = %q, want %q", u.Source, tt.source)
			}
			got := map[int]origin{}
			for n, loc := range u.LineMap {
				if loc.Line <= 0 {
					t.Fatalf("line %d has nonpositive coordinate %d", n, loc.Line)
				}
				got[n] = origin{loc.File, loc.Line}
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("LineMap = %v, want %v", got, tt.want)
			}
			printed := u.SourceWithLineDirectives()
			again := cudalineinfo.ParseUnit(u.Name, printed)
			if again.Name != u.Name || again.Source != u.Source || !reflect.DeepEqual(again.LineMap, u.LineMap) {
				t.Fatalf("typed round trip changed unit\nfirst: %#v\nsecond: %#v\nprinted: %q", u, again, printed)
			}
		})
	}
}
