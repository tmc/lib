package dsv4expertcache

import (
	"fmt"
	"strings"
	"testing"
)

func TestParseBudget(t *testing.T) {
	for _, tt := range []struct {
		input string
		want  Budget
		valid bool
	}{
		{"1", Budget{Slots: 1}, true}, {"4294967295", Budget{Slots: 4294967295}, true},
		{"0002", Budget{Slots: 2}, true}, {"16GB", Budget{Bytes: 16 << 30}, true},
		{"1gB", Budget{Bytes: 1 << 30}, true}, {"1Gb", Budget{Bytes: 1 << 30}, true},
		{"17179869183gb", Budget{Bytes: 17179869183 << 30}, true},
		{"", Budget{}, false}, {"0", Budget{}, false}, {"0GB", Budget{}, false},
		{"4294967296", Budget{}, false}, {"17179869184GB", Budget{}, false},
		{"+1", Budget{}, false}, {"-1", Budget{}, false}, {" 1", Budget{}, false},
		{"1 ", Budget{}, false}, {"1.5GB", Budget{}, false}, {"1GiB", Budget{}, false},
		{"18446744073709551616", Budget{}, false},
		{"00000000000000000000000000000001GB", Budget{}, false},
	} {
		t.Run(tt.input, func(t *testing.T) {
			got, err := ParseBudget(tt.input)
			if (err == nil) != tt.valid || got != tt.want {
				t.Fatalf("got %+v, %v; want %+v valid=%v", got, err, tt.want, tt.valid)
			}
		})
	}
}

func TestSlotsForBytes(t *testing.T) {
	for _, tt := range []struct {
		bytes, size uint64
		want        uint32
	}{
		{0, 1, 0}, {1, 0, 0}, {9, 10, 0}, {29, 10, 2}, {4294967295, 1, 4294967295}, {4294967296, 1, 0},
	} {
		if got := SlotsForBytes(tt.bytes, tt.size); got != tt.want {
			t.Fatalf("%+v: got %d", tt, got)
		}
	}
}

func TestReadHotlist(t *testing.T) {
	for _, tt := range []struct {
		name, input string
		limit       uint32
		want        []HotEntry
		bad         bool
	}{
		{"empty", "# header\n\n", 4, nil, false},
		{"filter", "0 1 0\n9 0 3\n0 9 3\n0 1 4 ignored\n0 1 7\n1 2 4294967296 NaN\n", 4, []HotEntry{{0, 1, 4}, {1, 2, 4294967295}}, false},
		{"limit", "0 1 4\nbad\n", 1, []HotEntry{{0, 1, 4}}, false},
		{"signs-and-cast", "+0 +1 -1\n4294967296 2 5\n", 4, []HotEntry{{0, 1, 4294967295}, {0, 2, 5}}, false},
		{"missing", "0 1\n", 4, nil, true},
		{"overflow", "0 1 18446744073709551616\n", 4, nil, true},
		{"trailing", "0 1 2xyz\n", 4, []HotEntry{{0, 1, 2}}, false},
		{"negative-layer", "-1 0 1\n0 0 1\n", 4, []HotEntry{{0, 0, 1}}, false},
		{"negative-overflow", "0 0 -18446744073709551616\n", 4, nil, true},
		{"prefix-no-space", "0+1+2ignored\n", 4, []HotEntry{{0, 1, 2}}, false},
		{"unsigned-layer-overflow", "18446744073709551616 0 1\n", 4, nil, true},
		{"zero-before-duplicate", "0 0 0\n0 0 7\n", 4, []HotEntry{{0, 0, 7}}, false},
		{"chunk-continuation", "#" + strings.Repeat("x", 254) + "0 1 7\n", 4, []HotEntry{{0, 1, 7}}, false},
		{"fgets", "#" + strings.Repeat("x", 254) + "bad\n", 4, nil, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ReadHotlist(strings.NewReader(tt.input), 2, 3, tt.limit)
			if (err != nil) != tt.bad || fmt.Sprint(got) != fmt.Sprint(tt.want) {
				t.Fatalf("got %v, %v; want %v bad=%v", got, err, tt.want, tt.bad)
			}
		})
	}
}

func ExampleParseBudget() {
	b, _ := ParseBudget("2GB")
	fmt.Println(b.Bytes)
	// Output: 2147483648
}

func ExampleSlotsForBytes() {
	fmt.Println(SlotsForBytes(1024, 256))
	// Output: 4
}

func ExampleReadHotlist() {
	entries, _ := ReadHotlist(strings.NewReader("0 2 10 0.5\n"), 43, 256, 4)
	fmt.Println(entries)
	// Output: [{0 2 10}]
}
