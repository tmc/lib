package cudablock

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

type topkCase struct {
	name             string
	c                TopKConfig
	input            []float32
	lengths, offsets []int32
}

type topkAnswer struct {
	indices, counts, status []int32
	values                  []float32
	resources               map[string]int
}

// Sorting an independently gathered row in float64 is deliberately different
// from the kernel's repeated selection and selected-index membership scans.
func topkSortReference(f topkCase) topkAnswer {
	a := topkAnswer{indices: make([]int32, f.c.Rows*f.c.K), values: make([]float32, f.c.Rows*f.c.K), counts: make([]int32, f.c.Rows), status: make([]int32, f.c.Rows)}
	for i := range a.indices {
		a.indices[i], a.values[i] = -1, float32(math.Inf(-1))
	}
	type entry struct {
		value float64
		index int
		bits  uint32
	}
	for row := 0; row < f.c.Rows; row++ {
		n, offset := f.lengths[row], f.offsets[row]
		if n < 0 || int(n) > f.c.Width {
			a.status[row] = int32(TopKBadLength)
			continue
		}
		if offset < 0 || n > 0 && int64(offset)+int64(n)-1 > math.MaxInt32 {
			a.status[row] = int32(TopKBadOffset)
			continue
		}
		var entries []entry
		for i := 0; i < int(n); i++ {
			v := f.input[row*f.c.RowStride+i]
			if math.IsNaN(float64(v)) {
				a.status[row] = int32(TopKNaN)
			}
			entries = append(entries, entry{float64(v), i, math.Float32bits(v)})
		}
		if a.status[row] != 0 {
			continue
		}
		sort.Slice(entries, func(i, j int) bool {
			if entries[i].value == entries[j].value {
				return entries[i].index < entries[j].index
			}
			return entries[i].value > entries[j].value
		})
		if len(entries) > f.c.K {
			entries = entries[:f.c.K]
		}
		a.counts[row] = int32(len(entries))
		for j, e := range entries {
			a.indices[row*f.c.K+j] = int32(int64(e.index) + int64(offset))
			a.values[row*f.c.K+j] = math.Float32frombits(e.bits)
		}
	}
	return a
}

func topkCompare(c TopKConfig, got, want topkAnswer) error {
	if !reflect.DeepEqual(got.status, want.status) || !reflect.DeepEqual(got.counts, want.counts) || !reflect.DeepEqual(got.indices, want.indices) {
		return errors.New("topk index/count/status disagreement")
	}
	if c.Values {
		if len(got.values) != len(want.values) {
			return errors.New("topk value length disagreement")
		}
		for i := range got.values {
			if math.Float32bits(got.values[i]) != math.Float32bits(want.values[i]) {
				return errors.New("topk value bits disagreement")
			}
		}
	}
	return nil
}

func topkFixtures() []topkCase {
	var fixtures []topkCase
	for _, n := range []int{0, 1, 31, 32, 33, 129, 4096} {
		for _, k := range []int{1, 32} {
			width := n
			if width == 0 {
				width = 1
			}
			f := topkCase{name: fmtTopKName(n, k), c: TopKConfig{Rows: 1, Width: width, RowStride: width + 3, K: k, Values: true}, input: make([]float32, width), lengths: []int32{int32(n)}, offsets: []int32{17}}
			for i := range f.input {
				f.input[i] = float32((i*37)%53) - 26
			}
			fixtures = append(fixtures, f)
		}
	}
	fixtures = append(fixtures,
		topkCase{"ties_zero_infinity", TopKConfig{1, 8, 8, 8, true}, []float32{float32(math.Inf(-1)), math.Float32frombits(0x80000000), 0, float32(math.Inf(1)), 7, 7, float32(math.Inf(-1)), float32(math.Inf(1))}, []int32{8}, []int32{0}},
		topkCase{"nan_short", TopKConfig{1, 3, 3, 32, true}, []float32{1, float32(math.NaN()), 4}, []int32{3}, []int32{0}},
		topkCase{"nan_long", TopKConfig{1, 3, 3, 1, false}, []float32{1, 4, float32(math.NaN())}, []int32{3}, []int32{0}},
		topkCase{"stride_padding", TopKConfig{4, 3, 5, 2, true}, []float32{2, 1, 3, 999, 999, 6, 999, 999, 999, 999, 7, 9, 999, 999, 999, 1, 0, -1}, []int32{3, 1, 2, 0}, []int32{100, 200, 300, 400}},
		topkCase{"indices_only", TopKConfig{2, 4, 4, 3, false}, []float32{1, 4, 2, 3, 8, 7, 6, 5}, []int32{4, 1}, []int32{2, 3}},
		topkCase{"offset_overflow", TopKConfig{1, 2, 2, 1, true}, []float32{2, 3}, []int32{2}, []int32{math.MaxInt32}},
		topkCase{"offset_limit", TopKConfig{1, 2, 2, 2, true}, []float32{2, 3}, []int32{2}, []int32{math.MaxInt32 - 1}},
		topkCase{"bad_length", TopKConfig{1, 2, 2, 1, true}, []float32{2, 3}, []int32{3}, []int32{0}},
		topkCase{"negative_length", TopKConfig{1, 2, 2, 1, false}, []float32{2, 3}, []int32{-1}, []int32{0}},
		topkCase{"negative_offset", TopKConfig{1, 2, 2, 1, false}, []float32{2, 3}, []int32{1}, []int32{-1}},
		topkCase{"all_negative_infinity", TopKConfig{1, 3, 3, 2, true}, []float32{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))}, []int32{3}, []int32{0}},
		topkCase{"nan_multirow", TopKConfig{2, 2, 2, 2, true}, []float32{1, 2, 3, float32(math.NaN())}, []int32{2, 2}, []int32{0, 0}},
		topkCase{"subnormal_neighbors", TopKConfig{1, 4, 4, 4, true}, []float32{math.Float32frombits(1), 0, math.Float32frombits(0x80000001), math.Float32frombits(2)}, []int32{4}, []int32{0}},
	)
	return fixtures
}

func fmtTopKName(n, k int) string { return fmt.Sprintf("length%d_k%d", n, k) }

func TestTopKSortReference(t *testing.T) {
	f := topkCase{"oracle", TopKConfig{1, 5, 5, 3, true}, []float32{3, 9, 1, 9, 5}, []int32{5}, []int32{10}}
	a := topkSortReference(f)
	if !reflect.DeepEqual(a.indices, []int32{11, 13, 14}) || !reflect.DeepEqual(a.values, []float32{9, 9, 5}) {
		t.Fatal(a)
	}
}

func TestTopKShortRows(t *testing.T) {
	for _, f := range topkFixtures() {
		if f.lengths[0] < 0 || f.lengths[0] > int32(f.c.K) {
			continue
		}
		t.Run(f.name, func(t *testing.T) {
			a := topkSortReference(f)
			if a.status[0] != 0 {
				return
			}
			for j := int(a.counts[0]); j < f.c.K; j++ {
				if a.indices[j] != -1 || !math.IsInf(float64(a.values[j]), -1) {
					t.Fatal("padding", a)
				}
			}
		})
	}
}

func TestTopKTiesAndNonFinite(t *testing.T) {
	f := topkFixtures()[14]
	a := topkSortReference(f)
	if !reflect.DeepEqual(a.indices, []int32{3, 7, 4, 5, 1, 2, 0, 6}) {
		t.Fatal(a.indices)
	}
	if math.Float32bits(a.values[4]) != 0x80000000 || math.Float32bits(a.values[5]) != 0 {
		t.Fatal("signed zeros changed")
	}
	for _, f := range topkFixtures() {
		if f.name == "nan_short" || f.name == "nan_long" {
			if a := topkSortReference(f); a.status[0] != int32(TopKNaN) || a.counts[0] != 0 {
				t.Fatal(a)
			}
		}
	}
}

func TestTopKRowStrideAndOffset(t *testing.T) {
	for _, f := range topkFixtures() {
		if f.name == "stride_padding" {
			a := topkSortReference(f)
			if !reflect.DeepEqual(a.indices, []int32{102, 100, 200, -1, 301, 300, -1, -1}) {
				t.Fatal(a.indices)
			}
		}
	}
	for _, x := range []struct {
		n, o int32
		want bool
	}{{2, math.MaxInt32, false}, {1, math.MaxInt32, true}, {0, math.MaxInt32, true}, {1, -1, false}, {-1, 0, false}, {3, 0, false}} {
		if err := topkCheckRow(x.n, x.o, 2); (err == nil) != x.want {
			t.Fatal(x, err)
		}
	}
}

func TestTopKAdmission(t *testing.T) {
	valid := TopKConfig{Rows: 2, Width: 3, RowStride: 5, K: 2, Values: true}
	e, err := valid.Extents()
	if err != nil || e.Input != 8 || e.Results != 4 || e.ScratchBytes != 48 {
		t.Fatal(e, err)
	}
	for _, c := range []TopKConfig{{0, 1, 1, 1, false}, {1, 4097, 4097, 1, false}, {1, 1, 1, 33, false}} {
		if _, err := c.Extents(); !errors.Is(err, ErrTopKUnsupported) {
			t.Fatal(c, err)
		}
	}
	for _, c := range []TopKConfig{{2, 3, 2, 1, false}, {2, 3, int(^uint(0) >> 1), 1, false}} {
		if _, err := c.Extents(); err == nil || errors.Is(err, ErrTopKUnsupported) {
			t.Fatal(c, err)
		}
	}
	dev := new(cuda.Device)
	base := []topkBuffer{{name: "a", ptr: 0x1000, length: 4, need: 4, owner: dev}, {name: "b", ptr: 0x2000, length: 4, need: 4, owner: dev}}
	if err := topkCheckBuffers(base, dev); err != nil {
		t.Fatal(err)
	}
	fourAligned := append([]topkBuffer(nil), base...)
	fourAligned[0].ptr += 4
	fourAligned[1].ptr += 4
	if err := topkCheckBuffers(fourAligned, dev); err != nil {
		t.Fatal("parameter-slot alignment was applied to pointed-to views", err)
	}
	for _, kind := range []string{"alias", "partial_alias", "alignment", "freed", "short", "owner", "overflow"} {
		t.Run(kind, func(t *testing.T) {
			b := append([]topkBuffer(nil), base...)
			switch kind {
			case "alias":
				b[1].ptr = 0x1000
			case "partial_alias":
				b[1].ptr = 0x1008
			case "alignment":
				b[1].ptr++
			case "freed":
				b[1].ptr = 0
			case "short":
				b[1].length--
			case "owner":
				b[1].owner = new(cuda.Device)
			case "overflow":
				b[1].ptr = ^uint64(0) - 7
			}
			if topkCheckBuffers(b, dev) == nil {
				t.Fatal("admitted", kind)
			}
		})
	}
	for _, status := range [][]int32{nil, {1}, {2}, {3}, {4}, {-1}, {0, 1}} {
		if CheckTopKStatus(status) == nil {
			t.Fatal(status)
		}
	}
	if err := CheckTopKStatus([]int32{0, 0}); err != nil {
		t.Fatal(err)
	}
	schema, err := TopKSchema()
	if err != nil || schema.TotalParamBytes != 80 || len(schema.Params) != 12 {
		t.Fatal(schema, err)
	}
	if schema.Params[7].Offset != 56 || schema.Params[11].Offset != 76 {
		t.Fatal(schema)
	}
}

func TestTopKIndicesOnly(t *testing.T) {
	c := TopKConfig{1, 2, 2, 2, false}
	e, err := c.Extents()
	if err != nil || e.ScratchBytes != 16 {
		t.Fatal(e, err)
	}
	if topkCheckBuffers([]topkBuffer{{name: "values", length: 1, need: 0}}, nil) == nil {
		t.Fatal("admitted unused value output")
	}
}

func TestTopKPublicationAndCompletion(t *testing.T) {
	c := TopKConfig{Rows: 2, K: 2}
	for _, mode := range []string{"healthy", "nan_other_row", "unknown_status", "bad_count", "missing_row", "launch_fail", "first_wait_fail", "publish_fail", "last_wait_fail"} {
		t.Run(mode, func(t *testing.T) {
			var steps []string
			waits := 0
			completed, err := topkInvoke(c, func() error {
				steps = append(steps, "launch")
				if mode == "launch_fail" {
					return errors.New("launch")
				}
				return nil
			}, func() error {
				waits++
				steps = append(steps, "wait")
				if mode == "first_wait_fail" || mode == "last_wait_fail" && waits == 2 {
					return errors.New("wait")
				}
				return nil
			}, func() ([]int32, []int32, error) {
				steps = append(steps, "read")
				status, count := []int32{0, 0}, []int32{2, 0}
				switch mode {
				case "nan_other_row":
					status[1] = 1
				case "unknown_status":
					status[1] = 99
				case "bad_count":
					count[1] = 3
				case "missing_row":
					status = status[:1]
				}
				return status, count, nil
			}, func() error {
				steps = append(steps, "publish")
				if mode == "publish_fail" {
					return errors.New("publish")
				}
				return nil
			})
			if (err == nil) != (mode == "healthy") {
				t.Fatal(mode, err)
			}
			if completed != (mode != "first_wait_fail" && mode != "last_wait_fail") {
				t.Fatal(mode, completed)
			}
			if mode == "healthy" && !reflect.DeepEqual(steps, []string{"launch", "wait", "read", "publish", "wait"}) {
				t.Fatal(steps)
			}
			if mode == "nan_other_row" || mode == "unknown_status" || mode == "bad_count" || mode == "missing_row" || mode == "launch_fail" || mode == "first_wait_fail" {
				for _, step := range steps {
					if step == "publish" {
						t.Fatal("poison published", mode)
					}
				}
			}
		})
	}
}

func FuzzTopKExtents(f *testing.F) {
	f.Add(2, 3, 5, 2)
	f.Add(65535, 4096, int(^uint(0)>>1), 32)
	f.Fuzz(func(t *testing.T, rows, width, stride, k int) {
		c := TopKConfig{rows, width, stride, k, true}
		e, err := c.Extents()
		if err != nil {
			return
		}
		if e.Input < width || e.Results != rows*k || uint64(e.Input) > uint64(^uint(0)>>1)/4 {
			t.Fatal(c, e)
		}
	})
}
