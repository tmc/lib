//go:build linux

package main

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAV41PoolPartitionsAndMutants(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in V4.1 CUDA gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	const width, total = 512, 9
	kv, scores := make([]float32, width*total), make([]float32, width*total)
	for row := 0; row < total; row++ {
		for col := 0; col < width; col++ {
			i := row*width + col
			kv[i] = float32(row*7-col%29) / 16
			scores[i] = float32((row*3+col)%7-3) / 4
		}
	}
	partitions := [][]int{{9}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {3, 2, 1, 3}, {2, 3, 4}, {1, 8}, {8, 1}}
	for _, variant := range []struct{ name, old, new string }{
		{"baseline", "", ""},
		{"carry", "prev_kv[col]=clear?0:kv[uint64_t(row)*width+col];", "prev_kv[col]=0;"},
		{"tail", "float ka=row<0?prev_kv[col]:kv[uint64_t(row)*width+col];", "float ka=row<0?kv[col]:kv[uint64_t(row)*width+col];"},
	} {
		t.Run(variant.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			source := cudaV41PoolSource
			if variant.old != "" {
				if strings.Count(source, variant.old) != 1 {
					t.Fatal("mutant site not unique")
				}
				source = strings.Replace(source, variant.old, variant.new, 1)
			}
			pool, err := compileCUDAV41Pool(r, width, source)
			if pool != nil {
				t.Cleanup(func() {
					if err := pool.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			violations := 0
			var canonical []uint32
			for repeat := 0; repeat < 2; repeat++ {
				for _, chunks := range partitions {
					before := violations
					if err := pool.Reset(); err != nil {
						t.Fatal(err)
					}
					var reference dsv41.Pool
					position := 0
					var assembled []uint32
					for _, rows := range chunks {
						start, end := position*width, (position+rows)*width
						n := ((rows + position%2) / 2) * width
						want := make([]float32, n)
						if err := reference.Append(want, kv[start:end], scores[start:end], width, position); err != nil {
							t.Fatal(err)
						}
						k, s := cudaSynthUpload(t, r, kv[start:end]), cudaSynthUpload(t, r, scores[start:end])
						var out cudamem.Slice[float32]
						if n != 0 {
							out = cudaSynthUpload(t, r, make([]float32, n))
						}
						if err := pool.Append(out, k, s, uint32(position)); err != nil {
							t.Fatal(err)
						}
						if pool.position != uint32(position+rows) || !pool.valid {
							t.Fatal("frontier not committed")
						}
						got := make([]float32, n)
						if n != 0 {
							if err := out.CopyToHost(got); err != nil {
								t.Fatal(err)
							}
						}
						for i, v := range got {
							bits := math.Float32bits(v)
							assembled = append(assembled, bits)
							if bits != math.Float32bits(want[i]) {
								violations++
							}
						}
						position += rows
					}
					if variant.name == "baseline" {
						if canonical == nil {
							canonical = append([]uint32(nil), assembled...)
						} else {
							if len(canonical) != len(assembled) {
								t.Fatal("partition shape differs")
							}
							for i := range canonical {
								if canonical[i] != assembled[i] {
									t.Fatalf("partition/repeat mismatch at %d", i)
								}
							}
						}
					}
					if variant.name != "baseline" && len(chunks) == 1 && violations != before {
						t.Fatal("carry mutant altered adjacent no-carry path")
					}
				}
			}
			t.Logf("all-value violations=%d partitions=%d repeats=2", violations, len(partitions))
			if variant.name == "baseline" {
				if violations != 0 {
					t.Fatal("device pooling differs from scalar")
				}
				testCUDAV41PoolAdmission(t, pool, r, width)
			} else if violations == 0 {
				t.Fatal("pool mutant survived")
			}
		})
	}
}

func testCUDAV41PoolAdmission(t *testing.T, pool *cudaV41Pool, r *cudaFoundation, width int) {
	t.Helper()
	if err := pool.Reset(); err != nil {
		t.Fatal(err)
	}
	input := make([]float32, width)
	for i := range input {
		input[i] = float32(i) / 16
	}
	k, s := cudaSynthUpload(t, r, input), cudaSynthUpload(t, r, input)
	var empty cudamem.Slice[float32]
	for _, call := range []func() error{
		func() error { return pool.Append(empty, empty, empty, 0) },
		func() error { return pool.Append(empty, k, s, 1) },
		func() error { return pool.Append(k, k, s, 0) },
		func() error { return pool.Append(empty, k, empty, 0) },
	} {
		if err := call(); err == nil {
			t.Fatal("invalid append admitted")
		}
		if pool.position != 0 || !pool.valid {
			t.Fatal("admission changed frontier")
		}
	}
	if err := pool.Append(empty, k, s, 0); err != nil {
		t.Fatal(err)
	}
	if pool.position != 1 {
		t.Fatal("odd carry not committed")
	}
	carry := make([]float32, width)
	if err := pool.kv.CopyToHost(carry); err != nil {
		t.Fatal(err)
	}
	for i := range carry {
		if math.Float32bits(carry[i]) != math.Float32bits(input[i]) {
			t.Fatalf("carry differs at %d", i)
		}
	}
	// The no-pair path may update carry, but cannot mutate borrowed inputs.
	got := make([]float32, width)
	if err := k.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != input[i] {
			t.Fatal("input changed")
		}
	}
	pool.valid = false
	if err := pool.Append(empty, k, s, 1); err == nil {
		t.Fatal("invalid request admitted")
	}
	if err := pool.Reset(); err != nil {
		t.Fatal(err)
	}
	if pool.position != 0 || !pool.valid {
		t.Fatal("reset did not recover")
	}
	// cudalaunch.Launch rejects Function(0) before driver submission. The first
	// real kernel still enqueues, exercising drain after a later launch error.
	two := append(append([]float32(nil), input...), input...)
	k2, s2 := cudaSynthUpload(t, r, two), cudaSynthUpload(t, r, two)
	out := cudaSynthUpload(t, r, make([]float32, width))
	function := pool.carry
	pool.carry = 0
	launches := cudalaunch.TotalLaunches()
	err := pool.Append(out, k2, s2, 0)
	pool.carry = function
	if err == nil {
		t.Fatal("invalid driver function admitted")
	}
	if pool.valid || pool.position != 0 {
		t.Fatal("failed launch published frontier")
	}
	if cudalaunch.TotalLaunches()-launches != 1 {
		t.Fatal("error control did not enqueue exactly one real pooling kernel")
	}
	if len(pool.pending) != 0 {
		t.Fatal("successful error-path drain retained borrow")
	}
	if err := pool.Append(out, k2, s2, 0); err == nil {
		t.Fatal("retry without reset admitted")
	}
	if err := pool.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := pool.Append(out, k2, s2, 0); err != nil {
		t.Fatal(err)
	}
	var reference dsv41.Pool
	want := make([]float32, width)
	if err := reference.Append(want, two, two, width, 0); err != nil {
		t.Fatal(err)
	}
	replayed := make([]float32, width)
	if err := out.CopyToHost(replayed); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Float32bits(replayed[i]) != math.Float32bits(want[i]) {
			t.Fatalf("reset replay differs at %d", i)
		}
	}
	if err := pool.Close(); err != nil {
		t.Fatal(err)
	}
	if err := pool.Append(empty, k, s, 0); err == nil {
		t.Fatal("closed append admitted")
	}
	if err := pool.Reset(); err == nil {
		t.Fatal("closed reset admitted")
	}
	if err := pool.Close(); err != nil {
		t.Fatal(err)
	}
}
