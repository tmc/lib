package main

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

// The references are checked against values worked out by hand, not against
// any GPU result. An oracle validated by the thing it judges validates nothing.
func TestReferences(t *testing.T) {
	a := []float32{1, -2, 0.5, 0}
	b := []float32{0.25, 4, -1, 3}
	for _, tt := range []struct {
		name  string
		fn    func(a, b []float32, alpha float32) []float32
		alpha float32
		want  []float32
	}{
		{"axpby", axpbyReference, 2, []float32{2.25, 0, 0, 3}},
		{"relu_axpby", reluAxpbyReference, 2, []float32{2.25, 0, 0, 3}},
		{"relu_axpby clamps", reluAxpbyReference, -4, []float32{0, 12, 0, 3}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.fn(a, b, tt.alpha)
			c, err := cudahotswap.Compare(got, tt.want, cudahotswap.Tolerance{Abs: 1e-6})
			if err != nil {
				t.Fatal(err)
			}
			if !c.OK() {
				t.Fatalf("reference disagrees with the hand-computed result: %+v", c.Worst)
			}
		})
	}
}

func TestRsqrtScaleReference(t *testing.T) {
	a := []float32{3, -1}
	b := []float32{4, 0}
	const eps = 0.25
	got := rsqrtScaleReference(a, b, eps)
	want := []float32{
		float32(3 / math.Sqrt(16+eps)),
		float32(-1 / math.Sqrt(eps)),
	}
	c, err := cudahotswap.Compare(got, want, cudahotswap.Tolerance{Rel: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	if !c.OK() {
		t.Fatalf("reference disagrees with the hand-computed result: %+v", c.Worst)
	}
}

// A seed must reproduce its data and two seeds must not agree, or the
// per-trial regeneration would be five copies of one draw.
func TestGenInputsIsSeeded(t *testing.T) {
	a1, b1 := genInputs(7, 64)
	a2, b2 := genInputs(7, 64)
	for i := range a1 {
		if a1[i] != a2[i] || b1[i] != b2[i] {
			t.Fatalf("seed 7 did not reproduce at element %d", i)
		}
	}
	a3, _ := genInputs(8, 64)
	same := true
	for i := range a1 {
		if a1[i] != a3[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("seeds 7 and 8 produced identical inputs")
	}
	// A generator stuck on one value would also satisfy the checks above.
	distinct := map[float32]bool{}
	for _, v := range a1 {
		distinct[v] = true
	}
	if len(distinct) < 8 {
		t.Fatalf("seed 7 produced only %d distinct values over 64 elements", len(distinct))
	}
}

// Each task's own baseline kernel must name the entry point the task requires,
// or the baseline arm would fail as missing_entry on a correct candidate.
func TestTasksAreSelfConsistent(t *testing.T) {
	for _, name := range TaskNames() {
		task := tasks[name]
		if task.Name != name {
			t.Errorf("task %q is keyed under %q", task.Name, name)
		}
		if task.Baseline == "" {
			t.Errorf("task %q has no baseline kernel", name)
			continue
		}
		if !contains(task.Baseline, "void "+task.Entry+"(") {
			t.Errorf("task %q baseline does not define entry %q", name, task.Entry)
		}
		if task.Grid == 0 || task.Block == 0 || task.N == 0 {
			t.Errorf("task %q has a zero launch dimension or element count", name)
		}
		if task.Tolerance.Abs == 0 && task.Tolerance.Rel == 0 {
			t.Errorf("task %q admits no tolerance at all", name)
		}
		if task.ReferenceDoc == "" {
			t.Errorf("task %q does not say where its reference comes from", name)
		}
	}
}

func contains(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

// The tolerances must reject the wrong answers the testdata fixtures produce.
// A tolerance wide enough to admit them would make every correctness result
// vacuous, and nothing about the CUDA path would reveal that.
func TestTolerancesRejectTheWrongFixtures(t *testing.T) {
	task := tasks["axpby"]
	a, b := genInputs(1, 4096)
	want := task.Reference(a, b, task.Alpha)

	wrong := make([]float32, len(a)) // alpha*a - b, as testdata/axpby_wrong.cu computes
	for i := range a {
		wrong[i] = task.Alpha*a[i] - b[i]
	}
	c, err := cudahotswap.Compare(wrong, want, task.Tolerance)
	if err != nil {
		t.Fatal(err)
	}
	if c.OK() {
		t.Fatal("the axpby tolerance admits the deliberately wrong fixture")
	}

	// testdata/axpby_partial.cu writes only the first grid's worth. With the
	// NaN prefill the tail is non-finite; the comparison must see it.
	partial := append([]float32(nil), want...)
	nan := float32(math.NaN())
	for i := len(partial) / 2; i < len(partial); i++ {
		partial[i] = nan
	}
	c, err = cudahotswap.Compare(partial, want, task.Tolerance)
	if err != nil {
		t.Fatal(err)
	}
	if c.OK() || c.NonFinite == 0 {
		t.Fatalf("an unwritten tail was admitted: %+v", c)
	}
}
