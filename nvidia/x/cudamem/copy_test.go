package cudamem

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

func mustCopyLayout(t *testing.T, text string) cudalayout.Layout {
	t.Helper()
	l, err := cudalayout.ParseLayout(text)
	if err != nil {
		t.Fatalf("ParseLayout(%q): %v", text, err)
	}
	return l
}

func TestCopyPlan_IncompatibleShapesStillCorrect(t *testing.T) {
	src := mustCopyLayout(t, "(5,7):(7,1)")
	dst := mustCopyLayout(t, "(7,5):(5,1)")
	plan, err := PlanCopy(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	if plan.VectorWidth != 1 {
		t.Fatalf("VectorWidth = %d, want 1", plan.VectorWidth)
	}
	if plan.CommonDomain.String() != "(1):(0)" || plan.SourceResidual.String() != src.String() || plan.DestinationResidual.String() != dst.String() {
		t.Fatalf("common=%v residuals=%v/%v", plan.CommonDomain, plan.SourceResidual, plan.DestinationResidual)
	}
	checkPlanAgainstReference(t, src, dst, plan)
}

func TestCopyPlan_NullspaceCollapses(t *testing.T) {
	src := mustCopyLayout(t, "4:0")
	dst := mustCopyLayout(t, "4:0")
	plan, err := PlanCopy(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	if plan.CollapseFactor != 4 {
		t.Fatalf("CollapseFactor = %d, want 4", plan.CollapseFactor)
	}
	if plan.Nullspace.String() != "4:1" {
		t.Fatalf("nullspace = %v, want 4:1", plan.Nullspace)
	}
	if got := totalRunElements(plan.runs); got != 1 {
		t.Fatalf("planned transfer elements = %d, want 1", got)
	}
}

func TestCopyPlan_WriteAfterWriteRejected(t *testing.T) {
	src := mustCopyLayout(t, "4:1")
	dst := mustCopyLayout(t, "4:0")
	before := []byte{0xa5}
	applied := false
	_, err := planCopyAndApply(src, dst, func(plan CopyPlan) error {
		applied = true
		return dispatchCopy(plan, len(before), 4, func(run CopyRun) error {
			before[run.DestinationOffset] = 0
			return nil
		})
	})
	if !errors.Is(err, ErrWriteAfterWrite) {
		t.Fatalf("PlanCopy error = %v, want ErrWriteAfterWrite", err)
	}
	if applied || !reflect.DeepEqual(before, []byte{0xa5}) {
		t.Fatalf("destination changed before planning succeeded: %v", before)
	}
}

func TestCopyPlan_VectorWidthFromContiguousRun(t *testing.T) {
	plan, err := PlanCopy(mustCopyLayout(t, "4:1"), mustCopyLayout(t, "4:1"))
	if err != nil {
		t.Fatal(err)
	}
	if plan.VectorWidth != 4 {
		t.Fatalf("VectorWidth = %d, want 4", plan.VectorWidth)
	}
	if plan.VectorReason != "unit-stride coalesced source over destination inverse" {
		t.Fatalf("vector reason = %q", plan.VectorReason)
	}
}

func TestCopyPlan_ExplainsWidthOne(t *testing.T) {
	plan, err := PlanCopy(mustCopyLayout(t, "(5,7):(7,1)"), mustCopyLayout(t, "(7,5):(5,1)"))
	if err != nil {
		t.Fatal(err)
	}
	if plan.VectorWidth == 1 && plan.VectorReason == "" {
		t.Fatal("width-one plan has no reason")
	}
}

func TestCopy_MatchesReference(t *testing.T) {
	var layouts []cudalayout.Layout
	for _, shape := range [][]int{{24}, {6, 4}, {4, 6}, {2, 3, 4}, {1, 24}, {2, 1, 12}} {
		for pattern := 0; pattern < 8; pattern++ {
			var shapes []cudalayout.Shape
			var strides []cudalayout.Stride
			prefix := 1
			for i, size := range shape {
				d := prefix
				switch pattern {
				case 1:
					d *= 2
				case 2:
					d = 0
				case 3:
					d = 1
				case 4:
					if i == 0 {
						d = 0
					}
				case 5:
					d = -d
				case 6:
					d = 24 / (prefix * size)
				case 7:
					d = 3
				}
				shapes = append(shapes, cudalayout.IntShape(size))
				strides = append(strides, cudalayout.IntStride(d))
				prefix *= size
			}
			layout, err := cudalayout.NewLayout(cudalayout.TupleShape(shapes...), cudalayout.TupleStride(strides...))
			if err != nil {
				t.Fatal(err)
			}
			layouts = append(layouts, layout)
		}
	}
	accepted, rejected := 0, 0
	for _, src := range layouts {
		for _, dst := range layouts {
			_, sourceErr := src.Cosize()
			_, destinationErr := dst.Cosize()
			waw := false
			if sourceErr == nil && destinationErr == nil {
				seen := make(map[int]int)
				for i := 0; i < src.Size(); i++ {
					so, _ := src.Index(i)
					do, _ := dst.Index(i)
					if previous, ok := seen[do]; ok && previous != so {
						waw = true
					}
					seen[do] = so
				}
			}
			plan, err := PlanCopy(src, dst)
			if sourceErr == nil && destinationErr == nil && errors.Is(err, ErrWriteAfterWrite) != waw {
				t.Fatalf("%v/%v: reference WAW=%t, error=%v", src, dst, waw, err)
			}
			if err != nil {
				if errors.Is(err, ErrWriteAfterWrite) || errors.Is(err, cudalayout.ErrUnsupported) {
					rejected++
					continue
				}
				t.Fatalf("PlanCopy(%v, %v): %v", src, dst, err)
			}
			accepted++
			checkPlanAgainstReference(t, src, dst, plan)
		}
	}
	if accepted == 0 || rejected == 0 || accepted+rejected < 200 {
		t.Fatalf("insufficient property coverage: accepted=%d rejected=%d", accepted, rejected)
	}
	t.Logf("generated pairs: accepted=%d rejected=%d", accepted, rejected)
}

func checkPlanAgainstReference(t *testing.T, src, dst cudalayout.Layout, plan CopyPlan) {
	t.Helper()
	sourceSize, err := src.Cosize()
	if err != nil {
		t.Fatal(err)
	}
	destinationSize, err := dst.Cosize()
	if err != nil {
		t.Fatal(err)
	}
	source := make([]byte, sourceSize)
	for i := range source {
		source[i] = byte(17*i + 3)
	}
	want := make([]byte, destinationSize)
	got := make([]byte, destinationSize)
	for i := 0; i < src.Size(); i++ {
		so, err := src.Index(i)
		if err != nil {
			t.Fatal(err)
		}
		do, err := dst.Index(i)
		if err != nil {
			t.Fatal(err)
		}
		want[do] = source[so]
	}
	err = dispatchCopy(plan, len(got), len(source), func(run CopyRun) error {
		copy(got[run.DestinationOffset:run.DestinationOffset+run.Length], source[run.SourceOffset:run.SourceOffset+run.Length])
		return nil
	})
	if err != nil {
		t.Fatalf("dispatch %v into %v: %v", src, dst, err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("planned copy = %v, want reference %v; runs=%+v", got, want, plan.runs)
	}
}

func TestCopyPlan_NullspaceErrors(t *testing.T) {
	dst := mustCopyLayout(t, "4:0")
	_, err := PlanCopy(mustCopyLayout(t, "4:1"), dst)
	if !errors.Is(err, ErrWriteAfterWrite) || errors.Is(err, ErrNullspaceInvariant) {
		t.Fatalf("WAW error = %v", err)
	}
	_, err = planCopy(mustCopyLayout(t, "4:1"), mustCopyLayout(t, "4:1"), copyAlgebra{
		nullspace:    func(cudalayout.Layout) (cudalayout.Layout, error) { return mustCopyLayout(t, "2:1"), nil },
		rightInverse: cudalayout.RightInverse,
	})
	if !errors.Is(err, ErrNullspaceInvariant) || errors.Is(err, ErrWriteAfterWrite) {
		t.Fatalf("malformed nullspace error = %v", err)
	}
}

func TestCopyPlan_OverlappingNonzeroStrides(t *testing.T) {
	l := mustCopyLayout(t, "(2,2):(1,1)")
	plan, err := PlanCopy(l, l)
	if err != nil {
		t.Fatal(err)
	}
	if plan.CollapseFactor != 2 || totalRunElements(plan.runs) != 3 {
		t.Fatalf("plan = %+v", plan)
	}
	checkPlanAgainstReference(t, l, l, plan)
	var lengths []int
	if err := dispatchCopy(plan, 3, 3, func(run CopyRun) error { lengths = append(lengths, run.Length); return nil }); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(lengths, []int{2, 1}) {
		t.Fatalf("vector then residual lengths = %v", lengths)
	}
	_, err = PlanCopy(mustCopyLayout(t, "(2,2):(1,2)"), l)
	if !errors.Is(err, ErrWriteAfterWrite) || errors.Is(err, ErrNullspaceInvariant) {
		t.Fatalf("overlap WAW = %v", err)
	}
}

func TestCopyPlan_RefusesUnsupportedStrides(t *testing.T) {
	f2, err := cudalayout.NewLayout(cudalayout.IntShape(4), cudalayout.F2Stride(0, 3))
	if err != nil {
		t.Fatal(err)
	}
	for _, unsupported := range []cudalayout.Layout{f2, mustCopyLayout(t, "4:-1")} {
		for _, pair := range [][2]cudalayout.Layout{{unsupported, mustCopyLayout(t, "4:1")}, {mustCopyLayout(t, "4:1"), unsupported}} {
			if _, err := PlanCopy(pair[0], pair[1]); !errors.Is(err, cudalayout.ErrUnsupported) {
				t.Fatalf("%v/%v: %v", pair[0], pair[1], err)
			}
		}
	}
}

func TestCopyPlan_VectorizationFallback(t *testing.T) {
	l := mustCopyLayout(t, "4:1")
	plan, err := planCopy(l, l, copyAlgebra{cudalayout.Nullspace, func(cudalayout.Layout) (cudalayout.Layout, error) {
		return cudalayout.Layout{}, fmt.Errorf("forced inverse failure")
	}})
	if err != nil {
		t.Fatal(err)
	}
	if plan.VectorWidth != 1 || plan.VectorReason != "vectorization failed: destination inverse: forced inverse failure" {
		t.Fatalf("fallback = %+v", plan)
	}
	checkPlanAgainstReference(t, l, l, plan)
}

func TestCopyPlan_Residuals(t *testing.T) {
	for _, tc := range []struct {
		src, dst, common, sr, dr string
		width                    int
	}{
		{"(6,4):(1,6)", "(4,6):(1,4)", "(2,2):(1,12)", "6:2", "6:2", 2},
		{"(5,7):(1,2)", "(7,5):(5,1)", "(1):(0)", "(5,7):(1,2)", "(7,5):(5,1)", 1},
	} {
		plan, err := PlanCopy(mustCopyLayout(t, tc.src), mustCopyLayout(t, tc.dst))
		if err != nil {
			t.Fatal(err)
		}
		if plan.CommonDomain.String() != tc.common || plan.SourceResidual.String() != tc.sr || plan.DestinationResidual.String() != tc.dr || plan.VectorWidth != tc.width {
			t.Fatalf("%s/%s: common=%v residuals=%v/%v width=%d", tc.src, tc.dst, plan.CommonDomain, plan.SourceResidual, plan.DestinationResidual, plan.VectorWidth)
		}
		checkPlanAgainstReference(t, mustCopyLayout(t, tc.src), mustCopyLayout(t, tc.dst), plan)
	}
}

func TestCopy_DispatchChecksAndCompleteness(t *testing.T) {
	if _, err := PlanCopy(mustCopyLayout(t, "4:1"), mustCopyLayout(t, "3:1")); !errors.Is(err, ErrCopySizeMismatch) {
		t.Fatalf("size mismatch = %v", err)
	}
	plan := CopyPlan{VectorWidth: 2, ElementCount: 5, transferCount: 5, runs: []CopyRun{{0, 0, 2}, {2, 2, 2}, {4, 4, 1}}}
	var calls []CopyRun
	transfer := func(run CopyRun) error { calls = append(calls, run); return nil }
	if err := dispatchCopy(plan, 4, 5, transfer); !errors.Is(err, ErrCopySizeMismatch) || len(calls) != 0 {
		t.Fatalf("bounds check = %v, calls=%v", err, calls)
	}
	if err := dispatchCopy(plan, 5, 5, transfer); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(calls, plan.runs) {
		t.Fatalf("calls=%v, want every run %v", calls, plan.runs)
	}
}

func TestCopy_DispatchRejectsIncompletePlan(t *testing.T) {
	for _, count := range []int{3, 5} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			plan := CopyPlan{VectorWidth: 2, ElementCount: 5, transferCount: count,
				runs: []CopyRun{{0, 0, 2}, {2, 2, 2}}}
			calls := 0
			err := dispatchCopy(plan, 5, 5, func(CopyRun) error { calls++; return nil })
			if !errors.Is(err, ErrCopySizeMismatch) || calls != 0 {
				t.Fatalf("dispatch error = %v, transfers = %d, want size error before transfer", err, calls)
			}
		})
	}
}

func totalRunElements(runs []CopyRun) int {
	n := 0
	for _, run := range runs {
		n += run.Length
	}
	return n
}
