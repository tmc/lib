package cudamem_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type qualificationCopyCase struct {
	name, source, destination string
	width, collapse           int
	refuse                    bool
}

func qualificationCopyCases() []qualificationCopyCase {
	return []qualificationCopyCase{
		{"irregular1003", "1003:1", "1003:1", 1003, 1, false},
		{"vector4", "4:1", "4:1", 4, 1, false},
		{"scalar35", "(5,7):(7,1)", "(7,5):(5,1)", 1, 1, false},
		{"gapped", "(4,2):(2,16)", "(4,2):(3,20)", 1, 1, false},
		{"collapsed", "4:0", "4:0", 1, 4, false},
		{"incompatible_alias", "4:1", "4:0", 0, 0, true},
	}
}

func TestCopyQualificationPacketLayouts(t *testing.T) {
	for _, tc := range qualificationCopyCases() {
		t.Run(tc.name, func(t *testing.T) {
			source, err := cudalayout.ParseLayout(tc.source)
			if err != nil {
				t.Fatal(err)
			}
			destination, err := cudalayout.ParseLayout(tc.destination)
			if err != nil {
				t.Fatal(err)
			}
			plan, err := cudamem.PlanCopy(source, destination)
			if tc.refuse {
				if !errors.Is(err, cudamem.ErrWriteAfterWrite) {
					t.Fatalf("plan error=%v, want incompatible alias refusal", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if plan.VectorWidth != tc.width || plan.CollapseFactor != tc.collapse {
				t.Fatalf("width/collapse=%d/%d want%d/%d", plan.VectorWidth, plan.CollapseFactor, tc.width, tc.collapse)
			}
		})
	}
}
