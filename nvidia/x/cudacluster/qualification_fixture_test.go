package cudacluster

import (
	"fmt"
	"testing"
)

type qualificationReduction struct {
	Kind   string
	Size   int
	Actual uint32
}

// The packet records completed reductions, not expected values relabeled as
// observations. The cooperative oracle and geometry checks run separately.
func checkQualificationReductions(rows []qualificationReduction) error {
	if len(rows) != 8 {
		return fmt.Errorf("expected eight completed reduction observations")
	}
	wrongPublication := false
	for i, row := range rows {
		kind, size := "publication-control", 4
		switch i {
		case 0:
			kind, size = "healthy", 2
		case 1:
			kind = "healthy"
		case 2:
			kind = "local-control"
		}
		if row.Kind != kind || row.Size != size {
			return fmt.Errorf("observation %d has wrong identity", i)
		}
		// Each rank publishes rank*100+7. Keep this arithmetic independent
		// of the reduction generator and its ExpectedReduction helper.
		want := uint32(100*size*(size-1)/2 + 7*size)
		switch kind {
		case "healthy":
			if row.Actual != want {
				return fmt.Errorf("healthy size %d: got %d, want %d", size, row.Actual, want)
			}
		case "local-control":
			if row.Actual != uint32(7*size) || row.Actual == want {
				return fmt.Errorf("local window control did not distinguish remote reads")
			}
		case "publication-control":
			wrongPublication = wrongPublication || row.Actual != want
		}
	}
	if !wrongPublication {
		return fmt.Errorf("all five missing-publication controls produced healthy totals")
	}
	return nil
}

func TestClusterQualification_CompletedObservationControls(t *testing.T) {
	good := []qualificationReduction{{"healthy", 2, 114}, {"healthy", 4, 628}, {"local-control", 4, 28}}
	for range 5 {
		good = append(good, qualificationReduction{"publication-control", 4, 7})
	}
	for _, tc := range []struct {
		name  string
		alter func([]qualificationReduction) []qualificationReduction
		valid bool
	}{
		{"completed", func(r []qualificationReduction) []qualificationReduction { return r }, true},
		{"missing-run", func(r []qualificationReduction) []qualificationReduction { return r[:7] }, false},
		{"wrong-healthy", func(r []qualificationReduction) []qualificationReduction { r[1].Actual = 28; return r }, false},
		{"remote-used-as-local", func(r []qualificationReduction) []qualificationReduction { r[2].Actual = 628; return r }, false},
		{"ineffective-publication-control", func(r []qualificationReduction) []qualificationReduction {
			for i := 3; i < 8; i++ {
				r[i].Actual = 628
			}
			return r
		}, false},
		{"wrong-case-identity", func(r []qualificationReduction) []qualificationReduction { r[0].Size = 4; return r }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows := tc.alter(append([]qualificationReduction(nil), good...))
			if err := checkQualificationReductions(rows); (err == nil) != tc.valid {
				t.Fatalf("valid=%t want %t: %v", err == nil, tc.valid, err)
			}
		})
	}
}
