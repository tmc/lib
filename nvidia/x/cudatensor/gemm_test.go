package cudatensor

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

func TestMapCanonicalGEMM(t *testing.T) {
	got, err := MapCanonicalGEMM("AB,BC->AC", map[cudashape.DimToken]int{"A": 64, "B": 16, "C": 32})
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]int{"B": 1, "M": 64, "N": 32, "K": 16} {
		if got.Problem[name] != want {
			t.Errorf("problem[%q] = %d, want %d", name, got.Problem[name], want)
		}
	}
	if got.RankNames["A"] != "M" || got.RankNames["B"] != "K" || got.RankNames["C"] != "N" || strings.Join(got.OperandRoles, ",") != "A,B" || got.OutputRole != "C" {
		t.Fatalf("rank and tensor mapping = %+v", got)
	}
	batched, err := MapCanonicalGEMM("ABC,ACD->ABD", map[cudashape.DimToken]int{"A": 2, "B": 64, "C": 16, "D": 32})
	if err != nil {
		t.Fatal(err)
	}
	if batched.Problem["B"] != 2 || batched.Problem["M"] != 64 || batched.Problem["N"] != 32 || batched.Problem["K"] != 16 {
		t.Fatalf("batched problem = %v", batched.Problem)
	}
}

func TestCanonicalGEMMDataMovementAndFailClosedLabels(t *testing.T) {
	movement, err := CanonicalGEMMDataMovement("AB,BC->AC", map[cudashape.DimToken]int{"A": 64, "B": 16, "C": 32},
		map[string]int{"M": 16, "N": 16}, map[string]int{"A": 2, "B": 2, "C": 2}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if movement.TotalBytes != 7168 || movement.TensorTransfers["A"] != 2048 || movement.TensorTransfers["B"] != 1024 || movement.TensorTransfers["C"] != 4096 {
		t.Fatalf("movement = %+v", movement)
	}
	for _, test := range []struct {
		equation string
		extents  map[cudashape.DimToken]int
	}{
		{"ABC,CD->ABD", map[cudashape.DimToken]int{"A": 2, "B": 8, "C": 4, "D": 16}},
		{"AB,BC->AC", map[cudashape.DimToken]int{"A": 2, "B": 8, "C": 16, "D": 32}},
		{"AB,BC->AC", map[cudashape.DimToken]int{"A": 2, "B": 8}},
	} {
		if _, err := MapCanonicalGEMM(test.equation, test.extents); err == nil || !strings.Contains(err.Error(), test.equation) {
			t.Errorf("MapCanonicalGEMM(%q) error = %v", test.equation, err)
		}
	}
	for _, equation := range []string{"AB,CB->AC", "BA,BC->AC", "AB,BC->CA"} {
		if _, err := MapCanonicalGEMM(equation, map[cudashape.DimToken]int{"A": 2, "B": 8, "C": 16}); err == nil || !strings.Contains(err.Error(), equation) {
			t.Errorf("MapCanonicalGEMM(%q) error = %v, want an equation-specific refusal", equation, err)
		}
	}
}
