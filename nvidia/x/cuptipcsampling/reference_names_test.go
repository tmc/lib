package cuptipcsampling_test

import (
	"strconv"
	"testing"
)

const (
	tutorialVectorAddName      = "_Z9VectorAddPKiS0_Pii"
	tutorialVectorSubtractName = "_Z14VectorSubtractPKiS0_Pii"
	tutorialVectorMultiplyName = "_Z14VectorMultiplyPKiS0_Pii"
)

func TestTutorialPCNamesMatchItaniumDeclarations(t *testing.T) {
	tests := []struct {
		identifier string
		name       string
	}{
		{"VectorAdd", tutorialVectorAddName},
		{"VectorSubtract", tutorialVectorSubtractName},
		{"VectorMultiply", tutorialVectorMultiplyName},
	}
	for _, test := range tests {
		test := test
		t.Run(test.identifier, func(t *testing.T) {
			want := "_Z" + strconv.Itoa(len(test.identifier)) + test.identifier + "PKiS0_Pii"
			if test.name != want {
				t.Fatalf("mangled name = %q, want %q", test.name, want)
			}
		})
	}
	if tutorialVectorAddName != "_Z9VectorAddPKiS0_Pii" {
		t.Fatalf("tutorial VectorAdd name = %q", tutorialVectorAddName)
	}
}
