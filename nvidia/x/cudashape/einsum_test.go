package cudashape

import (
	"errors"
	"reflect"
	"testing"
)

func TestParseDimTokens_DocstringExamples(t *testing.T) {
	tests := []struct {
		input string
		want  []DimToken
	}{
		{"ABC", []DimToken{"A", "B", "C"}},
		{"A1B1C1", []DimToken{"A1", "B1", "C1"}},
		{"A12B34", []DimToken{"A12", "B34"}},
		{"BC(P+R)(Q+S)", []DimToken{"B", "C", "(P+R)", "(Q+S)"}},
	}
	for _, tt := range tests {
		got, err := ParseDimTokens(tt.input, true)
		if err != nil || !reflect.DeepEqual(got, tt.want) {
			t.Errorf("ParseDimTokens(%q) = %v, %v; want %v", tt.input, got, err, tt.want)
		}
	}
}

func TestParseDimTokens_ParenthesesAndValidation(t *testing.T) {
	got, err := ParseDimTokens("(p+(q*r)) A0 / B12", true)
	if err != nil || !reflect.DeepEqual(got, []DimToken{"(P+(Q*R))", "A0", "B12"}) {
		t.Fatalf("got %v, %v", got, err)
	}
	if _, err := ParseDimTokens("AA", true); !errors.Is(err, ErrRepeatedRank) {
		t.Fatalf("repeated rank error = %v", err)
	}
	got, err = ParseDimTokens("AA", false)
	if err != nil || !reflect.DeepEqual(got, []DimToken{"A", "A"}) {
		t.Fatalf("validate=false got %v, %v", got, err)
	}
	got, err = ParseDimTokens("AB0", false)
	if err != nil || !reflect.DeepEqual(got, []DimToken{"A", "B0"}) {
		t.Fatalf("multi-letter prefix was not split as one-letter ranks: %v, %v", got, err)
	}
	for _, input := range []string{"(A", "A)", "(A)(B"} {
		if _, err := ParseDimTokens(input, false); !errors.Is(err, ErrUnbalancedGroup) {
			t.Errorf("ParseDimTokens(%q) error = %v", input, err)
		}
	}
}

func TestParseEinsumEquation_SharedRanksAcrossOperands(t *testing.T) {
	inputs, output, err := ParseEinsumEquation("AB,BC->AC")
	if err != nil {
		t.Fatal(err)
	}
	wantInputs := [][]DimToken{{"A", "B"}, {"B", "C"}}
	if !reflect.DeepEqual(inputs, wantInputs) || !reflect.DeepEqual(output, []DimToken{"A", "C"}) {
		t.Fatalf("got %v -> %v", inputs, output)
	}
	for _, eq := range []string{"AA,B->B", "A,B->CC", "A->B->C", "A,,B->C"} {
		if _, _, err := ParseEinsumEquation(eq); err == nil {
			t.Errorf("ParseEinsumEquation(%q) succeeded", eq)
		}
	}
	inputs, output, err = ParseEinsumEquation("AB->")
	if err != nil || !reflect.DeepEqual(inputs, [][]DimToken{{"A", "B"}}) || len(output) != 0 {
		t.Fatalf("scalar reduction equation got %v -> %v, %v", inputs, output, err)
	}
}
