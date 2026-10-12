package cudamega_test

import (
	"slices"
	"testing"
)

func TestStepsCopiesInputs(t *testing.T) {
	p, err := inputsProgram([]string{"a"}, [3]uint64{})
	if err != nil {
		t.Fatal(err)
	}
	first, second := p.Steps(), p.Steps()
	first[2].Task.Inputs[0] = "b"
	if got := first[2].Task.Inputs[0]; got != "b" {
		t.Fatalf("returned copy input = %q, want b", got)
	}
	for name, steps := range map[string][]string{
		"earlier copy": second[2].Task.Inputs,
		"fresh copy":   p.Steps()[2].Task.Inputs,
	} {
		if !slices.Equal(steps, []string{"a"}) {
			t.Errorf("%s inputs = %v, want [a]", name, steps)
		}
	}
}

func TestStepsEmptyInputs(t *testing.T) {
	for _, tt := range []struct {
		name   string
		inputs []string
	}{
		{name: "nil"},
		{name: "empty", inputs: []string{}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p, err := inputsProgram(tt.inputs, [3]uint64{})
			if err != nil {
				t.Fatal(err)
			}
			steps := p.Steps()
			// Graph construction normalizes both empty forms to nil.
			if steps[2].Task.Inputs != nil {
				t.Fatalf("empty inputs = %#v, want nil", steps[2].Task.Inputs)
			}
			steps[2].Task.Inputs = append(steps[2].Task.Inputs, "b")
			if got := p.Steps()[2].Task.Inputs; got != nil {
				t.Fatalf("append changed program inputs to %#v", got)
			}
		})
	}
}
