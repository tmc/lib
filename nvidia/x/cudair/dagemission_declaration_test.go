package cudair

import "testing"

func TestScalarMultiplicationIsNotPointerDeclaration(t *testing.T) {
	for _, body := range []string{
		"float scale = 2.0f; float normalized = (input[0] * scale) * input[0]; return input;",
		"float scale = 2.0f; float normalized = input[0] * scale; return input;",
		"float scale = 2.0f; float normalized = 0.0f; for (int i = 0; i < 1; i++) { normalized += input[i] * scale; } return input;",
	} {
		n := &pureMathTestEmitter{body: body}
		if _, err := EmitTree(n, "leaf"); err != nil {
			t.Fatal(body, err)
		}
	}
	for _, body := range []string{
		"float *pointer = input; return input;",
		"Value *pointer; return input;",
		"const Value *pointer = input; return input;",
		"*input = 1.0f; return input;",
		"float scalar; scalar = input[0]; input[0] = scalar; return input;",
	} {
		n := &pureMathTestEmitter{body: body}
		if _, err := EmitTree(n, "leaf"); err == nil {
			t.Fatal("pointer or parameter write admitted", body)
		}
	}
}
