package cudagrammar_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagrammar"
)

func TestJSONConstraintValidation(t *testing.T) {
	c := cudagrammar.NewJSONConstraint()

	if !c.IsValidToken("{\"name\":") {
		t.Error("expected { to be valid starting token")
	}

	c.Accept("{\"name\": \"llama\"}")
	if !c.IsComplete() {
		t.Error("expected constraint to be complete after closing brace")
	}
}

func TestJSONLogitMasking(t *testing.T) {
	c := cudagrammar.NewJSONConstraint()
	logits := []float32{1.0, 1.0, 1.0}
	vocab := map[string]int32{
		"{":     0,
		"hello": 1,
		"123":   2,
	}

	c.ApplyMask(logits, vocab)

	if logits[0] != 1.0 {
		t.Errorf("expected '{' logit to remain 1.0, got %g", logits[0])
	}
	if logits[1] != cudagrammar.MaskedLogit {
		t.Errorf("expected 'hello' logit to be masked, got %g", logits[1])
	}
}
