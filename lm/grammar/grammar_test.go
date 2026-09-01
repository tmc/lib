// SPDX-License-Identifier: Apache-2.0

package grammar_test

import (
	"testing"

	"github.com/tmc/lib/lm/grammar"
)

func TestBitmaskTable(t *testing.T) {
	bt := grammar.NewBitmaskTable(4, 128)

	bt.SetValid(0, 10)
	bt.SetValid(0, 20)

	if !bt.IsValid(0, 10) || !bt.IsValid(0, 20) {
		t.Errorf("expected token 10 and 20 to be valid in state 0")
	}
	if bt.IsValid(0, 30) {
		t.Errorf("token 30 should be invalid in state 0")
	}

	logits := make([]float32, 128)
	logits[10] = 5.0
	logits[20] = 3.0
	logits[30] = 4.0

	bt.ApplyMask(0, logits)

	if logits[10] != 5.0 || logits[20] != 3.0 {
		t.Errorf("valid tokens altered: logit[10]=%f, logit[20]=%f", logits[10], logits[20])
	}
	if logits[30] != -1e9 {
		t.Errorf("invalid token 30 was not masked to -1e9: got %f", logits[30])
	}
}

func TestJSONConstraintValidation(t *testing.T) {
	c := grammar.NewJSONConstraint()
	if !c.IsValidToken("{\"name\":") {
		t.Errorf("expected valid token")
	}
	c.Accept("{\"name\": \"test\"}")
	if c.State() != grammar.StateComplete {
		t.Errorf("expected state complete, got %v", c.State())
	}
}
