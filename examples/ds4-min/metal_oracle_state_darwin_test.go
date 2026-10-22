package main

import (
	"os"
	"testing"
)

// TestMetalOracleTrace replays the registered prompt-A oracle trace on
// the Metal backend under teacher forcing. It needs both the model
// (DS4_GGUF_PROFILE_MODEL) and a directory of the observer's captured
// states (DS4_ORACLE_STATE_DIR); the shipped one is
// evidence/oracle-provenance/capture-prompt-a-01/capture-prompt-a-01.
func TestMetalOracleTrace(t *testing.T) {
	model := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	dir := os.Getenv("DS4_ORACLE_STATE_DIR")
	if model == "" || dir == "" {
		t.Skip("model path or oracle state directory not set")
	}
	r, err := newMetalRuntime(model, 128)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := r.Close(); err != nil {
				t.Errorf("close runtime: %v", err)
			}
		}
	}()
	tok, err := newTokenizer(r.model)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tok.RenderNonThinking("You are a helpful assistant", "a")
	if err != nil {
		t.Fatal(err)
	}
	// The capture's input.json records the prompt the oracle ran, so a
	// renderer change is caught here rather than showing up as a
	// numerical difference several steps later.
	want := []int{0, 3476, 477, 260, 11502, 22896, 128803, 67, 128804, 128822}
	if len(prompt) != len(want) {
		t.Fatalf("rendered %d prompt tokens, oracle ran %d", len(prompt), len(want))
	}
	for i := range want {
		if prompt[i] != want[i] {
			t.Fatalf("prompt token %d = %d, oracle ran %d", i, prompt[i], want[i])
		}
	}
	s, err := newSession(r, 128, tok.stops)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	compareOracleTrace(t, s, dir, []int{19923, 3, 1730, 588, 342, 1694, 440, 4316, 33, 1})
	closed = true
	if err := r.Close(); err != nil {
		t.Fatalf("close runtime: %v", err)
	}
}
