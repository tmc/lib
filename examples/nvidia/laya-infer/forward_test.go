package main

import (
	"encoding/json"
	"os"
	"testing"
)

// TestScratchReuse checks that a pass carved out of scratch a larger batch
// left behind answers exactly as one on a fresh model does, and that growing
// the scratch mid-run does too.
func TestScratchReuse(t *testing.T) {
	dir := os.Getenv("LAYA_MODEL")
	if dir == "" {
		t.Skip("set LAYA_MODEL to a directory holding model.safetensors")
	}
	big := benchRequest(t)
	var req struct {
		State     json.RawMessage            `json:"state"`
		Questions map[string]json.RawMessage `json:"questions"`
	}
	if err := json.Unmarshal(big, &req); err != nil {
		t.Fatal(err)
	}
	small, err := json.Marshal(map[string]any{
		"state":     req.State,
		"questions": map[string]json.RawMessage{"escalate": req.Questions["escalate"]},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, prec := range []precision{fast, exact, half} {
		fresh := func(request []byte) string {
			m, err := newModel(dir, prec)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			reply, err := m.Predict(request)
			if err != nil {
				t.Fatal(err)
			}
			return string(reply)
		}
		want := map[string]string{"small": fresh(small), "big": fresh(big)}

		m, err := newModel(dir, prec)
		if err != nil {
			t.Fatal(err)
		}
		// small, then big grows the scratch, then small reuses the larger one.
		for _, name := range []string{"small", "big", "small", "big"} {
			request := small
			if name == "big" {
				request = big
			}
			reply, err := m.Predict(request)
			if err != nil {
				t.Fatal(err)
			}
			if string(reply) != want[name] {
				t.Errorf("precision %d, %s after reuse:\n got %s\nwant %s", prec, name, reply, want[name])
			}
		}
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	}
}
