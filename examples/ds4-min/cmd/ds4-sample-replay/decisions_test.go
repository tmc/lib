package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

func TestActualDecisionReplayMutations(t *testing.T) {
	for _, mutation := range []string{"none", "token", "rng", "history", "missing", "trailing", "missing-step", "missing-token", "missing-stop", "duplicate", "extra-vector"} {
		t.Run(mutation, func(t *testing.T) {
			dir := t.TempDir()
			prompt := filepath.Join(dir, "prompt.i32")
			history := []byte{7, 0, 0, 0}
			if err := os.WriteFile(prompt, history, 0600); err != nil {
				t.Fatal(err)
			}
			logits := []float32{1, 2, 3}
			data := make([]byte, 12)
			for i, v := range logits {
				binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
			}
			rng := sampler.RNG{State: 12345}
			id := sampler.New(3).Sample(logits, sampler.Default, &rng)
			d := decision{0, id, 12345, rng.State, false, fmt.Sprintf("%x", sha256.Sum256(history)), fmt.Sprintf("%x", sha256.Sum256(data))}
			switch mutation {
			case "token":
				d.Token = (id + 1) % 3
			case "rng":
				d.After++
			case "history":
				d.History = "wrong"
			}
			meta, err := json.Marshal(d)
			if err != nil {
				t.Fatal(err)
			}
			if mutation == "trailing" {
				meta = append(meta, []byte(" {}")...)
			}
			if mutation == "duplicate" {
				meta = append([]byte(`{"step":0,`), meta[1:]...)
			}
			if strings.HasPrefix(mutation, "missing-") {
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(meta, &fields); err != nil {
					t.Fatal(err)
				}
				delete(fields, strings.TrimPrefix(mutation, "missing-"))
				meta, err = json.Marshal(fields)
				if err != nil {
					t.Fatal(err)
				}
			}
			if mutation == "extra-vector" {
				if err := os.WriteFile(filepath.Join(dir, "step-0001.f32"), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if mutation != "missing" {
				if err := os.WriteFile(filepath.Join(dir, "step-0000.json"), meta, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(filepath.Join(dir, "step-0000.f32"), data, 0600); err != nil {
				t.Fatal(err)
			}
			err = verifyDecisions(dir, prompt, 1, 12345)
			if (err == nil) != (mutation == "none") {
				t.Fatalf("error=%v", err)
			}
		})
	}
}
