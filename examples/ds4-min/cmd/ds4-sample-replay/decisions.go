package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

type decision struct {
	Step    int    `json:"step"`
	Token   int    `json:"token"`
	Before  uint64 `json:"rng_before"`
	After   uint64 `json:"rng_after"`
	Stop    bool   `json:"stop"`
	History string `json:"history_sha256"`
	Logits  string `json:"logits_sha256"`
}

func readDecision(data []byte) (decision, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return decision{}, fmt.Errorf("decision must be an object")
	}
	fields := make(map[string]json.RawMessage)
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return decision{}, err
		}
		key, ok := tok.(string)
		if !ok {
			return decision{}, fmt.Errorf("invalid key")
		}
		if _, exists := fields[key]; exists {
			return decision{}, fmt.Errorf("duplicate decision field %q", key)
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return decision{}, err
		}
		fields[key] = value
	}
	if _, err := dec.Token(); err != nil {
		return decision{}, err
	}
	var extra any
	if dec.Decode(&extra) != io.EOF {
		return decision{}, fmt.Errorf("trailing decision JSON")
	}
	for _, key := range []string{"step", "token", "rng_before", "rng_after", "stop", "history_sha256", "logits_sha256"} {
		v, ok := fields[key]
		if !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return decision{}, fmt.Errorf("missing decision field %q", key)
		}
	}
	encoded, err := json.Marshal(fields)
	if err != nil {
		return decision{}, err
	}
	var d decision
	dec = json.NewDecoder(bytes.NewReader(encoded))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&d); err != nil {
		return decision{}, err
	}
	return d, nil
}

// verifyDecisions replays actual observed decisions, not an argmax sidecar.
func verifyDecisions(dir, prompt string, count int, seed uint64) error {
	if count <= 0 || seed == 0 {
		return fmt.Errorf("decision count and seed must be positive")
	}
	history, err := os.ReadFile(prompt)
	if err != nil {
		return err
	}
	if len(history) == 0 || len(history)%4 != 0 {
		return fmt.Errorf("invalid prompt bytes")
	}
	rng := sampler.RNG{State: seed}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	rows, vectors := 0, 0
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".json" {
			rows++
		}
		if filepath.Ext(e.Name()) == ".f32" {
			vectors++
		}
	}
	if rows != count || vectors != count {
		return fmt.Errorf("decision rows %d, want %d", rows, count)
	}
	for step := 0; step < count; step++ {
		base := filepath.Join(dir, fmt.Sprintf("step-%04d", step))
		meta, err := os.ReadFile(base + ".json")
		if err != nil {
			return err
		}
		d, err := readDecision(meta)
		if err != nil {
			return fmt.Errorf("step %d: %w", step, err)
		}
		data, err := os.ReadFile(base + ".f32")
		if err != nil {
			return err
		}
		if len(data) == 0 || len(data)%4 != 0 {
			return fmt.Errorf("step %d invalid logits", step)
		}
		if d.Step != step || d.History != fmt.Sprintf("%x", sha256.Sum256(history)) || d.Logits != fmt.Sprintf("%x", sha256.Sum256(data)) || d.Before != rng.State {
			return fmt.Errorf("step %d provenance/RNG mismatch", step)
		}
		logits := make([]float32, len(data)/4)
		for i := range logits {
			logits[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
			if math.IsNaN(float64(logits[i])) || math.IsInf(float64(logits[i]), 0) {
				return fmt.Errorf("step %d nonfinite", step)
			}
		}
		id := sampler.New(len(logits)).Sample(logits, sampler.Default, &rng)
		if id != d.Token || rng.State != d.After {
			return fmt.Errorf("step %d sampled token/RNG differs: token %d/%d RNG %d/%d", step, id, d.Token, rng.State, d.After)
		}
		if d.Stop {
			if step != count-1 {
				return fmt.Errorf("decision after stop")
			}
			return nil
		}
		var token [4]byte
		binary.LittleEndian.PutUint32(token[:], uint32(id))
		history = append(history, token[:]...)
	}
	return nil
}
