package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// sampleTranscript observes decisions, never logits or sampler inputs.
// A nil transcript is disabled. Close does not manufacture a terminal row:
// an incomplete run must retain its process failure rather than imply success.
type sampleTranscript struct {
	file    *os.File
	next    int
	stopped bool
}

func newSampleTranscript(path string) (*sampleTranscript, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, fmt.Errorf("sampled transcript: create: %w", err)
	}
	return &sampleTranscript{file: f}, nil
}

func (t *sampleTranscript) Decision(step, token int, before, after uint64, stop bool) error {
	if t == nil {
		return nil
	}
	if step != t.next || token < 0 || t.stopped {
		return fmt.Errorf("sampled transcript: invalid decision sequence")
	}
	record := struct {
		Step   int    `json:"step"`
		Token  int    `json:"token"`
		Before uint64 `json:"rng_before"`
		After  uint64 `json:"rng_after"`
		Stop   bool   `json:"stop"`
	}{step, token, before, after, stop}
	if err := json.NewEncoder(t.file).Encode(record); err != nil {
		return fmt.Errorf("sampled transcript: write: %w", err)
	}
	t.next++
	t.stopped = stop
	return nil
}

func (t *sampleTranscript) Close() error {
	if t == nil {
		return nil
	}
	if err := t.file.Close(); err != nil {
		return fmt.Errorf("sampled transcript: close: %w", err)
	}
	return nil
}
