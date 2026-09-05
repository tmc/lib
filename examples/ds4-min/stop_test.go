package main

import (
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/gguf"
)

// stopFixture builds tokenizer metadata for an eight-token vocabulary whose
// ids 0, 1, 5, and 6 are control tokens.
func stopFixture(extra map[string]gguf.Value) *gguf.File {
	types := []any{int32(3), int32(3), int32(1), int32(1), int32(1), int32(3), int32(3), int32(1)}
	g := &gguf.File{Metadata: map[string]gguf.Value{
		"tokenizer.ggml.token_type":   {TypeID: 9, Value: types},
		"tokenizer.ggml.eos_token_id": {TypeID: 4, Value: uint32(1)},
	}}
	for key, v := range extra {
		if v.TypeID == 0 && v.Value == nil {
			delete(g.Metadata, key)
			continue
		}
		g.Metadata[key] = v
	}
	return g
}

func TestGenerationStops(t *testing.T) {
	tests := []struct {
		name  string
		extra map[string]gguf.Value
		want  stopSet
	}{
		{
			name: "eos only",
			want: stopSet{1: true},
		},
		{
			name:  "no metadata ids falls back to the vocabulary token",
			extra: map[string]gguf.Value{"tokenizer.ggml.eos_token_id": {}},
			want:  stopSet{1: true},
		},
		{
			name:  "end of turn joins the set",
			extra: map[string]gguf.Value{"tokenizer.ggml.eot_token_id": {TypeID: 4, Value: uint32(5)}},
			want:  stopSet{1: true, 5: true},
		},
		{
			name: "end of turn and message join the set",
			extra: map[string]gguf.Value{
				"tokenizer.ggml.eot_token_id": {TypeID: 4, Value: uint32(5)},
				"tokenizer.ggml.eom_token_id": {TypeID: 4, Value: uint32(6)},
			},
			want: stopSet{1: true, 5: true, 6: true},
		},
		{
			name:  "metadata id differing from the vocabulary token joins it",
			extra: map[string]gguf.Value{"tokenizer.ggml.eos_token_id": {TypeID: 4, Value: uint32(6)}},
			want:  stopSet{1: true, 6: true},
		},
		{
			name:  "missing token types still derive a set",
			extra: map[string]gguf.Value{"tokenizer.ggml.token_type": {}},
			want:  stopSet{1: true},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tok := &tokenizer{tokens: make([]string, 8), eos: 1}
			got, err := tok.generationStops(stopFixture(tt.extra))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("stops = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGenerationStopsRejects(t *testing.T) {
	tests := []struct {
		name  string
		extra map[string]gguf.Value
	}{
		{"ordinary token", map[string]gguf.Value{"tokenizer.ggml.eot_token_id": {TypeID: 4, Value: uint32(2)}}},
		{"out of range", map[string]gguf.Value{"tokenizer.ggml.eot_token_id": {TypeID: 4, Value: uint32(8)}}},
		{"not an integer", map[string]gguf.Value{"tokenizer.ggml.eot_token_id": {TypeID: 8, Value: "5"}}},
		{"short token types", map[string]gguf.Value{"tokenizer.ggml.token_type": {TypeID: 9, Value: []any{int32(3)}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tok := &tokenizer{tokens: make([]string, 8), eos: 1}
			if _, err := tok.generationStops(stopFixture(tt.extra)); !errors.Is(err, errInvalidTokenizer) {
				t.Fatalf("error = %v, want invalid tokenizer", err)
			}
		})
	}
}

func TestSessionStopSetRequired(t *testing.T) {
	if _, err := newSession(new(fakeSessionBackend), 8, nil); err == nil {
		t.Fatal("empty stop set accepted")
	}
}

// TestSessionStopsOnEveryStopToken checks that a non-EOS stop token also halts
// generation before the backend evaluates it.
func TestSessionStopsOnEveryStopToken(t *testing.T) {
	b := new(fakeSessionBackend)
	s, err := newSession(b, 8, stopSet{1: true, 5: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync([]int{10}); err != nil {
		t.Fatal(err)
	}
	b.events = nil
	for _, token := range []int{1, 5} {
		stop, err := s.Accept(token)
		if err != nil || !stop {
			t.Fatalf("Accept(%d) = %t, %v", token, stop, err)
		}
	}
	if len(b.events) != 0 {
		t.Fatalf("events after stop = %v", b.events)
	}
}
