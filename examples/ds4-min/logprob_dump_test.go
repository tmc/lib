package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

func testLogprobTokenizer() *tokenizer {
	return &tokenizer{
		tokens: []string{"zero", "one", "two"},
		runeBytes: map[rune]byte{
			'z': 'z', 'e': 'e', 'r': 'r', 'o': 'o',
			'n': 'n', 't': 't', 'w': 'w',
			'\x00': 0,
			'\x01': 1,
			'\x02': 2,
			'\x03': 3,
			'\x04': 4,
			'\x05': 5,
			'\x06': 6,
			'\x07': 7,
			'\x08': 8,
			'\x09': 9,
			'\x0a': 10,
			'\x0b': 11,
			'\x0c': 12,
			'\x0d': 13,
			'\x0e': 14,
			'\x0f': 15,
		},
	}
}

func TestWriteLogprobsUsesUpstreamTokenSchema(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "scores.json")
	tok := testLogprobTokenizer()
	scores := sampler.TopLogprobs([]float32{1, 3, 3}, 3)
	if err := writeLogprobs(path, 4, 32, 256, []logprobStep{{Step: 0, Selected: 1, Scores: scores}}, tok); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var dump struct {
		Source string `json:"source"`
		TopK   int    `json:"top_k"`
		Steps  []struct {
			Selected struct {
				ID    int    `json:"id"`
				Text  string `json:"text"`
				Bytes []int  `json:"bytes"`
			} `json:"selected"`
			Top []struct {
				Token struct {
					ID int `json:"id"`
				} `json:"token"`
			} `json:"top_logprobs"`
		} `json:"steps"`
	}
	if err := json.Unmarshal(raw, &dump); err != nil {
		t.Fatal(err)
	}
	if dump.Source != "ds4-min" || dump.TopK != 128 || len(dump.Steps) != 1 {
		t.Fatalf("header = source %q top_k %d steps %d", dump.Source, dump.TopK, len(dump.Steps))
	}
	selected := dump.Steps[0].Selected
	if selected.ID != 1 || selected.Text != "one" || string(rune(selected.Bytes[0])) != "o" {
		t.Fatalf("selected token = %+v", selected)
	}
	if got := dump.Steps[0].Top[0].Token.ID; got != 1 {
		t.Fatalf("top token = %d, want lower-id tie 1", got)
	}
	if got, want := selected.Bytes, []int{111, 110, 101}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("token bytes = %v, want %v", got, want)
	}
}

func TestWriteLogprobsRejectsUnknownToken(t *testing.T) {
	err := writeLogprobs(filepath.Join(t.TempDir(), "scores.json"), 1, 2, 1,
		[]logprobStep{{Step: 0, Selected: 9}}, testLogprobTokenizer())
	if err == nil || !strings.Contains(err.Error(), "token id 9 out of range") {
		t.Fatalf("error = %v, want unknown token rejection", err)
	}
}
