package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPJRTTextPreflight(t *testing.T) {
	data, err := os.ReadFile("../../x/hftokenizer/testdata/llama.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name      string
		vocab     int
		tokenizer []byte
		want      string
	}{
		{"valid", 270, data, ""},
		{"bounds", 269, data, "maximum token ID"},
		{"malformed", 270, []byte("{"), "pjrt tokenizer"},
		{"missing", 270, nil, "tokenizer.json"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			cfg := pjrtConfigJSON(t)
			cfg["vocab_size"] = tt.vocab
			config, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "config.json"), config, 0600); err != nil {
				t.Fatal(err)
			}
			if tt.tokenizer != nil {
				if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), tt.tokenizer, 0600); err != nil {
					t.Fatal(err)
				}
			}
			tok, err := preflightPJRTText(dir)
			if tt.want == "" {
				if err != nil || tok == nil {
					t.Fatalf("preflight: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %q", err, tt.want)
			}
		})
	}
}
