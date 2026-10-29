package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadModelConfigContext(t *testing.T) {
	for _, tt := range []struct {
		name    string
		maxPos  string
		want    int
		wantErr bool
	}{
		{"8k", `"max_position_embeddings": 8192,`, 8192, false},
		{"128k", `"max_position_embeddings": 131072,`, 131072, false},
		{"missing", ``, 2048, false},
		{"negative", `"max_position_embeddings": -1,`, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			cfg := `{` + tt.maxPos + `"vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
				"num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2}`
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := LoadModelConfig(dir)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("LoadModelConfig = %d tokens, want error", got.MaxSeqLen)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.MaxSeqLen != tt.want {
				t.Errorf("MaxSeqLen = %d, want %d", got.MaxSeqLen, tt.want)
			}
		})
	}
}

func TestLimitContext(t *testing.T) {
	for _, tt := range []struct {
		n, want int
		wantErr bool
	}{
		{0, 8192, false},
		{2048, 2048, false},
		{2176, 2176, false},
		{8192, 8192, false},
		{8193, 0, true},
		{-1, 0, true},
	} {
		cfg := ModelConfig{MaxSeqLen: 8192}
		err := limitContext(&cfg, tt.n)
		if (err != nil) != tt.wantErr {
			t.Errorf("limitContext(%d) error = %v, want error %v", tt.n, err, tt.wantErr)
			continue
		}
		if err == nil && cfg.MaxSeqLen != tt.want {
			t.Errorf("limitContext(%d): MaxSeqLen = %d, want %d", tt.n, cfg.MaxSeqLen, tt.want)
		}
	}
}

// TestDecodeWithContext checks the 2048-token prompt used by the long-prompt
// comparison can emit all 128 requested tokens when the context has room.
func TestDecodeWithContext(t *testing.T) {
	const promptTokens, steps = 2048, 128
	for _, tt := range []struct {
		name    string
		context int
		want    int
	}{
		{"prompt only", promptTokens, 0},
		{"one decode slot", promptTokens + 1, 1},
		{"all requested tokens", promptTokens + steps, steps},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ModelConfig{MaxSeqLen: 8192}
			if err := limitContext(&cfg, tt.context); err != nil {
				t.Fatal(err)
			}
			prompt := make([]int32, promptTokens)
			st := newDecodeState(prompt, steps, nil)
			forward := func(tok int32, pos int) (int32, error) { return tok + 1, nil }
			calls, shown := 0, 0
			plain := func(step int, tok int32, pos int) (int32, error) {
				if step != calls || pos != promptTokens+calls || pos >= cfg.MaxSeqLen {
					t.Fatalf("decode call %d: step %d, position %d, context %d", calls, step, pos, cfg.MaxSeqLen)
				}
				calls++
				return forward(tok, pos)
			}
			show := func(tok int32) error {
				if tok != int32(7+shown) {
					t.Fatalf("token %d = %d, want %d", shown, tok, 7+shown)
				}
				shown++
				return nil
			}
			_, pos, err := decode(st, forward, plain, show, 7, len(prompt), cfg.MaxSeqLen, steps, 0)
			if err != nil {
				t.Fatal(err)
			}
			if len(st.generated) != tt.want || shown != tt.want || calls != tt.want || pos != promptTokens+tt.want {
				t.Fatalf("generated %d, shown %d, forwards %d, final position %d; want %d tokens", len(st.generated), shown, calls, pos, tt.want)
			}
		})
	}
}
