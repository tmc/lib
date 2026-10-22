package main

import (
	"bytes"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestParsePromptTokens(t *testing.T) {
	for _, tt := range []struct {
		name, input string
		want        []int
	}{
		{"ids", "[0,129279,42]", []int{0, 129279, 42}},
		{"null", "[null]", nil}, {"negative", "[-1]", nil},
		{"model independent", "[129280]", []int{129280}}, {"fraction", "[1.5]", nil},
		{"empty", "[]", nil}, {"null array", "null", nil},
		{"object", "{}", nil}, {"trailing", "[1] []", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePromptTokens([]byte(tt.input))
			if (err != nil) != (tt.want == nil) || !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v, %v; want %v", got, err, tt.want)
			}
		})
	}
}

func TestPromptTokensConfig(t *testing.T) {
	path := "../../scripts/testdata/oracle-workload-tokens2048.json"
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want, err := parsePrefillTokens(data)
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"-model", "m", "-backend", "cpu", "-prompt-tokens", path, "-tokens", "2"}
	cfg, err := parseConfig(args, errorReader{os.ErrInvalid}, new(bytes.Buffer))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cfg.promptTokens, want) || cfg.prompt != nil || cfg.tokens != 2 {
		t.Fatal("token fixture was changed or stdin was consumed")
	}
	for _, extra := range [][]string{{"-prompt", "hi"}, {"-prompt", ""}, {"-prompt-file", "missing"}, {"-ctx", "1"}} {
		if _, err := parseConfig(append(append([]string{}, args...), extra...), strings.NewReader(""), new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted %v", extra)
		}
	}
}

func TestPreparePromptVocabulary(t *testing.T) {
	for _, tt := range []struct {
		name    string
		vocab   int
		ids     []int
		wantErr bool
	}{
		{"last ID", 3, []int{0, 2}, false},
		{"past end", 3, []int{3}, true},
		{"negative", 3, []int{-1}, true},
		{"empty vocabulary", 0, []int{0}, true},
		{"larger vocabulary", 129281, []int{129280}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := config{promptTokens: tt.ids, context: 8, tokens: 2}
			got, err := preparePrompt(cfg, &tokenizer{tokens: make([]string, tt.vocab)})
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, want error %v", err, tt.wantErr)
			}
			if err == nil && !reflect.DeepEqual(got, tt.ids) {
				t.Fatalf("changed input: got %v, want %v", got, tt.ids)
			}
		})
	}
}

func TestPromptCapacity(t *testing.T) {
	for _, tt := range []struct {
		name                       string
		prompt, context, generated int
		wantErr                    bool
	}{
		{"spare slot", 2, 5, 2, false},
		{"exact fit", 2, 4, 2, false},
		{"one decode slot missing", 2, 3, 2, true},
		{"prompt fills context", 2, 2, 1, true},
		{"prompt exceeds context", 3, 2, 1, true},
		{"large generated count", 2, 4, int(^uint(0) >> 1), true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := config{context: tt.context, tokens: tt.generated}
			err := cfg.checkPromptCapacity(tt.prompt)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, want error %v", err, tt.wantErr)
			}
		})
	}
}

func TestPromptTokensConfigCapacity(t *testing.T) {
	path := "../../scripts/testdata/oracle-workload-tokens2048.json"
	for _, tt := range []struct {
		context string
		wantErr bool
	}{
		{"2047", true}, {"2048", true}, {"2049", true}, {"2050", false},
	} {
		t.Run(tt.context, func(t *testing.T) {
			_, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt-tokens", path, "-tokens", "2", "-ctx", tt.context}, errorReader{os.ErrInvalid}, new(bytes.Buffer))
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, want error %v", err, tt.wantErr)
			}
		})
	}
}

func TestPrepareProsePromptCapacity(t *testing.T) {
	tok := &tokenizer{tokens: make([]string, 260), tokenID: make(map[string]int), bos: 256, user: 257, assistant: 258, thinkingEnd: 259}
	for i := range 256 {
		r := gpt2ByteRune(byte(i))
		tok.byteRunes[i] = r
		tok.tokenID[string(r)] = i
	}
	prompt, err := tok.RenderNonThinking("You are a helpful assistant", "x")
	if err != nil {
		t.Fatal(err)
	}
	cfg := config{prompt: []byte("x"), context: len(prompt) + 2, tokens: 2}
	got, err := preparePrompt(cfg, tok)
	if err != nil || !reflect.DeepEqual(got, prompt) {
		t.Fatalf("exact fit: %v, %v", got, err)
	}
	cfg.context--
	if _, err := preparePrompt(cfg, tok); err == nil {
		t.Fatal("accepted rendered prompt without decode capacity")
	}
}

func TestPromptCapacityMatchesSession(t *testing.T) {
	cfg := config{promptTokens: []int{2, 3}, context: 4, tokens: 2}
	prompt, err := preparePrompt(cfg, &tokenizer{tokens: make([]string, 4)})
	if err != nil {
		t.Fatal(err)
	}
	s, err := newSession(new(fakeSessionBackend), cfg.context, stopSet{1: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	for range cfg.tokens {
		if stop, err := s.Accept(2); err != nil || stop {
			t.Fatalf("exact-fit decode: %v, %v", stop, err)
		}
	}
	if _, err := s.Accept(2); err == nil {
		t.Fatal("extra decode succeeded beyond capacity")
	}
}
