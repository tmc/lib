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
		{"range", "[129280]", nil}, {"fraction", "[1.5]", nil},
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
