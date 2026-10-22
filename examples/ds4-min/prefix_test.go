package main

import (
	"strings"
	"testing"
)

func TestParsePrefixFile(t *testing.T) {
	turns, err := parsePrefixFile([]byte("\xef\xbb\xbfUSER: hello\nworld\nASSISTANT:\tanswer\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(turns) != 2 || turns[0].role != prefixUser || turns[0].content != "hello\nworld" || turns[1].role != prefixAssistant || turns[1].content != "answer" {
		t.Fatalf("turns = %#v", turns)
	}
}

func TestParsePrefixFileErrors(t *testing.T) {
	for name, input := range map[string]string{
		"empty":             "",
		"nul":               "USER: a\x00\nASSISTANT: b",
		"preamble":          "hello\nUSER: a\nASSISTANT: b",
		"starts assist":     "ASSISTANT: b",
		"duplicate user":    "USER: a\nUSER: b\nASSISTANT: c",
		"missing assistant": "USER: a",
		"empty user":        "USER:\nASSISTANT: b",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := parsePrefixFile([]byte(input)); err == nil || !strings.Contains(err.Error(), "prefix") {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestParsePrefixFilePreservesInternalLines(t *testing.T) {
	turns, err := parsePrefixFile([]byte("USER: first\n\nsecond\nASSISTANT: reply"))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := turns[0].content, "first\n\nsecond"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}
