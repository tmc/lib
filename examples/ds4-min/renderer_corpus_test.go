package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRendererCorpusRoundTripAndMutation(t *testing.T) {
	tok := maxPrefixTokenizer()
	corpus := rendererCorpus{
		Version: 1,
		Cases: []rendererCase{
			{Name: "chat-none", Kind: "chat", System: "system", Prompt: "hello", Mode: "none", Context: 4096},
			{Name: "chat-high", Kind: "chat", Prompt: "hello", Mode: "high", Context: 4096},
			{Name: "chat-max", Kind: "chat", Prompt: "hello", Mode: "max", Context: maxThinkContext},
			{Name: "chat-rendered-input", Kind: "chat", Prompt: "<｜User｜>b", Mode: "none", Context: 4096},
			{Name: "chat-punctuation-letter", Kind: "chat", Prompt: "x>世界", Mode: "none", Context: 4096},
			{Name: "chat-punctuation-leading", Kind: "chat", Prompt: "#tag", Mode: "none", Context: 4096},
			{Name: "prefix-none", Kind: "prefix", System: "system", Mode: "none", Turns: []rendererTurn{{Role: "USER", Content: "hello"}, {Role: "ASSISTANT", Content: "answer"}}},
			{Name: "prefix-high", Kind: "prefix", Mode: "high", Turns: []rendererTurn{{Role: "USER", Content: "hello"}, {Role: "ASSISTANT", Content: "answer"}}},
		},
	}
	got, err := corpus.render(tok)
	if err != nil {
		t.Fatal(err)
	}
	for i := range corpus.Cases {
		corpus.Cases[i].Ordinal = i
		corpus.Cases[i].OracleCaseID = corpus.Cases[i].Name
		corpus.Cases[i].CandidateCaseID = corpus.Cases[i].Name
		corpus.Cases[i].Want = got[i]
		corpus.Cases[i].InputSHA256, err = rendererInputHash(corpus.Cases[i])
		if err != nil {
			t.Fatal(err)
		}
		corpus.Cases[i].TokenSHA256 = rendererTokenHash(got[i])
	}

	name := filepath.Join(t.TempDir(), "renderer-corpus.json")
	if err := writeRendererCorpus(name, corpus); err != nil {
		t.Fatal(err)
	}
	read, err := readRendererCorpus(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := read.Validate(tok); err != nil {
		t.Fatalf("round trip: %v", err)
	}

	read.Cases[0].Want[0]++
	if err := read.Validate(tok); err == nil {
		t.Fatal("mutated token corpus unexpectedly validated")
	}
	if _, err := os.Stat(name); err != nil {
		t.Fatal(err)
	}
}

func TestRendererCorpusRejectsMissingOrReorderedProvenance(t *testing.T) {
	tok := maxPrefixTokenizer()
	corpus := rendererCorpus{Version: 1, Cases: []rendererCase{
		{Name: "a", Kind: "chat", Mode: "none", Prompt: "a"},
		{Name: "b", Kind: "chat", Mode: "none", Prompt: "b"},
	}}
	got, err := corpus.render(tok)
	if err != nil {
		t.Fatal(err)
	}
	for i := range corpus.Cases {
		corpus.Cases[i].Ordinal = i
		corpus.Cases[i].OracleCaseID = corpus.Cases[i].Name
		corpus.Cases[i].CandidateCaseID = corpus.Cases[i].Name
		corpus.Cases[i].Want = got[i]
		corpus.Cases[i].InputSHA256, err = rendererInputHash(corpus.Cases[i])
		if err != nil {
			t.Fatal(err)
		}
		corpus.Cases[i].TokenSHA256 = rendererTokenHash(got[i])
	}
	missing := corpus
	missing.Cases[0].OracleCaseID = ""
	if err := missing.Validate(tok); err == nil {
		t.Fatal("missing oracle ID unexpectedly validated")
	}
	reordered := corpus
	reordered.Cases[0], reordered.Cases[1] = reordered.Cases[1], reordered.Cases[0]
	if err := reordered.Validate(tok); err == nil {
		t.Fatal("reordered cases unexpectedly validated")
	}
}

func TestRendererCorpusRejectsUnknownModeAndRole(t *testing.T) {
	tok := maxPrefixTokenizer()
	for name, corpus := range map[string]rendererCorpus{
		"mode": {Version: 1, Cases: []rendererCase{{Name: "bad", Kind: "chat", Mode: "bogus"}}},
		"role": {Version: 1, Cases: []rendererCase{{Name: "bad", Kind: "prefix", Mode: "none", Turns: []rendererTurn{{Role: "OTHER"}}}}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := corpus.Validate(tok); err == nil {
				t.Fatal("invalid corpus unexpectedly validated")
			}
		})
	}
}
