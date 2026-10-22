package main

import (
	"bufio"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
	"github.com/tmc/lib/gguf"
)

var evalOraclePrompts = flag.String("eval-oracle-prompts", "", "pinned oracle question TSV for CPU hard-smoke preflight")
var evalSmokeCorpus = flag.String("eval-smoke-corpus", "", "external generated corpus for CPU preflight")

func TestEvalSmokePromptPreflight(t *testing.T) {
	if *evalOraclePrompts == "" {
		t.Skip("oracle question TSV not supplied")
	}
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Fatal("DS4_MODEL is required")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Read(f, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	tok, err := newTokenizer(g)
	if err != nil {
		t.Fatal(err)
	}
	oracle, err := os.Open(*evalOraclePrompts)
	if err != nil {
		t.Fatal(err)
	}
	defer oracle.Close()
	s := bufio.NewScanner(oracle)
	s.Buffer(make([]byte, 4096), 1<<20)
	cs, err := dsv4eval.LoadSmokeCorpus(*evalSmokeCorpus)
	if err != nil {
		t.Fatal(err)
	}
	i := 0
	for s.Scan() {
		fields := strings.Split(s.Text(), "\t")
		if len(fields) != 3 || i >= len(cs) {
			t.Fatal("malformed or extra oracle row")
		}
		c := cs[i]
		if fields[0] != c.Source+"/"+c.ID || fields[1] != strconv.Itoa(c.MaxTokens) {
			t.Fatalf("case/budget mismatch at %d", i)
		}
		question, err := hex.DecodeString(fields[2])
		if err != nil {
			t.Fatal(err)
		}
		if string(question) != dsv4eval.QuestionPrompt(c) {
			t.Fatalf("rendered question bytes differ for %s", c.ID)
		}
		ids, err := tok.RenderChat(dsv4eval.EvalSystemPrompt, string(question), ThinkHigh, 32768)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids)+c.MaxTokens > 32768 {
			t.Fatalf("context overflow %s: %d+%d", c.ID, len(ids), c.MaxTokens)
		}
		t.Log(fmt.Sprintf("%s prompt_tokens=%d budget=%d total=%d ctx=32768 question_bytes_equal=true", fields[0], len(ids), c.MaxTokens, len(ids)+c.MaxTokens))
		i++
	}
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
	if i != 12 {
		t.Fatalf("got %d cases, want12", i)
	}
}
