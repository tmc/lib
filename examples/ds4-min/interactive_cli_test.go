package main

import (
	"bytes"
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4chat"
)

type countedChatState struct {
	stateSessionBackendTestState
	closes int
}

func (s *countedChatState) Close() error { s.closes++; return s.stateSessionBackendTestState.Close() }

func TestInteractiveStateTransfer(t *testing.T) {
	s := new(countedChatState)
	model := &stateSessionBackendTestBackend{state: s}
	legacy, closer, err := newStateSessionBackend(model, 100)
	if err != nil {
		t.Fatal(err)
	}
	b, err := transferInteractiveState(legacy)
	if err != nil {
		t.Fatal(err)
	}
	session, err := dsv4.NewSession(b, 100, dsv4.NewStopSet(9))
	if err != nil {
		t.Fatal(err)
	}
	if b.state != nil {
		t.Fatal("state transfer retained the state")
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	if s.closes != 1 || !model.close {
		t.Fatalf("state closed %d times, model closed %v", s.closes, model.close)
	}
}

type untouchedChatReader struct{}

func (untouchedChatReader) Read([]byte) (int, error) { panic("parseConfig consumed interactive stdin") }

func TestInteractiveConfig(t *testing.T) {
	args := []string{"-model", "m", "-backend", "cpu", "-interactive"}
	cfg, err := parseConfig(args, untouchedChatReader{}, new(bytes.Buffer))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.interactive || cfg.thinkMode != ThinkHigh || cfg.sampling.Temperature != 1 {
		t.Fatalf("defaults %+v", cfg)
	}
	for _, extra := range [][]string{{"-prompt", ""}, {"-prompt-file", "missing"}, {"-raw"}, {"-dump-tokens"}, {"-kv-save", "missing"}, {"-eval-suite", "hard-smoke"}} {
		if _, err := parseConfig(append(append([]string(nil), args...), extra...), untouchedChatReader{}, new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted %v", extra)
		}
	}
}

func TestInteractiveCLIScript(t *testing.T) {
	tok := maxPrefixTokenizer()
	tok.eos = 9
	tok.tokens = make([]string, 266)
	tok.stops = stopSet{9: true}
	b := &stateSessionBackendTestBackend{state: new(stateSessionBackendTestState)}
	// All fixture logits select token zero. Mark it as stop so the script
	// exercises real sampler/stop dispatch without generated printable text.
	tok.stops[0] = true
	cfg := config{context: 100, tokens: 3, system: "", thinkMode: ThinkHigh}
	var out bytes.Buffer
	if err := runInteractiveState(context.Background(), cfg, tok, b, strings.NewReader("hello\n/reset\nworld\n/quit\nignored\n"), &out, &out); err != nil {
		t.Fatal(err)
	}
	if got, want := out.String(), "> \n> > \n> "; got != want {
		t.Fatalf("output %q, want %q", got, want)
	}
	if !b.state.(*stateSessionBackendTestState).closed || b.close {
		t.Fatalf("state/model ownership state=%v model=%v", b.state, b.close)
	}
}

func TestInteractiveTokenHistory(t *testing.T) {
	tok := maxPrefixTokenizer()
	tok.eos = 9
	for _, mode := range []ThinkMode{ThinkNone, ThinkHigh, ThinkMax} {
		first := []dsv4chat.Message{{Role: "user", Text: "one"}}
		prefix, err := tok.renderInteractive("system", first, mode, 4096)
		if err != nil {
			t.Fatal(err)
		}
		wantFirst, err := tok.RenderChat("system", "one", mode, 4096)
		if err != nil || !reflect.DeepEqual(prefix, wantFirst) {
			t.Fatalf("first turn %v != %v (%v)", prefix, wantFirst, err)
		}
		turns := append(first, dsv4chat.Message{Role: "assistant", Text: "not reversible", Tokens: []int{700, 701}}, dsv4chat.Message{Role: "user", Text: "two"})
		got, err := tok.renderInteractive("system", turns, mode, 4096)
		if err != nil {
			t.Fatal(err)
		}
		text, _ := tok.Encode("two")
		want := append(append([]int(nil), prefix...), 700, 701, 9, tok.user)
		want = append(want, text...)
		want = append(want, tok.assistant)
		if mode == ThinkNone {
			want = append(want, tok.thinkingEnd)
		} else {
			want = append(want, tok.thinking)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("history %v != %v", got, want)
		}
	}
}
