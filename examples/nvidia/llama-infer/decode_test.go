package main

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/lm/speculative"
)

// feed is one forward pass recorded by a generate test: the function that
// ran it, the token fed, its position and the predicted token.
type feed struct {
	fn  string // "prefill", "forward" or "plain"
	tok int32
	pos int
	out int32
}

// recorded returns model as the function fn, logging each call to log.
func recorded(fn string, model func(int32, int) (int32, error), log *[]feed) func(int32, int) (int32, error) {
	return func(tok int32, pos int) (int32, error) {
		out, err := model(tok, pos)
		*log = append(*log, feed{fn, tok, pos, out})
		return out, err
	}
}

// TestGenerate is a behaviour test of generate, the prefill and decode of
// runGPUInference, with one model shared by its prefill, forward and plain
// functions, as runGPUInference shares one KV cache. It checks that the
// prefill feeds every prompt token at its position before any decode feed,
// that decoding starts at len(prompt) from the token predicted for the
// last prompt token, and that the output is plain greedy decoding of
// exactly the expected length, all shown, produced by speculative steps
// alone when drafting is on and by plain steps alone when it is off.
func TestGenerate(t *testing.T) {
	cycle := []int32{0, 1, 2, 3, 4, 0, 1}
	kvHash := func(m int) func(int32, int) (int32, error) { return (&fakeKV{kv: make([]int32, m)}).forward }
	cycleOf := func(m int) func(int32, int) (int32, error) { return cycleModel(m, 0) }
	for _, tt := range []struct {
		name          string
		model         func(maxSeq int) func(int32, int) (int32, error)
		prompt        []int32
		maxSeq, steps int
		draftN        int
		want          int // tokens generated
	}{
		{"plain", kvHash, []int32{1, 2, 3}, 24, 8, 0, 8},
		{"plain cache end", kvHash, []int32{1, 2, 3}, 9, 20, 0, 6},
		{"plain no steps", kvHash, []int32{1, 2, 3}, 24, 0, 0, 0},
		{"speculative", cycleOf, cycle, 32, 9, 3, 9},
		{"speculative cache end", cycleOf, cycle, 11, 20, 4, 4},
		{"speculative budget", cycleOf, cycle, 32, 2, 4, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			want, _, _, err := plainGreedy(tt.model(tt.maxSeq), tt.prompt, tt.maxSeq, tt.steps)
			if err != nil {
				t.Fatal(err)
			}
			if len(want) != tt.want {
				t.Fatalf("fixture: greedy decoding emits %d tokens, want %d", len(want), tt.want)
			}

			model := tt.model(tt.maxSeq)
			var log []feed
			var drafter *speculative.NgramDrafter
			if tt.draftN > 0 {
				drafter = speculative.NewNgramDrafter(3)
			}
			st := newDecodeState(tt.prompt, tt.steps, drafter)
			prefill := recorded("prefill", model, &log)
			forward := recorded("forward", model, &log)
			plainFeed := recorded("plain", model, &log)
			plain := func(step int, tok int32, pos int) (int32, error) { return plainFeed(tok, pos) }
			var shown []int32
			show := func(tok int32) error {
				shown = append(shown, tok)
				return nil
			}
			if _, _, err := generate(st, tt.prompt, prefill, forward, plain, show, tt.maxSeq, tt.steps, tt.draftN); err != nil {
				t.Fatal(err)
			}

			n := len(tt.prompt)
			if len(log) < n {
				t.Fatalf("%d forward passes, want at least the %d of the prefill", len(log), n)
			}
			for i, f := range log[:n] {
				if f.fn != "prefill" || f.tok != tt.prompt[i] || f.pos != i {
					t.Errorf("feed %d: %s(%d at %d), want prefill(%d at %d)", i, f.fn, f.tok, f.pos, tt.prompt[i], i)
				}
			}
			pending := log[n-1].out
			if pending == 0 {
				t.Fatal("fixture: the last prefill predicts 0, the zero pending token, so a lost prediction would go unseen")
			}
			counts := map[string]int{}
			for i, f := range log[n:] {
				counts[f.fn]++
				if f.pos != n+i {
					t.Errorf("decode feed %d: %s at %d, want position %d", i, f.fn, f.pos, n+i)
				}
			}
			if dec := log[n:]; len(dec) > 0 && dec[0].tok != pending {
				t.Errorf("first decode feed is token %d, want %d, the last prefill output", dec[0].tok, pending)
			}
			if !slices.Equal(st.generated, want) {
				t.Errorf("generated %v, want %v", st.generated, want)
			}
			if len(want) > 0 && st.generated[0] != pending {
				t.Errorf("first generated token %d, want %d, the last prefill output", st.generated[0], pending)
			}
			if !slices.Equal(shown, want) {
				t.Errorf("shown %v, want %v", shown, want)
			}
			wantCounts := map[string]int{"plain": tt.want}
			if tt.draftN > 0 {
				wantCounts = map[string]int{"forward": tt.want}
			}
			if counts["prefill"] != 0 || counts["forward"] != wantCounts["forward"] || counts["plain"] != wantCounts["plain"] {
				t.Errorf("decode feeds by function %v, want %v", counts, wantCounts)
			}
		})
	}
}

// TestGenerateErrors is a behaviour test of generate's errors. It wraps a
// prefill error with its position and runs nothing after it, and returns a
// decode error, from plain, a speculative forward or show, as decode
// returns it, with no forward pass after the failure.
func TestGenerateErrors(t *testing.T) {
	errFail := errors.New("forward failed")
	model := cycleModel(32, 0)
	// After the prompt the pending token is 2 at position 7.
	prompt := []int32{0, 1, 2, 3, 4, 0, 1}
	for _, tt := range []struct {
		name    string
		draftN  int
		failFn  string // the function that fails
		failAt  int    // the position it fails at, or for show, the call
		wantErr string
		feeds   int // forward passes, the failed one included
		shown   []int32
	}{
		{"prefill", 0, "prefill", 2, "gpu prefill step 2: prefill at 2: forward failed", 3, nil},
		{"plain", 0, "plain", 9, "plain at 9: forward failed", 10, []int32{2, 3, 4}},
		{"speculative", 3, "forward", 9, "speculative step at position 7: position 9: forward at 9: forward failed", 10, nil},
		{"show", 0, "show", 2, "show at 2: forward failed", 8, []int32{2, 3}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var returned error
			var log []feed
			failing := func(fn string) func(int32, int) (int32, error) {
				rec := recorded(fn, model, &log)
				return func(tok int32, pos int) (int32, error) {
					out, err := rec(tok, pos)
					if fn == tt.failFn && pos == tt.failAt {
						returned = fmt.Errorf("%s at %d: %w", fn, pos, errFail)
						return 0, returned
					}
					return out, err
				}
			}
			var drafter *speculative.NgramDrafter
			if tt.draftN > 0 {
				drafter = speculative.NewNgramDrafter(3)
			}
			st := newDecodeState(prompt, 16, drafter)
			plainFeed := failing("plain")
			plain := func(step int, tok int32, pos int) (int32, error) { return plainFeed(tok, pos) }
			var shown []int32
			show := func(tok int32) error {
				shown = append(shown, tok)
				if tt.failFn == "show" && len(shown) == tt.failAt {
					returned = fmt.Errorf("show at %d: %w", len(shown), errFail)
					return returned
				}
				return nil
			}
			_, _, err := generate(st, prompt, failing("prefill"), failing("forward"), plain, show, 32, 16, tt.draftN)
			if returned == nil {
				t.Fatal("fixture: nothing failed")
			}
			if !errors.Is(err, returned) || err.Error() != tt.wantErr {
				t.Errorf("err = %v, want %q wrapping the error %s returned", err, tt.wantErr, tt.failFn)
			}
			if len(log) != tt.feeds {
				t.Errorf("%d forward passes, want %d: %v", len(log), tt.feeds, log)
			}
			if !slices.Equal(shown, tt.shown) {
				t.Errorf("shown %v, want %v", shown, tt.shown)
			}
		})
	}
}

// TestGenerateWiring is a source-shape check, not a behaviour test: the
// GPU path that calls generate does not run without a device. It checks
// that runGPUInference passes generate its own prompt, closures, cache size,
// step budget and draft budget, returns generate's error, and that its
// prefill closure returns the token it read back.
func TestGenerateWiring(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"st := newDecodeState(promptTokens, *stepsFlag, drafter)\n",
		"prefillDuration, genDuration, err := generate(st, promptTokens, prefill, forward, plain, show, cfg.MaxSeqLen, *stepsFlag, *specDraftFlag)\n\tif err != nil {\n\t\treturn err\n\t}\n",
		"\t\t\tif err == nil {\n\t\t\t\ttok = scratch.HostTokenOut[0]\n\t\t\t}\n\t\t})\n\t\treturn tok, err\n\t}\n\tforward := func(",
	} {
		if n := strings.Count(string(src), want); n != 1 {
			t.Errorf("main.go contains %q %d times, want once", want, n)
		}
	}
}
