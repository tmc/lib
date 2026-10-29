package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

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
// GPU path that calls generate and runBenchmark does not run without a
// device. It checks that runGPUInference passes runBenchmark its bench
// closure and arguments and returns its error, and passes generate its own
// prompt, closures, cache size, step budget and draft budget, returns
// generate's error, and that its prefill closure returns the token it read
// back.
func TestGenerateWiring(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"\tif *benchFlag {\n\t\tforward := func(tok int32, pos int) (int32, error) {\n\t\t\tif dg != nil && pos >= len(promptTokens) {\n\t\t\t\treturn dg.Step(pos, nil)\n\t\t\t}\n\t\t\treturn GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tok, pos)\n\t\t}\n\t\treturn runBenchmark(os.Stdout, dtype, promptTokens, *stepsFlag, *benchCountFlag, cfg.MaxSeqLen, kvCache.Reset, forward)\n\t}\n",
		"st := newDecodeState(promptTokens, *stepsFlag, drafter)\n",
		"prefillDuration, genDuration, err := generate(st, promptTokens, prefill, forward, plain, show, cfg.MaxSeqLen, *stepsFlag, *specDraftFlag)\n\tif err != nil {\n\t\treturn err\n\t}\n",
		"\t\t\tif err == nil {\n\t\t\t\ttok = scratch.HostTokenOut[0]\n\t\t\t}\n\t\t})\n\t\treturn tok, err\n\t}\n\tforward := func(",
		"\tif len(st.generated) > 0 {\n\t\ttpot := genDuration / time.Duration(len(st.generated))\n\t\tif bw := formatBenchmarkBandwidth(gpuWeights.StreamedBytesPerToken(), tpot); bw != \"\" {\n\t\t\tfmt.Printf(\"Bandwidth    : %s\\n\", bw)\n\t\t}\n\t}\n",
	} {
		if n := strings.Count(string(src), want); n != 1 {
			t.Errorf("main.go contains %q %d times, want once", want, n)
		}
	}
}

// TestProfilerStopWiring checks in the source that the GPU path stops an
// attached profiler after runGPUInference returns, inside the device
// context: Nsight Systems flushes from a C atexit handler, which a Go exit
// skips, and its report of a run without the stop holds no kernels.
func TestProfilerStopWiring(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	want := "\t\treturn dev.WithContext(func() error {\n" +
		"\t\t\terr := runGPUInference(dev, ck, cfg, tied, tokenizer, promptTokens, major, minor, tracer, profSession)\n"
	stop := "\t\t\treturn errors.Join(err, driver.ProfilerStop())\n\t\t})\n"
	i := strings.Index(string(src), want)
	if i < 0 || strings.Count(string(src), want) != 1 {
		t.Fatalf("main.go does not call runGPUInference once in dev.WithContext")
	}
	if j := strings.Index(string(src[i:]), stop); j < 0 || strings.Contains(string(src[i:i+j]), "})") {
		t.Errorf("main.go does not return driver.ProfilerStop's error from the dev.WithContext closure that runs runGPUInference")
	}
}

// TestRunBenchmark is a CPU behaviour test of runBenchmark. It checks that
// warmup feeds every prompt token at its position once, that each benchmark
// iteration resets the KV cache before running generate (prefill then
// decode from len(prompt) with the last prefill prediction), and that TTFT,
// TPOT and decode lines are written for every iteration.
func TestRunBenchmark(t *testing.T) {
	const maxSeq = 16
	prompt := []int32{1, 2, 3}
	want, _, _, err := plainGreedy((&fakeKV{kv: make([]int32, maxSeq)}).forward, prompt, maxSeq, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(want) != 4 {
		t.Fatalf("fixture: greedy decoding emits %d tokens, want 4", len(want))
	}

	f := &fakeKV{kv: make([]int32, maxSeq)}
	var events []string
	var feeds []feed
	reset := func() {
		clear(f.kv)
		events = append(events, "reset")
	}
	forward := func(tok int32, pos int) (int32, error) {
		out, err := f.forward(tok, pos)
		events = append(events, fmt.Sprintf("fwd(%d@%d->%d)", tok, pos, out))
		feeds = append(feeds, feed{"forward", tok, pos, out})
		return out, err
	}
	var out strings.Builder
	if err := runBenchmark(&out, weightBF16, prompt, 4, 2, maxSeq, reset, forward); err != nil {
		t.Fatal(err)
	}

	// Warmup is 3 forwards; each of the 2 iterations resets and runs 3
	// prefill + 4 decode = 7 forwards.
	if len(feeds) != 3+2*7 {
		t.Fatalf("%d forwards, want %d: %v", len(feeds), 3+2*7, feeds)
	}
	for i, p := range prompt {
		if feeds[i].tok != p || feeds[i].pos != i {
			t.Errorf("warmup %d: %+v, want tok %d at %d", i, feeds[i], p, i)
		}
	}
	for iter := range 2 {
		if got := events[3+iter*8]; got != "reset" {
			t.Errorf("iter %d event before prefill = %q, want reset (events: %v)", iter, got, events)
		}
		base := 3 + iter*7
		for i, p := range prompt {
			if feeds[base+i].tok != p || feeds[base+i].pos != i {
				t.Errorf("iter %d prefill %d: %+v, want tok %d at %d", iter, i, feeds[base+i], p, i)
			}
		}
		for step := range 4 {
			got := feeds[base+len(prompt)+step]
			if got.tok != want[step] || got.pos != len(prompt)+step {
				t.Errorf("iter %d decode %d: %+v, want tok %d at %d", iter, step, got, want[step], len(prompt)+step)
			}
		}
	}

	lines := strings.Split(strings.TrimSuffix(out.String(), "\n"), "\n")
	if len(lines) != 6 {
		t.Fatalf("got %d output lines, want 6:\n%s", len(lines), out.String())
	}
	wantPrefix := "BenchmarkLlama32_1B/backend=cuda/weights=bf16/gen_tokens=16/prompt=neural_network/search_graphs=100/"
	for i, suffix := range []string{"TTFT-1 1 ", "TPOT-1 1 ", "decode-1 1 ", "TTFT-1 1 ", "TPOT-1 1 ", "decode-1 1 "} {
		if !strings.HasPrefix(lines[i], wantPrefix+suffix) {
			t.Errorf("line %d = %q, want prefix %q", i, lines[i], wantPrefix+suffix)
		}
	}

	// When maxSeq caps decode before steps tokens, decode stops at maxSeq.
	fCap := &fakeKV{kv: make([]int32, 5)}
	var capFeeds int
	var capOut strings.Builder
	if err := runBenchmark(&capOut, weightF32, prompt, 10, 1, 5, func() { clear(fCap.kv) }, func(tok int32, pos int) (int32, error) {
		capFeeds++
		return fCap.forward(tok, pos)
	}); err != nil {
		t.Fatalf("capped benchmark: %v", err)
	}
	// 3 warmup + 3 prefill + 2 decode (positions 3 and 4) = 8 forwards.
	if capFeeds != 8 {
		t.Errorf("capped benchmark forwards = %d, want 8", capFeeds)
	}
}

// TestRunBenchmarkErrors checks that runBenchmark refuses steps <= 0 before
// any forward pass or reset, refuses a zero-token decode when prompt fills
// maxSeq, and returns warmup, prefill, decode and output write errors.
func TestRunBenchmarkErrors(t *testing.T) {
	for _, steps := range []int{0, -1, -10} {
		called := false
		reset := func() { called = true }
		forward := func(int32, int) (int32, error) {
			called = true
			return 0, nil
		}
		var out strings.Builder
		err := runBenchmark(&out, weightF32, []int32{1, 2}, steps, 2, 16, reset, forward)
		if err == nil || !strings.Contains(err.Error(), "-steps > 0") {
			t.Errorf("steps=%d: err = %v, want -steps > 0 refusal", steps, err)
		}
		if called || out.Len() > 0 {
			t.Errorf("steps=%d: called=%v, out=%q; want no work before refusal", steps, called, out.String())
		}
	}

	// When len(prompt) >= maxSeq, decode emits 0 tokens even with steps > 0;
	// runBenchmark must refuse rather than divide by zero.
	var out strings.Builder
	err := runBenchmark(&out, weightF32, []int32{1, 2, 3}, 4, 1, 3, func() {}, func(int32, int) (int32, error) { return 1, nil })
	if err == nil || !strings.Contains(err.Error(), "generated 0 tokens") {
		t.Errorf("full cache: err = %v, want 0-token refusal", err)
	}

	errBoom := errors.New("forward failed")
	for _, tt := range []struct {
		name     string
		failCall int // 1-based forward call that fails
		wantErr  string
	}{
		{"warmup", 2, "warmup step 1: forward failed"},
		{"prefill", 4, "gpu prefill step 1: forward failed"},
		{"decode", 6, "decode step 3: forward failed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			forward := func(tok int32, pos int) (int32, error) {
				calls++
				if calls == tt.failCall {
					return 0, errBoom
				}
				return 1, nil
			}
			var buf strings.Builder
			err := runBenchmark(&buf, weightF32, []int32{1, 2}, 3, 2, 16, func() {}, forward)
			if !errors.Is(err, errBoom) || err.Error() != tt.wantErr {
				t.Errorf("err = %v, want %q wrapping %v", err, tt.wantErr, errBoom)
			}
			if calls != tt.failCall {
				t.Errorf("calls = %d, want %d", calls, tt.failCall)
			}
		})
	}

	// Output writer error is returned.
	err = runBenchmark(failWriter{}, weightF32, []int32{1, 2}, 2, 2, 16, func() {}, func(int32, int) (int32, error) { return 1, nil })
	if !errors.Is(err, errFakeWrite) {
		t.Errorf("write error: got %v, want %v", err, errFakeWrite)
	}
}

// TestFormatBenchmarkBandwidth tests achievedBandwidthGBps and
// formatBenchmarkBandwidth across valid streamed byte sizes and TPOT
// durations as well as zero and negative guards.
func TestFormatBenchmarkBandwidth(t *testing.T) {
	for _, tt := range []struct {
		name          string
		streamedBytes int64
		tpot          time.Duration
		wantGBps      float64
		wantText      string
	}{
		{
			name:          "1B BF16 2.47 GB in 10ms",
			streamedBytes: 2_470_000_000,
			tpot:          10 * time.Millisecond,
			wantGBps:      247.0,
			wantText:      "2470000000 bytes/tok (247.00 GB/s)",
		},
		{
			name:          "1B F32 4.94 GB in 20ms",
			streamedBytes: 4_940_000_000,
			tpot:          20 * time.Millisecond,
			wantGBps:      247.0,
			wantText:      "4940000000 bytes/tok (247.00 GB/s)",
		},
		{
			name:          "8B BF16 15.0 GB in 60ms",
			streamedBytes: 15_000_000_000,
			tpot:          60 * time.Millisecond,
			wantGBps:      250.0,
			wantText:      "15000000000 bytes/tok (250.00 GB/s)",
		},
		{
			name:          "zero bytes guard",
			streamedBytes: 0,
			tpot:          10 * time.Millisecond,
			wantGBps:      0,
			wantText:      "",
		},
		{
			name:          "negative bytes guard",
			streamedBytes: -1024,
			tpot:          10 * time.Millisecond,
			wantGBps:      0,
			wantText:      "",
		},
		{
			name:          "zero duration guard",
			streamedBytes: 2_470_000_000,
			tpot:          0,
			wantGBps:      0,
			wantText:      "",
		},
		{
			name:          "negative duration guard",
			streamedBytes: 2_470_000_000,
			tpot:          -5 * time.Millisecond,
			wantGBps:      0,
			wantText:      "",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			gotGBps := achievedBandwidthGBps(tt.streamedBytes, tt.tpot)
			if math.Abs(gotGBps-tt.wantGBps) > 1e-9 {
				t.Errorf("achievedBandwidthGBps(%d, %v) = %g, want %g", tt.streamedBytes, tt.tpot, gotGBps, tt.wantGBps)
			}
			gotText := formatBenchmarkBandwidth(tt.streamedBytes, tt.tpot)
			if gotText != tt.wantText {
				t.Errorf("formatBenchmarkBandwidth(%d, %v) = %q, want %q", tt.streamedBytes, tt.tpot, gotText, tt.wantText)
			}
		})
	}
}
