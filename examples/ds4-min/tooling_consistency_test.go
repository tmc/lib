package main

import (
	"bytes"
	"errors"
	"io"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type toolingBatchBackend struct {
	evaluationCLIBackend
	batches   [][]int
	mutateTop bool
}

func (b *toolingBatchBackend) Logits(state dsv4.State) []float32 {
	values := b.evaluationCLIBackend.Logits(state)
	if b.mutateTop && state.(*evaluationCLIState).id > 0 {
		values[2] = 100
	}
	return values
}

func (b *toolingBatchBackend) EvalBatch(state dsv4.State, tokens []int, pos int) error {
	b.batches = append(b.batches, append([]int(nil), tokens...))
	for i, token := range tokens {
		if err := b.Eval(state, token, pos+i); err != nil {
			return err
		}
	}
	return nil
}

func TestToolingFinalConsistency(t *testing.T) {
	b := new(toolingBatchBackend)
	state, _ := b.NewState(64)
	adapter := &stateSessionBackend{backend: b, state: state}
	live, _ := newSession(&batchStateSessionBackend{stateSessionBackend: adapter, batch: b}, 64, stopSet{2: true})
	prompt := []int{1, 1}
	var out, stdout bytes.Buffer
	if err := runDecodeConsistency(live, prompt, config{context: 64, decodeConsistency: 3, prefillChunk: 2048, log: &out}, &stdout); err != nil {
		t.Fatal(err)
	}
	if b.closed || len(b.batches) != 2 || len(b.batches[1]) != 5 {
		t.Fatalf("fresh did not batch full prefix on same foundation: %+v", b.batches)
	}
	if stdout.Len() != 0 {
		t.Fatal("C diagnostics leaked into stdout")
	}
	if !strings.Contains(out.String(), "positions=1") || !strings.Contains(out.String(), "live_top id=") || !strings.Contains(out.String(), "fresh_top id=") {
		t.Fatal("missing final/top diagnostics", out.String())
	}
}

func TestToolingConsistencyStopAndCleanup(t *testing.T) {
	for _, failClose := range []bool{false, true} {
		b := new(toolingBatchBackend)
		state, _ := b.NewState(64)
		adapter := &stateSessionBackend{backend: b, state: state}
		live, _ := newSession(&batchStateSessionBackend{stateSessionBackend: adapter, batch: b}, 64, stopSet{1: true})
		if err := live.Sync([]int{1, 1}); err != nil {
			t.Fatal(err)
		}
		if failClose {
			b.closeStateErr = io.ErrClosedPipe
		}
		var out bytes.Buffer
		err := runDecodeConsistency(live, []int{1, 1}, config{context: 64, decodeConsistency: 4, prefillChunk: 2048}, &out)
		if failClose {
			if !errors.Is(err, io.ErrClosedPipe) || out.Len() != 0 {
				t.Fatal("cleanup error published diagnostics", err)
			}
		} else if err != nil {
			t.Fatal(err)
		}
		if len(live.tokens) != 3 || len(b.batches[1]) != 3 || b.closed {
			t.Fatal("stop/foundation semantics changed")
		}
	}
}

func TestToolingConsistencyDisagreement(t *testing.T) {
	b := &toolingBatchBackend{mutateTop: true}
	state, _ := b.NewState(64)
	adapter := &stateSessionBackend{backend: b, state: state}
	live, _ := newSession(&batchStateSessionBackend{stateSessionBackend: adapter, batch: b}, 64, stopSet{2: true})
	var diagnostics, stdout bytes.Buffer
	err := runDecodeConsistency(live, []int{1, 1}, config{context: 64, decodeConsistency: 1, prefillChunk: 2048, log: &diagnostics}, &stdout)
	if err == nil || !strings.Contains(err.Error(), "final top1 differs") || !strings.Contains(diagnostics.String(), "top1=0/1") || stdout.Len() != 0 {
		t.Fatal("numerical disagreement hidden", err)
	}
}

func TestToolingPerplexityCaps(t *testing.T) {
	for _, tc := range []struct {
		name         string
		ctx, n, want int
	}{{"n-cap", 100, 2, 2}, {"context-cap", 34, 0, 2}, {"unlimited", 100, 0, 8}} {
		t.Run(tc.name, func(t *testing.T) {
			b := new(evaluationCLIBackend)
			state, _ := b.NewState(tc.ctx)
			s, _ := newSession(&stateSessionBackend{backend: b, state: state}, tc.ctx, stopSet{2: true})
			ids := make([]int, 40)
			for i := range ids {
				ids[i] = 1
			}
			var out bytes.Buffer
			if err := runPerplexity(s, ids, config{tokens: tc.n}, &out); err != nil {
				t.Fatal(err)
			}
			if state.(*evaluationCLIState).position != 32+tc.want-1 {
				t.Fatal("last score was evaluated or cap off by one")
			}
			want := "tokens=40 scored=" + strconv.Itoa(tc.want)
			if !strings.Contains(out.String(), want) {
				t.Fatal("wrong denominator", out.String())
			}
		})
	}
}
