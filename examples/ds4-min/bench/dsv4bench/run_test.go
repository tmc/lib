package dsv4bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type benchState struct {
	tokens      []int
	closed      bool
	failRestore bool
}
type benchSnapshot struct {
	tokens []int
	closed bool
}

func (s *benchState) Reset() error { s.tokens = nil; return nil }
func (s *benchState) Snapshot() (dsv4.Snapshot, error) {
	return &benchSnapshot{tokens: append([]int(nil), s.tokens...)}, nil
}
func (s *benchState) Restore(v dsv4.Snapshot) error {
	if s.failRestore {
		return errors.New("restore failed")
	}
	s.tokens = append([]int(nil), v.(*benchSnapshot).tokens...)
	return nil
}
func (s *benchState) Save(io.Writer) error { return dsv4.ErrUnimplemented }
func (s *benchState) Load(io.Reader) error { return dsv4.ErrUnimplemented }
func (s *benchState) Close() error         { s.closed = true; return nil }
func (s *benchSnapshot) Close() error      { s.closed = true; return nil }

type benchBackend struct {
	s         *benchState
	evaluated [][]int
}

func (b *benchBackend) NewState(int) (dsv4.State, error) { b.s = &benchState{}; return b.s, nil }
func (b *benchBackend) Eval(v dsv4.State, token, pos int) error {
	s := v.(*benchState)
	if len(s.tokens) != pos {
		return errors.New("incorrect position")
	}
	s.tokens = append(s.tokens, token)
	b.evaluated = append(b.evaluated, append([]int(nil), s.tokens...))
	return nil
}
func (b *benchBackend) Logits(v dsv4.State) []float32 {
	s := v.(*benchState)
	return []float32{100, float32(len(s.tokens)), float32(len(s.tokens) + 1)}
}
func (b *benchBackend) Close() error { return nil }
func runFixture() RunConfig {
	return RunConfig{Prompt: []int{1, 2, 1, 2, 1, 2, 1, 2}, Start: 2, Max: 4, Step: 2, Generate: 2, EOS: 0, Multiplier: 1, Provenance: Provenance{Revision: "candidate", BinarySHA256: strings.Repeat("1", 64), ModelSHA256: strings.Repeat("2", 64), Host: "fixture", Backend: "fixture", Context: 8}}
}

func TestRunHistory(t *testing.T) {
	for _, replay := range []bool{false, true} {
		for _, forced := range []bool{false, true} {
			cfg := runFixture()
			cfg.Replay = replay
			cfg.TeacherForced = forced
			b := new(benchBackend)
			got, err := Run(context.Background(), b, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if !b.s.closed {
				t.Fatal("state not closed")
			}
			if len(got.Measurements) != 2 {
				t.Fatalf("measurements=%d", len(got.Measurements))
			}
			for i, m := range got.Measurements {
				frontier := 2 + i*2
				want := append([]int(nil), cfg.Prompt[:frontier]...)
				if forced {
					want = append(want, cfg.Prompt[frontier:frontier+2]...)
				} else {
					want = append(want, 2, 2)
				}
				if !reflect.DeepEqual(m.History, want) {
					t.Fatalf("history=%v want=%v", m.History, want)
				}
				if m.HistorySHA256 != historyHash(want) {
					t.Fatal("history identity")
				}
				if m.Row.PrefillTokens != 2 || m.Row.GenTokens != 2 || m.Row.GenSteadyTokens != 1 {
					t.Fatalf("wrong denominator: %+v", m.Row)
				}
				if m.Logits[1] != float32(frontier) {
					t.Fatal("frontier state was not restored")
				}
			}
		}
	}
}

func TestRunRejectsProvenanceAndCancellation(t *testing.T) {
	cfg := runFixture()
	cfg.Provenance.ModelSHA256 = ""
	b := new(benchBackend)
	if _, err := Run(context.Background(), b, cfg); err == nil || b.s != nil {
		t.Fatal("missing provenance admitted")
	}
	cfg = runFixture()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Run(ctx, b, cfg); !errors.Is(err, context.Canceled) || b.s != nil {
		t.Fatal("cancelled run admitted")
	}
}

func TestHistoryMutation(t *testing.T) {
	left := []int{1, 2, 3, 4}
	right := []int{2, 2, 3, 4}
	if historyHash(left) == historyHash(right) {
		t.Fatal("earlier forced-history mutation escaped")
	}
	want := sha256.Sum256([]byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0})
	if historyHash(left) != hex.EncodeToString(want[:]) {
		t.Fatal("history is not int32 little endian")
	}
}

type batchBenchBackend struct {
	benchBackend
	sizes []int
}

func (b *batchBenchBackend) EvalBatch(s dsv4.State, tokens []int, pos int) error {
	b.sizes = append(b.sizes, len(tokens))
	for i, token := range tokens {
		if err := b.Eval(s, token, pos+i); err != nil {
			return err
		}
	}
	return nil
}
func TestRunReplayChunkIdentity(t *testing.T) {
	cfg := runFixture()
	cfg.Replay = true
	cfg.Max = 6
	cfg.Generate = 1
	b := new(batchBenchBackend)
	if _, err := Run(context.Background(), b, cfg); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(b.sizes, []int{2, 2, 2, 2, 2, 2}) {
		t.Fatalf("replay merged frontier chunks: %v", b.sizes)
	}
}

func TestRunForcedEOS(t *testing.T) {
	cfg := runFixture()
	cfg.TeacherForced = true
	cfg.Prompt[2] = cfg.EOS
	got, err := Run(context.Background(), new(benchBackend), cfg)
	if err != nil {
		t.Fatal(err)
	}
	m := got.Measurements[0]
	if m.Row.GenTokens != 0 || !reflect.DeepEqual(m.History, []int{1, 2, 0}) {
		t.Fatalf("forced EOS denominator/history: %+v", m)
	}
}

type failBenchBackend struct{ benchBackend }

func (b *failBenchBackend) NewState(ctx int) (dsv4.State, error) {
	s, err := b.benchBackend.NewState(ctx)
	b.s.failRestore = true
	return s, err
}
func TestRunRestoreFailureCloses(t *testing.T) {
	b := new(failBenchBackend)
	got, err := Run(context.Background(), b, runFixture())
	if err == nil || len(got.Measurements) != 0 || !b.s.closed {
		t.Fatal("restore failure returned partial result or leaked state")
	}
}

type nanBenchBackend struct{ benchBackend }

func (b *nanBenchBackend) Logits(s dsv4.State) []float32 {
	if len(s.(*benchState).tokens) > 2 {
		return []float32{float32(math.NaN()), 1, 2}
	}
	return b.benchBackend.Logits(s)
}
func TestRunForcedRejectsNonfiniteOutput(t *testing.T) {
	for _, count := range []int{1, 2} {
		cfg := runFixture()
		cfg.TeacherForced = true
		cfg.Generate = count
		b := new(nanBenchBackend)
		got, err := Run(context.Background(), b, cfg)
		if err == nil || len(got.Measurements) != 0 || !b.s.closed {
			t.Fatal("nonfinite forced output admitted or state leaked")
		}
	}
}

type cancelBenchBackend struct {
	benchBackend
	cancel context.CancelFunc
}

func (b *cancelBenchBackend) EvalBatch(s dsv4.State, tokens []int, pos int) error {
	b.cancel()
	return nil
}
func TestRunCancellationAfterBatch(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	b := &cancelBenchBackend{cancel: cancel}
	cfg := runFixture()
	cfg.Generate = 0
	got, err := Run(ctx, b, cfg)
	if !errors.Is(err, context.Canceled) || len(got.Measurements) != 0 || !b.s.closed {
		t.Fatal("batch cancellation escaped or state leaked")
	}
}

func ExampleRun() {
	result, err := Run(context.Background(), new(benchBackend), runFixture())
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, m := range result.Measurements {
		fmt.Println(m.Row.CtxTokens, m.Row.PrefillTokens, m.History)
	}
	// Output:
	// 2 2 [1 2 2 2]
	// 4 2 [1 2 1 2 2 2]
}
