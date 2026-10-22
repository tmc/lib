package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/bench/dsv4bench"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type evaluationCLIState struct {
	stateSessionBackendTestState
	position, id int
	owner        *evaluationCLIBackend
}

func (s *evaluationCLIState) Close() error {
	return s.owner.closeStateErr
}

type evaluationCLIBackend struct {
	states        int
	mutate        bool
	closed        bool
	closeErr      error
	evalErr       error
	closeStateErr error
	beforeClose   func()
}

func (b *evaluationCLIBackend) NewState(int) (dsv4.State, error) {
	s := &evaluationCLIState{id: b.states, owner: b}
	b.states++
	return s, nil
}

func TestEvaluationCLIChildCleanup(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	cause := errors.New("child cleanup failed")
	b := &evaluationCLIBackend{closeStateErr: cause}
	var out bytes.Buffer
	err := produceEvaluation(context.Background(), b, b, evaluationCLIConfig(), path, &out, 0, 0)
	if !errors.Is(err, cause) || b.closed || out.Len() != 0 {
		t.Fatal("child failed cleanup released backend or published", err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("failed child cleanup retained receipt")
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("child cleanup retry owner lost")
	}
	b.closeStateErr = nil
	if err := owner.Close(); err != nil || !b.closed {
		t.Fatal("ordered cleanup retry did not release backend", err)
	}
}
func (b *evaluationCLIBackend) Eval(v dsv4.State, token, pos int) error {
	if b.evalErr != nil {
		return b.evalErr
	}
	s := v.(*evaluationCLIState)
	if s.position != pos {
		return errors.New("wrong position")
	}
	s.position++
	return nil
}

type evaluationFailWriter struct{}

func (evaluationFailWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

type evaluationShortWriter struct{}

func (evaluationShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestEvaluationCLIShortWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	b := new(evaluationCLIBackend)
	err := produceEvaluation(context.Background(), b, b, evaluationCLIConfig(), path, evaluationShortWriter{}, 0, 0)
	if !errors.Is(err, io.ErrShortWrite) || !b.closed {
		t.Fatal("short output admitted", err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("short output retained receipt")
	}
}

func TestEvaluationCLIFailedExecution(t *testing.T) {
	for _, fail := range []string{"eval", "output"} {
		t.Run(fail, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "receipt.json")
			b := new(evaluationCLIBackend)
			var out bytes.Buffer
			var writer io.Writer = &out
			if fail == "eval" {
				b.evalErr = io.ErrUnexpectedEOF
			} else {
				writer = evaluationFailWriter{}
			}
			if err := produceEvaluation(context.Background(), b, b, evaluationCLIConfig(), path, writer, 0, 0); err == nil {
				t.Fatal("failed command passed")
			}
			if !b.closed || out.Len() != 0 {
				t.Fatal("cleanup/output failure")
			}
			if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("failed execution published receipt")
			}
		})
	}
}
func (b *evaluationCLIBackend) Logits(v dsv4.State) []float32 {
	s := v.(*evaluationCLIState)
	x := float32(s.position)
	if b.mutate && s.id > 0 {
		x++
	}
	return []float32{0, x, 2}
}
func (b *evaluationCLIBackend) Close() error {
	if b.beforeClose != nil {
		b.beforeClose()
	}
	b.closed = true
	return b.closeErr
}

func evaluationCLIConfig() dsv4bench.EvaluationConfig {
	return dsv4bench.EvaluationConfig{Prompt: []int{1, 2}, Continuation: []int{2, 1}, Provenance: dsv4bench.Provenance{Revision: "fixture", BinarySHA256: strings.Repeat("1", 64), ModelSHA256: strings.Repeat("2", 64), Host: "fixture", Backend: "fixture", Context: 8}}
}

func TestEvaluationCLIReceipts(t *testing.T) {
	for _, mutate := range []bool{false, true} {
		t.Run(map[bool]string{false: "pass", true: "numerical-fail"}[mutate], func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "receipt.json")
			b := &evaluationCLIBackend{mutate: mutate}
			b.beforeClose = func() {
				if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
					t.Fatal("receipt published before cleanup")
				}
			}
			var out bytes.Buffer
			err := produceEvaluation(context.Background(), b, b, evaluationCLIConfig(), path, &out, 0, 0)
			if (err != nil) != mutate || !b.closed {
				t.Fatalf("result err=%v closed=%v", err, b.closed)
			}
			data, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(data, out.Bytes()) {
				t.Fatal("missing output", err)
			}
			var receipt evaluationCLIReceipt
			if err := json.Unmarshal(data, &receipt); err != nil {
				t.Fatal(err)
			}
			want := "PASS"
			if mutate {
				want = "FAIL"
			}
			if !receipt.Complete || receipt.Ran != 2 || receipt.ConsistencyVerdict != want || receipt.Provenance.ModelSHA256 != evaluationCLIConfig().Provenance.ModelSHA256 {
				t.Fatalf("bad receipt %+v", receipt)
			}
		})
	}
}

func TestEvaluationCLICleanupFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	cause := errors.New("cleanup failed")
	b := &evaluationCLIBackend{closeErr: cause}
	var out bytes.Buffer
	err := produceEvaluation(context.Background(), b, b, evaluationCLIConfig(), path, &out, 0, 0)
	if !errors.Is(err, cause) || out.Len() != 0 {
		t.Fatal("cleanup failure hidden", err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("failed cleanup published receipt")
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("retry owner lost")
	}
	b.closeErr = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestEvaluationCLIAdmission(t *testing.T) {
	for _, args := range [][]string{nil, {"-max-abs", "NaN"}, {"-model", "fixture"}} {
		var out, stderr bytes.Buffer
		if err := runFrontierBenchmark(context.Background(), append([]string{"-evaluate"}, args...), &out, &stderr); err == nil || out.Len() != 0 {
			t.Fatal("bad invocation admitted", err)
		}
	}
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(path, []byte("original"), 0600); err != nil {
		t.Fatal(err)
	}
	b := new(evaluationCLIBackend)
	if err := produceEvaluation(context.Background(), b, b, evaluationCLIConfig(), path, io.Discard, 0, 0); err == nil {
		t.Fatal("existing receipt overwritten")
	}
	data, _ := os.ReadFile(path)
	if string(data) != "original" {
		t.Fatal("existing artifact changed")
	}
}

func TestEvaluationCLIConstructorOwnership(t *testing.T) {
	cause := errors.New("constructor failed")
	b := new(evaluationCLIBackend)
	if err := closeEvaluationBackend(cause, b); !errors.Is(err, cause) || !b.closed {
		t.Fatal("constructor owner lost", err)
	}
	b = new(evaluationCLIBackend)
	if err := produceEvaluation(context.Background(), b, b, evaluationCLIConfig(), "", io.Discard, 0, 0); err == nil || !b.closed {
		t.Fatal("invalid invocation leaked transferred owner", err)
	}
}
