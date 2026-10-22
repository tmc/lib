package main

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4server"
)

func TestServerSteeringHTTP(t *testing.T) {
	raw := new(serverSteeringBackend)
	b := serverSteeringFixture(raw)
	m := &serverModel{backend: b, tok: maxPrefixTokenizer(), context: 1024, stops: dsv4.NewStopSet(9)}
	h, err := dsv4server.New(m, dsv4server.Config{Model: "test", ContextLength: 1024, MaxTokens: 1, SessionSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0}`))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("HTTP %d: %s", rec.Code, rec.Body.String())
		}
	}
	if len(raw.created) != 2 || len(raw.evaluated) == 0 {
		t.Fatal("HTTP did not use initialized request States")
	}
	for _, s := range raw.created {
		if s.vectors != b.vectors || s.ffn != 1 {
			t.Fatal("HTTP State lacks steering")
		}
	}
	if err := h.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestServerSteeringAdmission(t *testing.T) {
	for _, tt := range []struct {
		name string
		args []string
		want string
	}{
		{"help", []string{"-help"}, "flag: help requested"},
		{"nonfinite", []string{"-dir-steering-ffn", "NaN"}, "steering"},
		{"missing-directions", []string{"-dir-steering-ffn", "1"}, "steering"},
		{"zero-skips-directions", []string{"-dir-steering-file", "missing", "-dir-steering-ffn", "0", "-dir-steering-attn", "0"}, "model is required"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var output bytes.Buffer
			err := runServer(tt.args, &output, &output)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("runServer = %v", err)
			}
			if tt.name == "help" && !strings.Contains(output.String(), "dir-steering-file") {
				t.Fatal("steering flags omitted from help")
			}
		})
	}
}

type serverSteeringState struct {
	stateSessionBackendTestState
	vectors         *steeringVectors
	attention, ffn  float32
	loadErr         error
	transitionErr   error
	partialSnapshot dsv4.Snapshot
	closeFailures   int
}

func (s *serverSteeringState) loadSteering(v *steeringVectors, attention, ffn float32) error {
	s.vectors, s.attention, s.ffn = v, attention, ffn
	return s.loadErr
}
func (s *serverSteeringState) steeringFFN() float32           { return s.ffn }
func (s *serverSteeringState) setSteeringFFN(v float32) error { s.ffn = v; return s.transitionErr }
func (s *serverSteeringState) Reset() error                   { return s.transitionErr }
func (s *serverSteeringState) Snapshot() (dsv4.Snapshot, error) {
	return s.partialSnapshot, s.transitionErr
}
func (s *serverSteeringState) Restore(dsv4.Snapshot) error { return s.transitionErr }
func (s *serverSteeringState) Save(io.Writer) error        { return s.transitionErr }
func (s *serverSteeringState) Load(io.Reader) error        { return s.transitionErr }
func (s *serverSteeringState) Close() error {
	if s.closeFailures > 0 {
		s.closeFailures--
		return io.ErrUnexpectedEOF
	}
	return s.stateSessionBackendTestState.Close()
}

type serverSteeringBackend struct {
	stateSessionBackendTestBackend
	created       []*serverSteeringState
	loadErr       error
	closeFailures int
	evaluated     []dsv4.State
	evalErr       error
	evalStarted   chan struct{}
	evalResume    chan struct{}
	newErr        error
}

func (b *serverSteeringBackend) NewState(int) (dsv4.State, error) {
	s := &serverSteeringState{loadErr: b.loadErr, closeFailures: b.closeFailures}
	b.created = append(b.created, s)
	return s, b.newErr
}
func (b *serverSteeringBackend) Eval(s dsv4.State, token, position int) error {
	b.evaluated = append(b.evaluated, s)
	if b.evalStarted != nil {
		close(b.evalStarted)
		<-b.evalResume
	}
	return b.evalErr
}

func (b *serverSteeringBackend) Logits(dsv4.State) []float32 {
	return []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
}

func serverSteeringFixture(raw *serverSteeringBackend) *steeringBackend {
	return &steeringBackend{IndependentStates: raw, vectors: &steeringVectors{values: []float32{2, 3}, layers: 1, width: 2}, attention: 0.5, ffn: 1,
		states: make(map[*steeringRequestState]bool), pending: make(map[*steeringRequestCleanup]bool)}
}

func TestServerSteeringFreshSessions(t *testing.T) {
	raw := new(serverSteeringBackend)
	b := serverSteeringFixture(raw)
	m := &serverModel{backend: b, context: 32, stops: dsv4.NewStopSet(9)}
	a, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	c, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	if len(raw.created) != 2 || raw.created[0] == raw.created[1] {
		t.Fatal("requests share State")
	}
	for _, s := range raw.created {
		if s.vectors != b.vectors || s.attention != 0.5 || s.ffn != 1 {
			t.Fatal("request published without startup steering")
		}
	}
	for s := range b.states {
		if s.state == raw.created[0] {
			if err := s.setSteeringFFN(-2); err != nil {
				t.Fatal(err)
			}
		}
	}
	if raw.created[0].ffn != -2 || raw.created[1].ffn != 1 || b.ffn != 1 {
		t.Fatal("request controls leak")
	}
	if err := a.Sync([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := c.Sync([]int{3}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(raw.evaluated, []dsv4.State{raw.created[0], raw.created[0], raw.created[1]}) {
		t.Fatal("backend received wrapper or wrong request")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestServerSteeringFailedInitializationRetainsOwner(t *testing.T) {
	raw := &serverSteeringBackend{loadErr: io.ErrShortWrite, closeFailures: 1}
	b := serverSteeringFixture(raw)
	s, err := b.NewState(32)
	if s != nil || !errors.Is(err, io.ErrShortWrite) || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("NewState = %v, %v", s, err)
	}
	var owner io.Closer
	if !errors.As(err, &owner) || len(b.pending) != 1 {
		t.Fatal("failed candidate owner lost")
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if len(b.pending) != 0 || !raw.created[0].closed {
		t.Fatal("retry did not release candidate")
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestServerSteeringShutdownOrdersFoundation(t *testing.T) {
	raw := &serverSteeringBackend{closeFailures: 1}
	b := serverSteeringFixture(raw)
	s, err := b.NewState(32)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(err)
	}
	if raw.close {
		t.Fatal("foundation closed before request")
	}
	if err := b.Eval(s, 1, 0); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal(err)
	}
	if _, err := b.NewState(32); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if !raw.close || !raw.created[0].closed {
		t.Fatal("retry lost resources")
	}
}

func TestServerSteeringEvaluationCleanup(t *testing.T) {
	var events []string
	original := &ownedCleanupFixture{&backendCleanupFixture{"evaluation", io.ErrShortWrite, &events}, io.EOF}
	raw := &serverSteeringBackend{evalErr: original}
	b := serverSteeringFixture(raw)
	s, err := b.NewState(32)
	if err != nil {
		t.Fatal(err)
	}
	err = b.Eval(s, 1, 0)
	var owner io.Closer
	if !errors.Is(err, io.EOF) || !errors.As(err, &owner) {
		t.Fatal("evaluation cleanup boundary lost")
	}
	if err := b.Close(); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	if raw.close {
		t.Fatal("foundation closed while evaluation owns resources")
	}
	original.failure = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 || !raw.close {
		t.Fatalf("cleanup retries = %v", events)
	}
}

func TestServerSteeringConcurrentShutdownRetainsEvaluation(t *testing.T) {
	var events []string
	original := &ownedCleanupFixture{&backendCleanupFixture{"evaluation", io.ErrShortWrite, &events}, io.EOF}
	raw := &serverSteeringBackend{evalErr: original, evalStarted: make(chan struct{}), evalResume: make(chan struct{})}
	b := serverSteeringFixture(raw)
	s, err := b.NewState(32)
	if err != nil {
		t.Fatal(err)
	}
	evaluation := make(chan error, 1)
	go func() { evaluation <- b.Eval(s, 1, 0) }()
	<-raw.evalStarted
	// Hold the State lock so Close can snapshot the ledger, then block on
	// the active evaluation. Publish its cleanup owner after that snapshot.
	closing := make(chan error, 1)
	go func() { closing <- b.Close() }()
	for {
		b.mu.Lock()
		started := b.closing
		b.mu.Unlock()
		if started {
			break
		}
		runtime.Gosched()
	}
	close(raw.evalResume)
	if err := <-evaluation; !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
	if err := <-closing; err == nil {
		t.Fatal("shutdown lost late evaluation cleanup")
	}
	if raw.close {
		t.Fatal("foundation closed before late owner")
	}
	original.failure = nil
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if !raw.close {
		t.Fatal("foundation not released after retry")
	}
}

func TestServerSteeringConcurrentClose(t *testing.T) {
	raw := new(serverSteeringBackend)
	b := serverSteeringFixture(raw)
	if _, err := b.NewState(32); err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	for range 2 {
		group.Add(1)
		go func() {
			defer group.Done()
			if err := b.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	group.Wait()
	if !raw.close || len(b.states) != 0 {
		t.Fatal("concurrent shutdown leaked resources")
	}
}

func TestServerSteeringTransitionCleanup(t *testing.T) {
	for _, operation := range []string{"reset", "snapshot", "restore", "save", "load", "control"} {
		t.Run(operation, func(t *testing.T) {
			var events []string
			original := &ownedCleanupFixture{&backendCleanupFixture{operation, io.ErrShortWrite, &events}, io.EOF}
			raw := new(serverSteeringBackend)
			b := serverSteeringFixture(raw)
			state, err := b.NewState(32)
			if err != nil {
				t.Fatal(err)
			}
			s := state.(*steeringRequestState)
			raw.created[0].transitionErr = original
			switch operation {
			case "reset":
				err = s.Reset()
			case "snapshot":
				_, err = s.Snapshot()
			case "restore":
				err = s.Restore(nil)
			case "save":
				err = s.Save(io.Discard)
			case "load":
				err = s.Load(strings.NewReader(""))
			case "control":
				err = s.setSteeringFFN(2)
			}
			var owner io.Closer
			if !errors.Is(err, io.EOF) || !errors.As(err, &owner) || len(b.pending) != 1 {
				t.Fatal("transition cleanup owner lost")
			}
			// Deliberately discard the returned error: the backend must still
			// retain its owner, including a failed Snapshot backup cleanup.
			if err := b.Close(); !errors.Is(err, io.ErrShortWrite) {
				t.Fatal(err)
			}
			if raw.close {
				t.Fatal("foundation released before transition owner")
			}
			original.failure = nil
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
			if len(events) != 2 || !raw.close {
				t.Fatal("transition retry lost owner")
			}
		})
	}
}

func TestServerSteeringPartialConstructionOwner(t *testing.T) {
	raw := &serverSteeringBackend{newErr: io.EOF, closeFailures: 1}
	b := serverSteeringFixture(raw)
	s, err := b.NewState(32)
	var owner io.Closer
	if s != nil || !errors.Is(err, io.EOF) || !errors.As(err, &owner) {
		t.Fatalf("partial construction = %v, %v", s, err)
	}
	if err := b.Close(); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(err)
	}
	if raw.close || raw.created[0].vectors != nil {
		t.Fatal("partial candidate published or foundation released")
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if !raw.close || !raw.created[0].closed {
		t.Fatal("partial candidate ownership lost")
	}
}

func TestServerSteeringPartialSnapshotOwner(t *testing.T) {
	for _, exposed := range []bool{false, true} {
		t.Run(map[bool]string{false: "separate", true: "already-exposed"}[exposed], func(t *testing.T) {
			var events []string
			snapshot := &ownedCleanupFixture{&backendCleanupFixture{"snapshot", io.ErrShortWrite, &events}, io.EOF}
			raw := new(serverSteeringBackend)
			b := serverSteeringFixture(raw)
			state, err := b.NewState(32)
			if err != nil {
				t.Fatal(err)
			}
			raw.created[0].partialSnapshot = snapshot
			raw.created[0].transitionErr = io.EOF
			if exposed {
				raw.created[0].transitionErr = errors.Join(io.EOF, snapshot)
			}
			partial, err := state.Snapshot()
			var owner io.Closer
			if partial != nil || !errors.Is(err, io.EOF) || !errors.As(err, &owner) || len(b.pending) != 1 {
				t.Fatal("partial snapshot cleanup not transferred")
			}
			for pending := range b.pending {
				if len(pending.owners.owners) != 1 {
					t.Fatal("snapshot owner duplicated")
				}
			}
			// Discard both returned values, as a failed handler can do.
			if err := b.Close(); !errors.Is(err, io.ErrShortWrite) {
				t.Fatal(err)
			}
			if raw.close || len(events) != 1 {
				t.Fatal("snapshot duplicated or foundation released early")
			}
			snapshot.failure = nil
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
			if !raw.close || len(events) != 2 {
				t.Fatal("partial snapshot lost or released twice")
			}
		})
	}
}

func TestServerSteeringSuccessfulSnapshotRemainsCallerOwned(t *testing.T) {
	var events []string
	snapshot := &backendCleanupFixture{"snapshot", nil, &events}
	raw := new(serverSteeringBackend)
	b := serverSteeringFixture(raw)
	state, err := b.NewState(32)
	if err != nil {
		t.Fatal(err)
	}
	raw.created[0].partialSnapshot = snapshot
	got, err := state.Snapshot()
	if err != nil || got != snapshot || len(b.pending) != 0 {
		t.Fatal("successful snapshot ownership changed")
	}
	if err := got.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 {
		t.Fatal("caller-owned snapshot closed again")
	}
}
