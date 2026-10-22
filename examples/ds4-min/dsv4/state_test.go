package dsv4

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

type contractBackend struct{}

func (contractBackend) NewState(int) (State, error)       { return nil, ErrUnimplemented }
func (contractBackend) Eval(State, int, int) error        { return ErrUnimplemented }
func (contractBackend) Logits(State) []float32            { return nil }
func (contractBackend) EvalBatch(State, []int, int) error { return ErrUnimplemented }
func (contractBackend) Close() error                      { return nil }

func TestIndependentStateContractIsFailClosed(t *testing.T) {
	var b Backend = contractBackend{}
	state, err := b.NewState(128)
	if !errors.Is(err, ErrUnimplemented) {
		t.Fatalf("NewState error = %v, want ErrUnimplemented", err)
	}
	if state != nil {
		t.Fatal("NewState returned a state with ErrUnimplemented")
	}
}

// checkIndependentStates is the host self-oracle for the Backend contract.
// It compares both observable logits and the opaque state bytes, so an
// implementation cannot pass by keeping only the request token timeline
// independent while sharing the mutable model state.
func checkIndependentStates(b Backend) error {
	a, err := b.NewState(16)
	if err != nil {
		return err
	}
	defer a.Close()
	bState, err := b.NewState(16)
	if err != nil {
		return err
	}
	defer bState.Close()
	if a == nil || bState == nil {
		return errors.New("backend returned nil state")
	}
	if err := b.Eval(a, 11, 0); err != nil {
		return err
	}
	if err := b.Eval(bState, 29, 0); err != nil {
		return err
	}
	if got, want := b.Logits(a), []float32{11}; !equalFloat32(got, want) {
		return errors.New("first state was changed by the second evaluation")
	}
	if got, want := b.Logits(bState), []float32{29}; !equalFloat32(got, want) {
		return errors.New("second state did not retain its evaluation")
	}
	var savedA, savedB bytes.Buffer
	if err := a.Save(&savedA); err != nil {
		return err
	}
	if err := bState.Save(&savedB); err != nil {
		return err
	}
	if bytes.Equal(savedA.Bytes(), savedB.Bytes()) {
		return errors.New("independent states have identical serialized state")
	}
	return nil
}

func TestIndependentStateOracle(t *testing.T) {
	if err := checkIndependentStates(sessionBackend{}); err != nil {
		t.Fatalf("independent state oracle: %v", err)
	}
}

func TestStatePersistenceContract(t *testing.T) {
	b := sessionBackend{}
	first, err := b.NewState(32)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := b.NewState(32)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()

	for position, token := range []int{11, 29} {
		if err := b.Eval(first, token, position); err != nil {
			t.Fatalf("seed state at %d: %v", position, err)
		}
	}
	snapshot, err := first.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()

	if err := b.Eval(first, 43, 2); err != nil {
		t.Fatal(err)
	}
	if err := first.Restore(snapshot); err != nil {
		t.Fatalf("restore snapshot: %v", err)
	}
	if got, want := b.Logits(first), []float32{29}; !equalFloat32(got, want) {
		t.Fatalf("restored logits = %v, want %v", got, want)
	}

	var saved bytes.Buffer
	if err := first.Save(&saved); err != nil {
		t.Fatalf("save state: %v", err)
	}
	if err := second.Load(bytes.NewReader(saved.Bytes())); err != nil {
		t.Fatalf("load state: %v", err)
	}
	var reencoded bytes.Buffer
	if err := second.Save(&reencoded); err != nil {
		t.Fatalf("save loaded state: %v", err)
	}
	if !bytes.Equal(saved.Bytes(), reencoded.Bytes()) {
		t.Fatal("save-load-save changed state bytes")
	}

	beforeFailure := append([]byte(nil), reencoded.Bytes()...)
	if err := second.Load(bytes.NewReader(beforeFailure[:len(beforeFailure)-1])); err == nil {
		t.Fatal("accepted truncated state")
	}
	var afterFailure bytes.Buffer
	if err := second.Save(&afterFailure); err != nil {
		t.Fatalf("save state after failed load: %v", err)
	}
	if !bytes.Equal(beforeFailure, afterFailure.Bytes()) {
		t.Fatal("failed load changed destination state")
	}
}

type aliasedBackend struct{ state *sessionState }

func (b *aliasedBackend) NewState(int) (State, error) {
	if b.state == nil {
		b.state = &sessionState{}
	}
	return b.state, nil
}
func (b *aliasedBackend) Eval(state State, token, position int) error {
	return sessionBackend{}.Eval(state, token, position)
}
func (b *aliasedBackend) Logits(state State) []float32 { return sessionBackend{}.Logits(state) }
func (b *aliasedBackend) Close() error                 { return nil }

func TestIndependentStateOracleDetectsAliasedBackend(t *testing.T) {
	if err := checkIndependentStates(&aliasedBackend{}); err == nil {
		t.Fatal("independence oracle accepted an aliased backend")
	}
}

func TestBatchEvaluatorIsObservationallySerial(t *testing.T) {
	for _, bad := range []bool{false, true} {
		b := batchContractBackend{bad: bad}
		serial, err := b.NewState(32)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := b.NewState(32)
		if err != nil {
			t.Fatal(err)
		}
		tokens := []int{11, 29, 43, 71}
		for i, token := range tokens {
			if err := b.Eval(serial, token, i); err != nil {
				t.Fatal(err)
			}
		}
		if err := b.EvalBatch(batch, tokens, 0); err != nil {
			t.Fatal(err)
		}
		var serialBytes, batchBytes bytes.Buffer
		if err := serial.Save(&serialBytes); err != nil {
			t.Fatal(err)
		}
		if err := batch.Save(&batchBytes); err != nil {
			t.Fatal(err)
		}
		equivalent := bytes.Equal(serialBytes.Bytes(), batchBytes.Bytes()) &&
			equalFloat32(b.Logits(serial), b.Logits(batch))
		if bad == equivalent {
			t.Fatalf("batch equivalence = %v, bad backend = %v", equivalent, bad)
		}
	}
}

type batchContractBackend struct{ bad bool }

func (b batchContractBackend) NewState(context int) (State, error) {
	return sessionBackend{}.NewState(context)
}

func (b batchContractBackend) Eval(state State, token, position int) error {
	return sessionBackend{}.Eval(state, token, position)
}

func (b batchContractBackend) EvalBatch(state State, tokens []int, position int) error {
	for i, token := range tokens {
		if b.bad {
			token++
		}
		if err := b.Eval(state, token, position+i); err != nil {
			return err
		}
	}
	return nil
}

func (b batchContractBackend) Logits(state State) []float32 { return sessionBackend{}.Logits(state) }
func (b batchContractBackend) Close() error                 { return nil }

func TestContractImplementationsCompile(t *testing.T) {
	var _ Backend = contractBackend{}
	var _ BatchEvaluator = contractBackend{}
	var _ IndependentStates = contractBackend{}
	var _ State = contractState{}
	var _ Snapshot = contractSnapshot{}
}

type contractState struct{}

func (contractState) Reset() error                { return nil }
func (contractState) Snapshot() (Snapshot, error) { return contractSnapshot{}, nil }
func (contractState) Restore(Snapshot) error      { return nil }
func (contractState) Save(io.Writer) error        { return nil }
func (contractState) Load(io.Reader) error        { return nil }
func (contractState) Close() error                { return nil }

type contractSnapshot struct{}

func (contractSnapshot) Close() error { return nil }

func equalFloat32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
