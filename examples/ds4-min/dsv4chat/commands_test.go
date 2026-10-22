package dsv4chat

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type resizeState struct {
	state
	values   []int
	closeErr error
}

func (s *resizeState) Close() error { return errors.Join(s.state.Close(), s.closeErr) }

type resizeBackend struct {
	states      []*resizeState
	closeErrors []error
}

func (b *resizeBackend) NewState(int) (dsv4.State, error) {
	s := new(resizeState)
	if len(b.states) < len(b.closeErrors) {
		s.closeErr = b.closeErrors[len(b.states)]
	}
	b.states = append(b.states, s)
	return s, nil
}

func TestContextReplacementCloseFailure(t *testing.T) {
	oldFailure := errors.New("old state close failed")
	cleanupFailure := errors.New("replacement close failed")
	for _, both := range []bool{false, true} {
		t.Run(fmt.Sprint(both), func(t *testing.T) {
			b := &resizeBackend{closeErrors: []error{oldFailure, nil}}
			if both {
				b.closeErrors[1] = cleanupFailure
			}
			cfg := fixture(new(backend))
			cfg.Backend = b
			cfg.Command = func(string) (Action, error) { return Action{Context: 16, Handled: true}, nil }
			cfg.ContextChanged = func(int) { t.Fatal("published replacement after close failure") }
			err := Run(context.Background(), strings.NewReader("/ctx 16\n"), new(bytes.Buffer), cfg)
			if !errors.Is(err, oldFailure) || both && !errors.Is(err, cleanupFailure) {
				t.Fatalf("error %v", err)
			}
			for i, s := range b.states {
				want := 1
				if i == 0 {
					want = 2
				}
				if s.closes != want {
					t.Fatalf("state closes %d", s.closes)
				}
			}
		})
	}
}
func (*resizeBackend) Eval(s dsv4.State, token, position int) error {
	v := s.(*resizeState)
	if position != len(v.values) {
		return fmt.Errorf("position %d, length %d", position, len(v.values))
	}
	v.values = append(v.values, token)
	return nil
}
func (*resizeBackend) Logits(dsv4.State) []float32 { return []float32{1} }
func (*resizeBackend) Close() error                { return nil }

func TestContextReplacementPreservesUsableState(t *testing.T) {
	b := new(resizeBackend)
	cfg := fixture(new(backend))
	cfg.Backend = b
	var capacities []int
	cfg.ContextChanged = func(n int) { capacities = append(capacities, n) }
	cfg.Command = func(line string) (Action, error) {
		if line == "/ctx 16" {
			return Action{Context: 16, Handled: true}, nil
		}
		if line == "/ctx 2" {
			return Action{Context: 2, Handled: true}, nil
		}
		return Action{}, nil
	}
	var errors bytes.Buffer
	cfg.Errors = &errors
	if err := Run(context.Background(), strings.NewReader("one\n/ctx 16\n/ctx 2\ntwo\n/quit"), new(bytes.Buffer), cfg); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(capacities, []int{16}) {
		t.Fatalf("capacities %v", capacities)
	}
	if len(b.states) != 3 {
		t.Fatalf("states %d", len(b.states))
	}
	if !reflect.DeepEqual(b.states[0].values, []int{1, 2, 3, 7}) || !reflect.DeepEqual(b.states[1].values, []int{1, 2, 3, 7, 99, 2, 3, 7}) {
		t.Fatalf("timelines %v %v", b.states[0].values, b.states[1].values)
	}
	for _, s := range b.states {
		if s.closes != 1 {
			t.Fatalf("state closes %d", s.closes)
		}
	}
	if !strings.Contains(errors.String(), "prompt exceeds context") {
		t.Fatalf("diagnostic %q", errors.String())
	}
}

func ExampleAction() {
	action := Action{Context: 4096, Handled: true}
	fmt.Println(action.Context, action.Handled)
	// Output: 4096 true
}
