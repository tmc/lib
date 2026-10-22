package dsv4chat

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type backend struct {
	state     *state
	evaluated []int
}
type state struct{ resets, closes int }

func (b *backend) NewState(int) (dsv4.State, error) { b.state = new(state); return b.state, nil }
func (b *backend) Eval(_ dsv4.State, token, _ int) error {
	b.evaluated = append(b.evaluated, token)
	return nil
}
func (*backend) Logits(dsv4.State) []float32    { return []float32{1} }
func (*backend) Close() error                   { return nil }
func (s *state) Reset() error                   { s.resets++; return nil }
func (s *state) Close() error                   { s.closes++; return nil }
func (*state) Snapshot() (dsv4.Snapshot, error) { return nil, dsv4.ErrUnimplemented }
func (*state) Restore(dsv4.Snapshot) error      { return dsv4.ErrUnimplemented }
func (*state) Save(io.Writer) error             { return dsv4.ErrUnimplemented }
func (*state) Load(io.Reader) error             { return dsv4.ErrUnimplemented }

func fixture(b *backend) Config {
	step := 0
	return Config{Backend: b, Context: 100, Stops: dsv4.NewStopSet(99), Tokens: 4,
		Render: func(messages []Message) ([]int, error) {
			ids := []int{1}
			for _, m := range messages {
				if m.Role == "user" {
					ids = append(ids, 2)
				} else {
					ids = append(ids, 3)
					ids = append(ids, m.Tokens...)
					ids = append(ids, 99)
				}
			}
			return append(ids, 3), nil
		},
		Sample: func([]float32) (int, error) {
			step++
			if step%2 == 0 {
				return 99, nil
			}
			return 7, nil
		},
		Decode: func(int) (string, error) { return "ok", nil },
	}
}

func TestRun(t *testing.T) {
	for _, tt := range []struct {
		name, input string
		want        []int
		resets      int
	}{
		{"reuse", "one\ntwo", []int{1, 2, 3, 7, 99, 2, 3, 7}, 0},
		{"reset", "one\n/reset\ntwo\n/quit\nignored", []int{1, 2, 3, 7, 1, 2, 3, 7}, 1},
		{"exit", "/exit\nignored", nil, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := new(backend)
			var out bytes.Buffer
			if err := Run(context.Background(), strings.NewReader(tt.input), &out, fixture(b)); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(b.evaluated, tt.want) {
				t.Fatalf("evaluated %v, want %v", b.evaluated, tt.want)
			}
			if b.state.resets != tt.resets || b.state.closes != 1 {
				t.Fatalf("state %+v", b.state)
			}
		})
	}
}

func TestFailureClosesState(t *testing.T) {
	for _, stage := range []string{"render", "sample", "decode", "cancel"} {
		t.Run(stage, func(t *testing.T) {
			b := new(backend)
			cfg := fixture(b)
			failure := errors.New("injected")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch stage {
			case "render":
				cfg.Render = func([]Message) ([]int, error) { return nil, failure }
			case "sample":
				cfg.Sample = func([]float32) (int, error) { return 0, failure }
			case "decode":
				cfg.Decode = func(int) (string, error) { return "", failure }
			case "cancel":
				cfg.Tokens = 1
				cfg.Sample = func([]float32) (int, error) { cancel(); return 7, nil }
				failure = context.Canceled
			}
			if err := Run(ctx, strings.NewReader("one\ntwo"), io.Discard, cfg); !errors.Is(err, failure) {
				t.Fatalf("error %v", err)
			}
			if b.state.closes != 1 {
				t.Fatalf("closes %d", b.state.closes)
			}
		})
	}
}

func ExampleRun() {
	b := new(backend)
	var output bytes.Buffer
	_ = Run(context.Background(), strings.NewReader("Hello\n/quit\n"), &output, fixture(b))
	fmt.Print(output.String())
	// Output: ok
}
