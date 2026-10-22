package dsv4chat

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func ExampleConfig() {
	cfg := fixture(new(backend))
	cfg.ConfigureState = func(dsv4.State) error { fmt.Println("configure"); return nil }
	cfg.StateChanged = func(dsv4.State) { fmt.Println("publish") }
	var output bytes.Buffer
	_ = Run(context.Background(), strings.NewReader("hello\n/quit"), &output, cfg)
	fmt.Println(output.String())
	// Output:
	// configure
	// publish
	// ok
}

type replayFailureBackend struct {
	resizeBackend
	failure error
}

func (b *replayFailureBackend) Eval(s dsv4.State, token, position int) error {
	if len(b.states) > 1 && s == b.states[1] {
		return b.failure
	}
	return b.resizeBackend.Eval(s, token, position)
}

func TestConfiguredReplayFailureDoesNotPublish(t *testing.T) {
	failure := errors.New("replay failed")
	cleanup := errors.New("replacement cleanup failed")
	b := &replayFailureBackend{resizeBackend: resizeBackend{closeErrors: []error{nil, cleanup}}, failure: failure}
	cfg := fixture(new(backend))
	cfg.Backend = b
	configured, published := 0, 0
	cfg.ConfigureState = func(dsv4.State) error { configured++; return nil }
	cfg.StateChanged = func(dsv4.State) { published++ }
	cfg.Command = func(string) (Action, error) { return Action{Context: 16, Handled: true}, nil }
	var diagnostic bytes.Buffer
	cfg.Errors = &diagnostic
	err := Run(context.Background(), strings.NewReader("one\n/ctx 16\ntwo"), new(bytes.Buffer), cfg)
	if !errors.Is(err, failure) || !errors.Is(err, cleanup) {
		t.Fatalf("cleanup error %v", err)
	}
	if configured != 2 || published != 1 || len(b.states[0].values) != 8 {
		t.Fatalf("configure %d publish %d oldtimeline %v", configured, published, b.states[0].values)
	}
	if !strings.Contains(diagnostic.String(), failure.Error()) || !strings.Contains(diagnostic.String(), cleanup.Error()) {
		t.Fatalf("diagnostic %q", diagnostic.String())
	}
	for _, s := range b.states {
		if s.closes != 1 {
			t.Fatalf("closed %d times", s.closes)
		}
	}
}

func TestStateConfigurationPublication(t *testing.T) {
	b := new(resizeBackend)
	cfg := fixture(new(backend))
	cfg.Backend = b
	var configured, published []dsv4.State
	cfg.ConfigureState = func(s dsv4.State) error {
		if len(s.(*resizeState).values) != 0 {
			t.Fatal("configured after replay")
		}
		configured = append(configured, s)
		return nil
	}
	cfg.StateChanged = func(s dsv4.State) {
		if len(published) > 0 && published[len(published)-1].(*resizeState).closes != 1 {
			t.Fatal("published before old close")
		}
		published = append(published, s)
	}
	cfg.Command = func(line string) (Action, error) {
		size := 16
		if line == "/ctx 2" {
			size = 2
		}
		return Action{Context: size, Handled: true}, nil
	}
	if err := Run(context.Background(), strings.NewReader("one\n/ctx 16\n/ctx 2\ntwo"), new(bytes.Buffer), cfg); err != nil {
		t.Fatal(err)
	}
	if len(configured) != 3 || len(published) != 2 || published[0] != configured[0] || published[1] != configured[1] {
		t.Fatalf("configured %d published %d", len(configured), len(published))
	}
}

func TestStateConfigurationFailureCleanup(t *testing.T) {
	failure := errors.New("configure failed")
	cleanup := errors.New("close failed")
	for _, initial := range []bool{true, false} {
		t.Run(map[bool]string{true: "initial", false: "replacement"}[initial], func(t *testing.T) {
			b := &resizeBackend{}
			if initial {
				b.closeErrors = []error{cleanup}
			} else {
				b.closeErrors = []error{nil, cleanup}
			}
			cfg := fixture(new(backend))
			cfg.Backend = b
			configured := 0
			published := 0
			cfg.ConfigureState = func(dsv4.State) error {
				configured++
				if initial || configured == 2 {
					return failure
				}
				return nil
			}
			cfg.StateChanged = func(dsv4.State) { published++ }
			cfg.Command = func(string) (Action, error) { return Action{Context: 16, Handled: true}, nil }
			var diagnostic bytes.Buffer
			cfg.Errors = &diagnostic
			err := Run(context.Background(), strings.NewReader("one\n/ctx 16\ntwo"), new(bytes.Buffer), cfg)
			if initial {
				if !errors.Is(err, failure) || !errors.Is(err, cleanup) || published != 0 {
					t.Fatalf("initial error %v publications %d", err, published)
				}
			} else {
				if !errors.Is(err, cleanup) || !errors.Is(err, failure) || published != 1 || !strings.Contains(diagnostic.String(), failure.Error()) || !strings.Contains(diagnostic.String(), cleanup.Error()) {
					t.Fatalf("replacement error %v publications %d diagnostic %q", err, published, diagnostic.String())
				}
				if len(b.states[0].values) != 8 {
					t.Fatalf("old session unusable: %v", b.states[0].values)
				}
			}
			for _, s := range b.states {
				if s.closes != 1 {
					t.Fatalf("closed %d times", s.closes)
				}
			}
		})
	}
}
