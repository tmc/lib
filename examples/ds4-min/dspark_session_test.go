package main

import (
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type dsparkPrefixTestState struct {
	stateSessionBackendTestState
	tokens               []int
	restoreErr, closeErr error
	closes               int
}

type dsparkPrefixTestSnapshot struct {
	owner  *dsparkPrefixTestState
	tokens []int
}

func (s *dsparkPrefixTestState) Snapshot() (dsv4.Snapshot, error) {
	return &dsparkPrefixTestSnapshot{s, append([]int(nil), s.tokens...)}, nil
}

func (s *dsparkPrefixTestState) Restore(v dsv4.Snapshot) error {
	if s.restoreErr != nil {
		return s.restoreErr
	}
	p, ok := v.(*dsparkPrefixTestSnapshot)
	if !ok || p.owner != s {
		return errors.New("foreign snapshot")
	}
	s.tokens = append(s.tokens[:0], p.tokens...)
	return nil
}

func (s *dsparkPrefixTestSnapshot) Close() error {
	s.owner.closes++
	return s.owner.closeErr
}

func TestDSparkAcceptedPrefix(t *testing.T) {
	failure := errors.New("injected run failure")
	for _, tc := range []struct {
		name          string
		ids, want     []int
		runErr        error
		stopped, fail bool
	}{
		{"accepted", []int{3, 4, 5}, []int{1, 2, 3, 4, 5}, nil, false, false},
		{"stop-first", []int{9, 4, 5}, []int{1, 2}, nil, true, false},
		{"stop-middle", []int{3, 9, 5}, []int{1, 2, 3}, nil, true, false},
		{"failed-run", []int{3, 4}, []int{1, 2}, failure, false, true},
		{"empty", nil, []int{1, 2}, nil, false, true},
		{"overflow", []int{3, 4, 5, 6}, []int{1, 2}, nil, false, true},
		{"invalid-id", []int{3, -1}, []int{1, 2}, nil, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &dsparkPrefixTestState{tokens: []int{1, 2}}
			var positions []int
			ids, stopped, err := runDSparkAcceptedPrefix(s, func() ([]int, error) {
				s.tokens = append(s.tokens, tc.ids...)
				return tc.ids, tc.runErr
			}, func(id, position int) error {
				if position != len(s.tokens) {
					return errors.New("incorrect replay position")
				}
				positions = append(positions, position)
				s.tokens = append(s.tokens, id)
				return nil
			}, []int{1, 2}, 3, func(id int) bool { return id == 9 })
			if (err != nil) != tc.fail || stopped != tc.stopped || !reflect.DeepEqual(s.tokens, tc.want) || s.closes != 1 {
				t.Fatalf("ids=%v stop=%v err=%v state=%v closes=%d", ids, stopped, err, s.tokens, s.closes)
			}
			if tc.name == "stop-middle" && !reflect.DeepEqual(positions, []int{2}) {
				t.Fatalf("replay positions=%v", positions)
			}
		})
	}
}

func TestDSparkRollbackFailure(t *testing.T) {
	s := &dsparkPrefixTestState{restoreErr: io.ErrUnexpectedEOF}
	_, _, err := runDSparkAcceptedPrefix(s, func() ([]int, error) { return nil, io.ErrShortWrite }, func(int, int) error { t.Fatal("replay after failed restore"); return nil }, nil, 1, func(int) bool { return false })
	if !errors.Is(err, dsv4.ErrDesynchronized) || !errors.Is(err, io.ErrUnexpectedEOF) || !errors.Is(err, io.ErrShortWrite) || s.closes != 1 {
		t.Fatalf("err=%v closes=%d", err, s.closes)
	}
}

func TestDSparkSnapshotCleanupOwner(t *testing.T) {
	s := &dsparkPrefixTestState{closeErr: io.ErrUnexpectedEOF}
	_, _, err := runDSparkAcceptedPrefix(s, func() ([]int, error) { return []int{3}, nil }, func(int, int) error { return nil }, nil, 1, func(int) bool { return false })
	owner, ok := err.(io.Closer)
	if !ok || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("cleanup error loses snapshot owner: %v", err)
	}
	s.closeErr = nil
	if err := owner.Close(); err != nil || s.closes != 2 {
		t.Fatalf("retry err=%v closes=%d", err, s.closes)
	}
}

func TestDSparkReplayFailure(t *testing.T) {
	s := &dsparkPrefixTestState{tokens: []int{1, 2}}
	_, _, err := runDSparkAcceptedPrefix(s, func() ([]int, error) { s.tokens = append(s.tokens, 3, 9); return []int{3, 9}, nil }, func(id, position int) error { s.tokens = append(s.tokens, id); return io.ErrShortWrite }, []int{1, 2}, 2, func(id int) bool { return id == 9 })
	if !errors.Is(err, io.ErrShortWrite) || !reflect.DeepEqual(s.tokens, []int{1, 2}) || s.closes != 1 {
		t.Fatalf("err=%v state=%v closes=%d", err, s.tokens, s.closes)
	}
}

func TestDSparkSupportConfig(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*config, *dsparkSupportConfig)
		fail   bool
	}{
		{"valid", func(*config, *dsparkSupportConfig) {}, false},
		{"missing-hash", func(_ *config, s *dsparkSupportConfig) { s.sha256 = "" }, true},
		{"uppercase", func(_ *config, s *dsparkSupportConfig) { s.sha256 = strings.Repeat("A", 64) }, true},
		{"confidence-zero", func(_ *config, s *dsparkSupportConfig) { s.confidence = 0 }, false},
		{"confidence-negative", func(_ *config, s *dsparkSupportConfig) { s.confidence = -.1 }, true},
		{"sampling", func(c *config, _ *dsparkSupportConfig) { c.sampling.Temperature = 1 }, true},
		{"cuda", func(c *config, _ *dsparkSupportConfig) { c.backend = "cuda" }, false},
		{"cpu", func(c *config, _ *dsparkSupportConfig) { c.backend = "cpu" }, true},
		{"persistence", func(c *config, _ *dsparkSupportConfig) { c.kvLoad = "state" }, true},
		{"streaming", func(c *config, _ *dsparkSupportConfig) { c.streaming.enabled = true }, true},
		{"streaming-cold", func(c *config, _ *dsparkSupportConfig) { c.streaming.cold = true }, true},
		{"streaming-auto", func(c *config, _ *dsparkSupportConfig) { c.streaming.autoCache = true }, true},
		{"streaming-budget", func(c *config, _ *dsparkSupportConfig) { c.streaming.budget = "2" }, true},
		{"streaming-preload", func(c *config, _ *dsparkSupportConfig) { c.streaming.preload = 1 }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config{backend: "metal"}
			support := dsparkSupportConfig{path: "support.gguf", sha256: strings.Repeat("a", 64), confidence: .5}
			tc.change(&cfg, &support)
			if err := support.validate(cfg); (err != nil) != tc.fail {
				t.Fatalf("validation=%v", err)
			}
		})
	}
}
func TestDSparkGenerationLimit(t *testing.T) {
	for _, tc := range []struct {
		name                          string
		context, history, limit, want int
		fail                          bool
	}{
		{"ordinary", 8, 2, 3, 3, false},
		{"short-context", 8, 7, 16, 1, false},
		{"full", 8, 8, 16, 0, true},
		{"zero", 8, 2, 0, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dsparkGenerationLimit(tc.context, tc.history, tc.limit)
			if got != tc.want || (err != nil) != tc.fail {
				t.Fatalf("limit=%d err=%v", got, err)
			}
		})
	}
}

func TestDSparkCLIUnexposed(t *testing.T) {
	_, err := parseConfig([]string{"-dspark", "support.gguf"}, strings.NewReader(""), io.Discard)
	if err == nil || !strings.Contains(err.Error(), "flag provided but not defined") {
		t.Fatalf("unexpected CLI exposure: %v", err)
	}
}

type dsparkCloseTestOwner struct {
	name     string
	events   *[]string
	failures int
}

func (o *dsparkCloseTestOwner) Close() error {
	*o.events = append(*o.events, o.name)
	if o.failures > 0 {
		o.failures--
		return io.ErrUnexpectedEOF
	}
	return nil
}

func TestDSparkOpenCleanup(t *testing.T) {
	for _, name := range []string{"success", "construction-error", "resources-retained", "file-retained", "runtime-retained"} {
		t.Run(name, func(t *testing.T) {
			var events []string
			resources := &dsparkCloseTestOwner{name: "resources", events: &events}
			file := &dsparkCloseTestOwner{name: "file", events: &events}
			cause := errors.New("construction failed")
			var initial error
			var want []string
			switch name {
			case "success":
				want = []string{"file"}
			case "construction-error":
				initial, want = cause, []string{"resources", "file"}
			case "resources-retained":
				initial, want, resources.failures = cause, []string{"resources"}, 1
			case "file-retained":
				want, file.failures = []string{"file", "resources"}, 1
			case "runtime-retained":
				initial = &backendCleanupOwner{cause: cause, backend: &dsparkCloseTestOwner{name: "runtime", events: &events, failures: 1}}
			}
			err := finishDSparkOpen(initial, resources, file)
			if !reflect.DeepEqual(events, want) || (err == nil) != (name == "success") {
				t.Fatalf("err=%v events=%v want=%v", err, events, want)
			}
			if name == "construction-error" && !errors.Is(err, cause) {
				t.Fatal("lost construction error")
			}
			if name == "success" || name == "construction-error" {
				return
			}
			owner, ok := err.(io.Closer)
			if !ok {
				t.Fatalf("lost cleanup owner: %v", err)
			}
			if name == "runtime-retained" {
				if err := owner.Close(); err == nil || !reflect.DeepEqual(events, []string{"runtime"}) {
					t.Fatalf("premature dependency cleanup: err=%v events=%v", err, events)
				}
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			switch name {
			case "resources-retained":
				want = []string{"resources", "resources", "file"}
			case "file-retained":
				want = []string{"file", "resources", "file"}
			case "runtime-retained":
				want = []string{"runtime", "runtime", "resources", "file"}
			}
			if !reflect.DeepEqual(events, want) {
				t.Fatalf("cleanup=%v want=%v", events, want)
			}
		})
	}
}
