package main

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

type cleanupOwners struct {
	program, free, module, device, pending bool
}

func ownedCleanup(c *probeCleanup) cleanupOwners {
	return cleanupOwners{c.program != nil, c.free != nil, c.module != nil, c.device != nil, c.pending}
}

func TestProbeCleanup(t *testing.T) {
	primary := errors.New("work failed")
	drainErr := errors.New("drain failed")
	freeErr := errors.New("free failed")
	moduleErr := errors.New("unload failed")
	deviceErr := errors.New("device release failed")
	contextErr := errors.New("context failed")
	programErr := errors.New("program destroy failed")
	all := cleanupOwners{free: true, module: true, device: true}
	pending := all
	pending.pending = true
	tests := []struct {
		name                        string
		owned                       cleanupOwners
		primary                     error
		failures                    map[string]error
		contextBefore, contextAfter error
		wantCalls                   []string
		wantErrors                  []error
		wantOwned                   cleanupOwners
	}{
		{
			name: "no resources", wantCalls: []string{"work"},
		},
		{
			name: "device only", owned: cleanupOwners{device: true},
			primary: primary, wantCalls: []string{"work", "device"}, wantErrors: []error{primary},
		},
		{
			name: "module only", owned: cleanupOwners{module: true, device: true},
			wantCalls: []string{"work", "context", "module", "device"},
		},
		{
			name: "quiescent", owned: all,
			wantCalls: []string{"work", "context", "free", "module", "device"},
		},
		{
			name: "drain before release", owned: pending,
			wantCalls: []string{"work", "context", "drain", "free", "module", "device"},
		},
		{
			name: "unknown drain retains dependencies", owned: pending,
			failures:  map[string]error{"drain": drainErr},
			wantCalls: []string{"work", "context", "drain"}, wantErrors: []error{drainErr}, wantOwned: pending,
		},
		{
			name: "primary and drain errors", owned: pending, primary: primary,
			failures:  map[string]error{"drain": drainErr},
			wantCalls: []string{"work", "context", "drain"}, wantErrors: []error{primary, drainErr}, wantOwned: pending,
		},
		{
			name: "free error", owned: all, failures: map[string]error{"free": freeErr},
			wantCalls: []string{"work", "context", "free", "module"}, wantErrors: []error{freeErr},
			wantOwned: cleanupOwners{free: true, device: true},
		},
		{
			name: "module error", owned: all, failures: map[string]error{"module": moduleErr},
			wantCalls: []string{"work", "context", "free", "module"}, wantErrors: []error{moduleErr},
			wantOwned: cleanupOwners{module: true, device: true},
		},
		{
			name: "primary and independent cleanup errors", owned: all, primary: primary,
			failures:  map[string]error{"free": freeErr, "module": moduleErr},
			wantCalls: []string{"work", "context", "free", "module"}, wantErrors: []error{primary, freeErr, moduleErr}, wantOwned: all,
		},
		{
			name: "device error", owned: all, failures: map[string]error{"device": deviceErr},
			wantCalls: []string{"work", "context", "free", "module", "device"}, wantErrors: []error{deviceErr},
			wantOwned: cleanupOwners{device: true},
		},
		{
			name: "context entry failure retains dependencies", owned: pending, contextBefore: contextErr,
			wantCalls: []string{"work", "context"}, wantErrors: []error{contextErr}, wantOwned: pending,
		},
		{
			name: "context restoration error preserves completed releases", owned: all, contextAfter: contextErr,
			wantCalls: []string{"work", "context", "free", "module", "device"}, wantErrors: []error{contextErr},
		},
		{
			name: "context restoration and device errors", owned: all, contextAfter: contextErr,
			failures:  map[string]error{"device": deviceErr},
			wantCalls: []string{"work", "context", "free", "module", "device"}, wantErrors: []error{contextErr, deviceErr},
			wantOwned: cleanupOwners{device: true},
		},
		{
			name: "program retry", owned: cleanupOwners{program: true, device: true}, primary: primary,
			wantCalls: []string{"work", "program", "device"}, wantErrors: []error{primary},
		},
		{
			name: "program error has no context dependency", owned: cleanupOwners{program: true, device: true}, primary: primary,
			failures:  map[string]error{"program": programErr},
			wantCalls: []string{"work", "program", "device"}, wantErrors: []error{primary, programErr},
			wantOwned: cleanupOwners{program: true},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			callback := func(name string) func() error {
				return func() error {
					calls = append(calls, name)
					return tt.failures[name]
				}
			}
			c := probeCleanup{
				withContext: func(fn func() error) error {
					calls = append(calls, "context")
					if tt.contextBefore != nil {
						return tt.contextBefore
					}
					return errors.Join(fn(), tt.contextAfter)
				},
				drain: callback("drain"), pending: tt.owned.pending,
			}
			if tt.owned.program {
				c.program = callback("program")
			}
			if tt.owned.free {
				c.free = callback("free")
			}
			if tt.owned.module {
				c.module = callback("module")
			}
			if tt.owned.device {
				c.device = callback("device")
			}
			err := c.run(func() error {
				calls = append(calls, "work")
				return tt.primary
			})
			if !reflect.DeepEqual(calls, tt.wantCalls) {
				t.Errorf("calls = %v, want %v", calls, tt.wantCalls)
			}
			if len(tt.wantErrors) == 0 && err != nil {
				t.Errorf("run: %v", err)
			}
			for _, want := range tt.wantErrors {
				if !errors.Is(err, want) {
					t.Errorf("error = %v, missing %v", err, want)
				}
			}
			if got := ownedCleanup(&c); got != tt.wantOwned {
				t.Errorf("retained owners = %+v, want %+v", got, tt.wantOwned)
			}
		})
	}
}

type retryError struct {
	err   error
	close func() error
}

func (e *retryError) Error() string { return e.err.Error() }
func (e *retryError) Unwrap() error { return e.err }
func (e *retryError) Close() error  { return e.close() }

func TestProbeCleanupRetry(t *testing.T) {
	primary := errors.New("operation failed")
	initial := errors.New("initial cleanup failed")
	failedRetry := errors.New("retry failed")
	for _, object := range []string{"program", "module"} {
		for _, retryErr := range []error{nil, failedRetry} {
			name := object + "/success"
			if retryErr != nil {
				name = object + "/failure"
			}
			t.Run(name, func(t *testing.T) {
				var retries, closes int
				var current bool
				owned := &retryError{
					err: errors.Join(primary, initial),
					close: func() error {
						retries++
						if current != (object == "module") {
							t.Errorf("retry current context = %v for %s", current, object)
						}
						return retryErr
					},
				}
				wrapped := fmt.Errorf("operation: %w", owned)
				c := probeCleanup{
					withContext: func(fn func() error) error {
						current = true
						defer func() { current = false }()
						return fn()
					},
					device: func() error { closes++; return nil },
				}
				err := c.run(func() error {
					if object == "program" {
						c.program = retryCleanup(wrapped)
					} else {
						c.module = retryCleanup(wrapped)
					}
					return wrapped
				})
				if retries != 1 {
					t.Errorf("retry calls = %d, want 1", retries)
				}
				for _, want := range []error{primary, initial, retryErr} {
					if want != nil && !errors.Is(err, want) {
						t.Errorf("error = %v, missing %v", err, want)
					}
				}
				wantOwned := cleanupOwners{}
				wantCloses := 1
				if retryErr != nil {
					if object == "program" {
						wantOwned.program = true
					} else {
						wantOwned.module, wantOwned.device = true, true
						wantCloses = 0
					}
				}
				if closes != wantCloses {
					t.Errorf("device closes = %d, want %d", closes, wantCloses)
				}
				if got := ownedCleanup(&c); got != wantOwned {
					t.Errorf("retained owners = %+v, want %+v", got, wantOwned)
				}
			})
		}
	}
	if got := retryCleanup(errors.New("no retained object")); got != nil {
		t.Error("plain error has a cleanup callback")
	}
}

func TestProbeDrainRetry(t *testing.T) {
	launchErr := errors.New("launch failed")
	firstErr := errors.New("first drain failed")
	retryErr := errors.New("cleanup drain failed")
	tests := []struct {
		name          string
		launch, first error
		cleanup       error
		wantDrains    int
	}{
		{"successful drain", nil, nil, nil, 1},
		{"failed launch drained", launchErr, nil, nil, 1},
		{"failed launch unknown", launchErr, nil, retryErr, 1},
		{"failed drain recovered", nil, firstErr, nil, 2},
		{"failed drain retained", nil, firstErr, retryErr, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var drains int
			var released []string
			c := probeCleanup{
				withContext: func(fn func() error) error { return fn() },
				drain: func() error {
					drains++
					if tt.launch == nil && drains == 1 {
						return tt.first
					}
					return tt.cleanup
				},
				free:   func() error { released = append(released, "free"); return nil },
				module: func() error { released = append(released, "module"); return nil },
				device: func() error { released = append(released, "device"); return nil },
			}
			err := c.run(func() error {
				c.pending = true
				if tt.launch != nil {
					return tt.launch
				}
				if err := c.drain(); err != nil {
					return err
				}
				c.pending = false
				return nil
			})
			if drains != tt.wantDrains {
				t.Errorf("drain calls = %d, want %d", drains, tt.wantDrains)
			}
			for _, want := range []error{tt.launch, tt.first, tt.cleanup} {
				if want != nil && !errors.Is(err, want) {
					t.Errorf("error = %v, missing %v", err, want)
				}
			}
			if tt.launch == nil && tt.first == nil && err != nil {
				t.Errorf("run: %v", err)
			}
			if tt.cleanup != nil {
				if len(released) != 0 {
					t.Errorf("released resources after unknown drain: %v", released)
				}
				if got := ownedCleanup(&c); got != (cleanupOwners{free: true, module: true, device: true, pending: true}) {
					t.Errorf("lost retained owners: %+v", got)
				}
			} else if !reflect.DeepEqual(released, []string{"free", "module", "device"}) {
				t.Errorf("releases = %v", released)
			} else if got := ownedCleanup(&c); got != (cleanupOwners{}) {
				t.Errorf("owners retained after successful cleanup: %+v", got)
			}
		})
	}
}
