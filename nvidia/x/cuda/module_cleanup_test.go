package cuda

import (
	"errors"
	"reflect"
	"testing"
)

func TestModuleCloseRetry(t *testing.T) {
	failure := errors.New("injected failure")
	for _, tt := range []struct {
		name                    string
		first, retry            []string
		wantModule, wantContext bool
	}{
		{"enter", []string{"enter"}, []string{"enter", "sync", "unload", "restore", "release"}, true, true},
		{"sync", []string{"enter", "sync", "restore"}, []string{"enter", "sync", "unload", "restore", "release"}, true, true},
		{"unload", []string{"enter", "sync", "unload", "restore"}, []string{"enter", "sync", "unload", "restore", "release"}, true, true},
		{"restore", []string{"enter", "sync", "unload", "restore", "release"}, nil, false, false},
		{"release", []string{"enter", "sync", "unload", "restore", "release"}, []string{"release"}, false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := &Module{raw: 1, ctx: new(Context)}
			var calls []string
			fail := true
			call := func(name string) error {
				calls = append(calls, name)
				if fail && name == tt.name {
					return failure
				}
				return nil
			}
			run := func(fn func() error) error {
				if err := call("enter"); err != nil {
					return err
				}
				return errors.Join(fn(), call("restore"))
			}
			close := func() error {
				m.mu.Lock()
				defer m.mu.Unlock()
				return m.close(run, func() error { return call("sync") }, func() error { return call("unload") }, func() error { return call("release") })
			}
			if err := close(); !errors.Is(err, failure) {
				t.Fatalf("first close: %v", err)
			}
			if !reflect.DeepEqual(calls, tt.first) {
				t.Fatalf("first calls = %v, want %v", calls, tt.first)
			}
			if (m.raw != 0) != tt.wantModule || (m.ctx != nil) != tt.wantContext {
				t.Fatalf("remaining module=%v context=%v", m.raw != 0, m.ctx != nil)
			}
			fail = false
			calls = nil
			if err := close(); err != nil {
				t.Fatalf("retry: %v", err)
			}
			if !reflect.DeepEqual(calls, tt.retry) {
				t.Fatalf("retry calls = %v, want %v", calls, tt.retry)
			}
			if m.raw != 0 || m.ctx != nil {
				t.Fatal("retry retained released resources")
			}
			calls = nil
			if err := close(); err != nil {
				t.Fatalf("repeat: %v", err)
			}
			if len(calls) != 0 {
				t.Fatalf("repeat calls = %v", calls)
			}
		})
	}
}

func TestModuleClosePreservesCombinedErrors(t *testing.T) {
	unloadErr := errors.New("unload failed")
	restoreErr := errors.New("restore failed")
	releaseErr := errors.New("release failed")
	for _, failUnload := range []bool{false, true} {
		m := &Module{raw: 1, ctx: new(Context)}
		released := false
		err := m.close(func(fn func() error) error { return errors.Join(fn(), restoreErr) },
			func() error { return nil },
			func() error {
				if failUnload {
					return unloadErr
				}
				return nil
			},
			func() error { released = true; return releaseErr })
		if !errors.Is(err, restoreErr) || errors.Is(err, unloadErr) != failUnload || errors.Is(err, releaseErr) == failUnload {
			t.Fatalf("unload failure=%v: lost errors: %v", failUnload, err)
		}
		if released == failUnload || (m.raw != 0) != failUnload || m.ctx == nil {
			t.Fatalf("unload failure=%v: released=%v module=%v context=%v", failUnload, released, m.raw != 0, m.ctx != nil)
		}
	}
}
