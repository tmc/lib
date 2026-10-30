package cuda

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

type constructorCase struct {
	name       string
	bindErr    error
	createErr  error
	restoreErr error
}

func constructorCases() []constructorCase {
	bind := errors.New("bind failed")
	create := errors.New("create failed")
	restore := errors.New("restore failed")
	return []constructorCase{
		{name: "success"},
		{name: "bind failure", bindErr: bind},
		{name: "creation failure", createErr: create},
		{name: "restore failure", restoreErr: restore},
		{name: "creation and restore failure", createErr: create, restoreErr: restore},
	}
}

func (c constructorCase) run(fn func() error) error {
	if c.bindErr != nil {
		return c.bindErr
	}
	return errors.Join(fn(), c.restoreErr)
}

func checkConstructorError(t *testing.T, got error, c constructorCase) {
	t.Helper()
	if (got == nil) != (c.bindErr == nil && c.createErr == nil && c.restoreErr == nil) {
		t.Fatalf("constructor error = %v", got)
	}
	for _, want := range []error{c.bindErr, c.createErr, c.restoreErr} {
		if want != nil && !errors.Is(got, want) {
			t.Errorf("constructor error %v does not retain %v", got, want)
		}
	}
}

func TestStreamConstructorPreservesCustody(t *testing.T) {
	for _, c := range constructorCases() {
		t.Run(c.name, func(t *testing.T) {
			d := &Device{}
			called := false
			const raw = driver.Stream(123)
			s, err := d.newStreamWith(driver.StreamNonBlocking, c.run, func(flags driver.StreamFlags) (driver.Stream, error) {
				called = true
				if flags != driver.StreamNonBlocking {
					t.Errorf("flags = %v, want nonblocking", flags)
				}
				if c.createErr != nil {
					return 0, c.createErr
				}
				return raw, nil
			})
			checkConstructorError(t, err, c)
			if called != (c.bindErr == nil) {
				t.Errorf("creation called = %v", called)
			}
			wantOwned := c.bindErr == nil && c.createErr == nil
			if (s != nil) != wantOwned {
				t.Fatalf("stream = %v, want ownership %v", s, wantOwned)
			}
			if s != nil && (s.raw != raw || s.device != d) {
				t.Error("returned stream lost its handle or device")
			}
		})
	}
}

func TestEventConstructorPreservesCustody(t *testing.T) {
	for _, c := range constructorCases() {
		t.Run(c.name, func(t *testing.T) {
			d := &Device{}
			called := false
			const raw = driver.Event(456)
			e, err := d.newEventWith(driver.EventDisableTiming, c.run, func(flags driver.EventFlags) (driver.Event, error) {
				called = true
				if flags != driver.EventDisableTiming {
					t.Errorf("flags = %v, want timing disabled", flags)
				}
				if c.createErr != nil {
					return 0, c.createErr
				}
				return raw, nil
			})
			checkConstructorError(t, err, c)
			if called != (c.bindErr == nil) {
				t.Errorf("creation called = %v", called)
			}
			wantOwned := c.bindErr == nil && c.createErr == nil
			if (e != nil) != wantOwned {
				t.Fatalf("event = %v, want ownership %v", e, wantOwned)
			}
			if e != nil && (e.raw != raw || e.device != d) {
				t.Error("returned event lost its handle or device")
			}
		})
	}
}
