package cudapdl_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

type pdlFixtureOwner struct {
	name  string
	close func() error
}

// pdlFixtureCustody is used by the native fixture, including failed launches.
// An unresolved cleanup keeps its remaining owner callbacks and never retries
// implicitly or reports those owners released.
type pdlFixtureCustody struct {
	owners     []pdlFixtureOwner
	pending    bool
	unresolved error
}

func (c *pdlFixtureCustody) own(name string, close func() error) {
	c.owners = append(c.owners, pdlFixtureOwner{name, close})
}

func (c *pdlFixtureCustody) ownError(name string, err error) {
	var owner io.Closer
	if errors.As(err, &owner) {
		c.own(name, owner.Close)
	}
}

func (c *pdlFixtureCustody) launch(launch func() error) error {
	if c.unresolved != nil {
		return c.unresolved
	}
	c.pending = true // An error may follow partial submission.
	return launch()
}

func (c *pdlFixtureCustody) observe(wait func() error) error {
	if c.unresolved != nil {
		return c.unresolved
	}
	err := wait()
	if err == nil {
		c.pending = false
	}
	return err
}

func (c *pdlFixtureCustody) release(mark int, drain func() error) error {
	if c.unresolved != nil {
		return c.unresolved
	}
	if c.pending {
		if err := drain(); err != nil {
			c.unresolved = fmt.Errorf("native drain unresolved: %w", err)
			return c.unresolved
		}
		c.pending = false
	}
	for len(c.owners) > mark {
		i := len(c.owners) - 1
		owner := c.owners[i]
		if err := owner.close(); err != nil {
			c.unresolved = fmt.Errorf("%s cleanup unresolved: %w", owner.name, err)
			return c.unresolved
		}
		c.owners = c.owners[:i]
	}
	return nil
}

// Keep actual Go owners and graph argument storage reachable until process
// exit after an unresolved native result. A later fixture invocation is refused.
type pdlFixtureRetainer struct {
	mu       sync.Mutex
	fixtures []*pdlFixtureCustody
}

func (r *pdlFixtureRetainer) keep(c *pdlFixtureCustody) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fixtures = append(r.fixtures, c)
}

func (r *pdlFixtureRetainer) unresolved() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.fixtures) > 0
}

var unresolvedPDLFixtures pdlFixtureRetainer

func TestPDLFixtureCustody(t *testing.T) {
	launchErr, drainErr, destroyErr := errors.New("launch"), errors.New("drain"), errors.New("destroy")
	for _, tt := range []struct {
		name                            string
		launch, observe, drain, destroy error
		want                            []string
		pending                         bool
		remaining                       int
		unresolved                      error
	}{
		{name: "success", want: []string{"exec", "graph", "buffer", "module", "stream", "device", "lock"}},
		{name: "launch failure", launch: launchErr, want: []string{"drain", "exec", "graph", "buffer", "module", "stream", "device", "lock"}},
		{name: "observer cancellation", observe: context.Canceled, want: []string{"drain", "exec", "graph", "buffer", "module", "stream", "device", "lock"}},
		{name: "drain failure", observe: context.Canceled, drain: drainErr, want: []string{"drain"}, pending: true, remaining: 7, unresolved: drainErr},
		{name: "destroy failure", destroy: destroyErr, want: []string{"exec"}, remaining: 7, unresolved: destroyErr},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			c := new(pdlFixtureCustody)
			for _, name := range []string{"lock", "device", "stream", "module", "buffer", "graph", "exec"} {
				c.own(name, func() error {
					calls = append(calls, name)
					if name == "exec" {
						return tt.destroy
					}
					return nil
				})
			}
			if err := c.launch(func() error { return tt.launch }); !errors.Is(err, tt.launch) {
				t.Fatal(err)
			}
			if tt.launch == nil {
				if err := c.observe(func() error { return tt.observe }); !errors.Is(err, tt.observe) {
					t.Fatal(err)
				}
			}
			err := c.release(0, func() error { calls = append(calls, "drain"); return tt.drain })
			if !errors.Is(err, tt.unresolved) || c.pending != tt.pending || len(c.owners) != tt.remaining || !slices.Equal(calls, tt.want) {
				t.Fatalf("cleanup=%v pending=%v owners=%d calls=%v", err, c.pending, len(c.owners), calls)
			}
			if tt.unresolved != nil {
				if err := c.launch(func() error { t.Fatal("submitted after unresolved cleanup"); return nil }); !errors.Is(err, tt.unresolved) {
					t.Fatal(err)
				}
				if err := c.observe(func() error { t.Fatal("observed after unresolved cleanup"); return nil }); !errors.Is(err, tt.unresolved) {
					t.Fatal(err)
				}
				before := slices.Clone(calls)
				if err := c.release(0, func() error { t.Fatal("retried unresolved drain"); return nil }); !errors.Is(err, tt.unresolved) {
					t.Fatal(err)
				}
				if !slices.Equal(calls, before) {
					t.Fatal("retried unresolved owners")
				}
				var retained pdlFixtureRetainer
				retained.keep(c)
				if !retained.unresolved() || retained.fixtures[0] != c {
					t.Fatal("lost unresolved ownership")
				}
			}
		})
	}
}

func TestPDLFixtureConstructorAndArmCustody(t *testing.T) {
	c := new(pdlFixtureCustody)
	var calls []string
	c.own("device", func() error { calls = append(calls, "device"); return nil })
	constructorErr := errors.New("stream constructor lost an inaccessible handle")
	c.unresolved = constructorErr
	if err := c.release(0, func() error { t.Fatal("unexpected drain"); return nil }); !errors.Is(err, constructorErr) || len(calls) != 0 || len(c.owners) != 1 {
		t.Fatal("constructor failure released dependency")
	}
	c = new(pdlFixtureCustody)
	c.own("device", func() error { calls = append(calls, "device"); return nil })
	mark := len(c.owners)
	c.own("graph error-result owner", func() error { calls = append(calls, "graph"); return nil })
	c.own("exec error-result owner", func() error { calls = append(calls, "exec"); return nil })
	if err := c.release(mark, func() error { t.Fatal("unsubmitted arm drained"); return nil }); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(calls, []string{"exec", "graph"}) || len(c.owners) != 1 {
		t.Fatalf("arm cleanup=%v remaining=%d", calls, len(c.owners))
	}
	if err := c.release(0, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
}

func TestPDLFixtureNestedLockCustody(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	own, err := cudalock.Acquire(ctx, cudalock.WithDir(dir), cudalock.WithNoReclaim())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := own.Release(); err != nil {
			t.Error(err)
		}
	})
	nested, err := cudalock.Acquire(ctx, cudalock.WithDir(dir), cudalock.WithNoReclaim())
	if err != nil {
		t.Fatal(err)
	}
	if !nested.IsNested() {
		t.Fatal("second acquisition is not nested")
	}
	if err := nested.Release(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "owner")); err != nil {
		t.Fatalf("nested release removed fixture hold: %v", err)
	}
	c := new(pdlFixtureCustody)
	c.own("fixture lock", own.Release)
	if err := c.launch(func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	drainErr := errors.New("unresolved fake drain")
	if err := c.release(0, func() error { return drainErr }); !errors.Is(err, drainErr) {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "owner")); err != nil {
		t.Fatalf("unresolved cleanup released fixture hold: %v", err)
	}
	// This is a CPU-only private-directory lock with no submitted work.
	// The test's cleanup releases it after these custody assertions.
}

type pdlFixtureCleanupError struct {
	cause error
	close func() error
}

func (e *pdlFixtureCleanupError) Error() string { return e.cause.Error() }
func (e *pdlFixtureCleanupError) Unwrap() error { return e.cause }

func (e *pdlFixtureCleanupError) Close() error { return e.close() }

func TestPDLFixtureErrorOwner(t *testing.T) {
	cause := errors.New("constructor")
	closed := 0
	owner := &pdlFixtureCleanupError{cause, func() error { closed++; return nil }}
	c := new(pdlFixtureCustody)
	c.ownError("returned cleanup owner", fmt.Errorf("construct: %w", owner))
	if len(c.owners) != 1 {
		t.Fatal("lost returned error owner")
	}
	if err := c.release(0, func() error { t.Fatal("unsubmitted work drained"); return nil }); err != nil {
		t.Fatal(err)
	}
	if closed != 1 || len(c.owners) != 0 {
		t.Fatal("error owner not closed once")
	}
}

func TestPDLFixtureSuccessfulLockRelease(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	own, err := cudalock.Acquire(ctx, cudalock.WithDir(dir), cudalock.WithNoReclaim())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := own.Release(); err != nil {
			t.Error(err)
		}
	})
	nested, err := cudalock.Acquire(ctx, cudalock.WithDir(dir), cudalock.WithNoReclaim())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := nested.Release(); err != nil {
			t.Error(err)
		}
	})
	c := new(pdlFixtureCustody)
	releases := 0
	c.own("fixture lock", func() error { releases++; return own.Release() })
	if err := c.release(0, func() error { t.Fatal("unsubmitted work drained"); return nil }); err != nil {
		t.Fatal(err)
	}
	if releases != 1 {
		t.Fatal("fixture lock release count", releases)
	}
	if _, err := os.Stat(filepath.Join(dir, "owner")); err != nil {
		t.Fatal("fixture release removed nested hold", err)
	}
	if err := nested.Release(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("resolved lock remains", err)
	}
	if err := c.release(0, func() error { return nil }); err != nil || releases != 1 {
		t.Fatal("repeated fixture cleanup released twice", err)
	}
}
