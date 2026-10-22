package cudasearch

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type abnormalCleanupExecutable struct {
	Executable
	calls      int
	goexit     bool
	panicValue any
}

func (x *abnormalCleanupExecutable) Close() error {
	x.calls++
	if x.goexit {
		runtime.Goexit()
	}
	panic(x.panicValue)
}

func TestObserverCleanupAbandonment(t *testing.T) {
	for _, goexit := range []bool{false, true} {
		for _, observerPanics := range []bool{false, true} {
			for _, compileFails := range []bool{false, true} {
				name := "panic"
				if goexit {
					name = "goexit"
				}
				if observerPanics {
					name += "/observer-panic"
				}
				if compileFails {
					name += "/compile-error-owner"
				}
				t.Run(name, func(t *testing.T) {
					e, _, compiled, order := executableFixture(t)
					ctx, b := executableMemoryContext(1 << 20)
					original := errors.New("original cleanup panic")
					x := &abnormalCleanupExecutable{Executable: compiled, goexit: goexit, panicValue: original}
					compileErr := errors.New("compile returned failed owner")
					e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
						if compileFails {
							return x, compileErr
						}
						return x, nil
					}
					var terminal *EvaluationFinish
					o := &recordingObserver{log: order, atFinish: func(f EvaluationFinish) {
						terminal = &f
						if observerPanics {
							panic("observer panic")
						}
					}}
					e.Observer = o
					type exit struct {
						returned   bool
						panicValue any
					}
					done := make(chan exit, 1)
					go func() {
						var r exit
						defer func() { r.panicValue = recover(); done <- r }()
						s := executableState()
						candidate, _ := e.Evaluate(ctx, s, s.Roots["root"])
						candidate.Close()
						r.returned = true
					}()
					select {
					case r := <-done:
						if r.returned || goexit && r.panicValue != nil || !goexit && r.panicValue != original {
							t.Fatalf("abnormal exit replaced: %+v", r)
						}
					case <-time.After(5 * time.Second):
						t.Fatal("abnormal evaluation did not exit")
					}
					if x.calls != 1 || terminal == nil || terminal.Candidate != nil || !errors.Is(terminal.Err, errEvaluationAbandoned) || compileFails && !errors.Is(terminal.Err, compileErr) {
						t.Fatalf("cleanup calls=%d terminal=%+v", x.calls, terminal)
					}
					if err := e.Close(); err != nil {
						t.Fatal(err)
					}
					if x.calls != 1 || b.live != 0 {
						t.Fatalf("callback retried or metadata retained: closes=%d bytes=%d", x.calls, b.live)
					}
				})
			}
		}
	}

}

func TestExecutableRunHookPreservesOwnedOutputs(t *testing.T) {
	e, base, compiled, _ := executableFixture(t)
	owner := &outputCloserFixture{}
	x := &ownedOutputFixture{testExecutable: base, owner: owner}
	e.Baseline = x
	ctx, b := executableMemoryContext(1 << 20)
	var arms []string
	e.Run = func(ctx context.Context, arm string, x Executable) (cudatrial.Receipt, error) {
		if arm == "baseline" && x != e.Baseline || arm == "candidate" && x != compiled {
			t.Fatal("run hook received replacement owner")
		}
		arms = append(arms, arm)
		return x.Run(ctx)
	}
	s := executableState()
	c, err := e.Evaluate(ctx, s, s.Roots["root"])
	if err != nil {
		t.Fatal(err)
	}
	c.Close()
	if owner.calls != 1 || len(arms) != 4 {
		t.Fatalf("private ownership erased: closes=%d arms=%v", owner.calls, arms)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if b.live != 0 {
		t.Fatalf("retained output or metadata bytes=%d", b.live)
	}
}
