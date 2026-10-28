package cudasearch

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// attemptLog stands in for a durable append-only attempt archive.
type attemptLog struct{ full bool }

func (l *attemptLog) StartEvaluation(_ context.Context, s EvaluationStart) error {
	if l.full {
		return errors.New("archive full")
	}
	fmt.Println("start", s.Attempt, s.Kind)
	return nil
}

func (l *attemptLog) FinishEvaluation(_ context.Context, f EvaluationFinish) error {
	fmt.Println("finish", f.Attempt, f.Candidate != nil, len(f.Evidence.Trials), f.Err)
	return nil
}

func ExampleEvaluationObserver() {
	// Host-only fixture. Real callers use private executable buffers and a live lease.
	order := []string{}
	x := &testExecutable{id: "image", value: 1, kind: cudatrial.TimingSummedKernel, duration: 1, order: &order}
	compiles := 0
	log := &attemptLog{}
	e := &ExecutableEvaluator{Workload: WorkloadBindings{Name: "example", InputNames: []string{"x"}, InputIdentities: map[string]string{"x": "input-x"}, NumericalPolicy: NumericalPolicyExact},
		Baseline: x, References: map[string][]float32{"out": {1}}, Rounds: 2, TimingKind: cudatrial.TimingSummedKernel, HardwareLockChecker: func() bool { return true },
		Compile: func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) {
			compiles++
			y := *x
			return &y, nil
		},
		Observer: log,
	}
	defer e.Close()
	s := executableState()
	c, err := e.Evaluate(context.Background(), s, s.Roots["root"])
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(c.ID)
	c.Close()

	// A start the archive cannot record refuses the attempt before compiling.
	log.full = true
	c, err = e.Evaluate(context.Background(), s, s.Roots["root"])
	fmt.Println(c == nil, err, compiles)
	// Output:
	// start 1 search
	// finish 1 true 4 <nil>
	// image
	// true cudasearch: record evaluation start: archive full 1
}
