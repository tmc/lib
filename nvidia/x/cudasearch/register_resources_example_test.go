package cudasearch_test

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudareg"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// This host-only executable supplies controlled values and synthetic timing.
// It does not compile or run a CUDA kernel or establish hardware acceptance.
type registerExampleExecutable struct{ id string }

func (x registerExampleExecutable) Identity() string { return x.id }
func (x registerExampleExecutable) Close() error     { return nil }
func (x registerExampleExecutable) Outputs(context.Context) (map[string][]float32, error) {
	return map[string][]float32{"out": {1}}, nil
}
func (x registerExampleExecutable) Run(context.Context) (cudatrial.Receipt, error) {
	r := cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingSummedKernel}
	_ = r.RecordTiming(r.TimingKind, time.Microsecond)
	return r, nil
}

func ExampleExecutableEvaluator_SearchRegisters() {
	schedule, _, _ := cudareg.GenerateFixture()
	resources := cudareg.Resources{
		ImageDigest: strings.Repeat("a", 64), Entry: "fixture", Target: schedule.Target.Name, DecodedTarget: "sm_90",
		Allocated: cudareg.Fact{Status: cudareg.FactAvailable, Count: 64},
		PeakLive:  cudareg.Fact{Status: cudareg.FactUnavailable, Reason: "not analyzed"},
		Cap:       cudareg.Fact{Status: cudareg.FactUnavailable, Reason: "not supplied"},
		Modeled:   cudareg.Fact{Status: cudareg.FactUnavailable, Reason: "not modeled"},
	}
	publication, _ := cudareg.PublishCandidate(&resources, schedule)
	e := &cudasearch.ExecutableEvaluator{
		Workload: cudasearch.WorkloadBindings{Name: "host example", InputNames: []string{"x"}, InputIdentities: map[string]string{"x": "synthetic-input"}, NumericalPolicy: cudasearch.NumericalPolicyExact},
		Baseline: registerExampleExecutable{"baseline"}, References: map[string][]float32{"out": {1}},
		Rounds: 2, TimingKind: cudatrial.TimingSummedKernel,
		HardwareLockChecker: func() bool { return true }, // synthetic adapter, no GPU authority
	}
	defer e.Close()
	options := []cudasearch.RegisterOption{{Identity: "synthetic-image-entry-schedule", Candidate: publication}}
	limits := cudasearch.RegisterSearchLimits{MaxCandidates: 1, MaxResourceBytes: 64 << 10}
	result, err := e.SearchRegisters(context.Background(), &egraph.Expr{Op: "fixture"}, options, limits,
		func(_ context.Context, _ *egraph.Expr, _ cudasearch.WorkloadBindings, option cudasearch.RegisterOption) (cudasearch.Executable, error) {
			return registerExampleExecutable{option.Identity}, nil
		})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer result.Close()
	fmt.Println(result.Evaluations, result.Registers.Status)
	// Output: 1 protocol feasible
}

func ExampleRegisterOption() {
	o := cudasearch.RegisterOption{Identity: "image-entry-target-and-schedule"}
	fmt.Println(o.Identity)
	// Output: image-entry-target-and-schedule
}

func ExampleRegisterSearchLimits() {
	l := cudasearch.RegisterSearchLimits{MaxCandidates: 8, MaxResourceBytes: 1 << 20}
	fmt.Println(l.MaxCandidates, l.MaxResourceBytes)
	// Output: 8 1048576
}

func ExampleRegisterSearchResult_Close() {
	r := &cudasearch.RegisterSearchResult{}
	r.Close()
	r.Close()
	fmt.Println(r.Winner == nil)
	// Output: true
}
