package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudareg"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func registerOptions(t *testing.T) []RegisterOption {
	t.Helper()
	s, _, err := cudareg.GenerateFixture()
	if err != nil {
		t.Fatal(err)
	}
	r := cudareg.Resources{
		ImageDigest: strings.Repeat("a", 64), Entry: "entry", Target: s.Target.Name, DecodedTarget: "sm_90",
		Allocated: cudareg.Fact{Status: cudareg.FactAvailable, Count: 64},
		PeakLive:  cudareg.Fact{Status: cudareg.FactUnavailable, Reason: "analysis unavailable"},
		Cap:       cudareg.Fact{Status: cudareg.FactUnverified, Count: 128, Reason: "caller compiler cap"},
		Modeled:   cudareg.Fact{Status: cudareg.FactAvailable, Count: 0},
	}
	var options []RegisterOption
	for i, split := range [][2]uint32{{32, 96}, {40, 88}, {48, 80}} {
		s.Steps[0].Count, s.Steps[1].Count = split[0], split[1]
		c, err := cudareg.PublishCandidate(&r, s)
		if err != nil {
			t.Fatal(err)
		}
		options = append(options, RegisterOption{Identity: fmt.Sprintf("image-entry-sm90a-split-%d", i), Candidate: c})
	}
	return options
}

func registerLimits() RegisterSearchLimits {
	return RegisterSearchLimits{MaxCandidates: 3, MaxResourceBytes: 1 << 20}
}

func TestRegisterSearchMeasuredConsumer(t *testing.T) {
	e, base, _, _ := executableFixture(t)
	e.Generation = 7
	defer e.Close()
	options := registerOptions(t)
	want := options[1].Candidate
	var compiled []*testExecutable
	compile := func(ctx context.Context, expr *egraph.Expr, workload WorkloadBindings, option RegisterOption) (Executable, error) {
		if workload.Key() != e.Workload.Key() || expr.Op != "fixture" {
			t.Fatal("lost workload or expression")
		}
		durations := []time.Duration{9, 3, 6}
		x := &testExecutable{id: option.Identity, value: 1, kind: base.kind, duration: durations[len(compiled)] * time.Microsecond, order: base.order}
		compiled = append(compiled, x)
		// A compiler owns its detached argument, not the retained evidence.
		option.Candidate.Resources.Entry = "compiler mutation"
		delete(option.Candidate.Schedule.Groups, "producer")
		return x, nil
	}
	previous := reflect.ValueOf(e.Compile).Pointer()
	result, err := e.SearchRegisters(context.Background(), &egraph.Expr{Op: "fixture"}, options, registerLimits(), compile)
	if err != nil {
		t.Fatal(err)
	}
	defer result.Close()
	if result.Evaluations != 3 || result.Winner.ID != options[1].Identity || result.Winner.Cost != 3000 || result.Winner.CostSource != CostMeasured || result.Winner.Generation != 7 {
		t.Fatalf("result=%+v winner=%+v", result, result.Winner)
	}
	if !reflect.DeepEqual(result.Registers, want) || result.Winner.WorkloadKey != e.Workload.Key() || result.Winner.InputKey != inputKey(e.Workload) || result.Winner.VerificationPolicy != e.Policy {
		t.Fatal("lost resource caveats, identity or numerical policy")
	}
	if unsafe.StringData(result.Registers.Resources.Entry) == unsafe.StringData(options[1].Candidate.Resources.Entry) || unsafe.StringData(result.Winner.ID) == unsafe.StringData(options[1].Identity) {
		t.Fatal("winner retains caller string backing storage")
	}
	if result.Registers.Resources.PeakLive.Status != cudareg.FactUnavailable || result.Registers.Resources.Modeled.Count != 0 || result.Registers.TargetReason == "" {
		t.Fatal("turned absence into a measured value or lost suffix caveat")
	}
	options[1].Candidate.Resources.Entry = "caller mutation"
	options[1].Candidate.Schedule.Steps[0].Count = 24
	if result.Registers.Resources.Entry != "entry" || result.Registers.Schedule.Steps[0].Count != 40 {
		t.Fatal("winner aliases caller storage")
	}
	for _, x := range compiled {
		if x.closed != 1 {
			t.Fatalf("%s closes=%d", x.id, x.closed)
		}
	}
	if base.closed != 0 || reflect.ValueOf(e.Compile).Pointer() != previous {
		t.Fatal("closed borrowed baseline or changed compiler")
	}
	if driver.Loaded() || nvrtc.Loaded() {
		t.Fatal("host consumer loaded a vendor library")
	}
	result.Close()
	result.Close()
	if result.Winner != nil || result.Registers.Resources != nil {
		t.Fatal("close retains winner")
	}
}

func TestRegisterSearchPreflight(t *testing.T) {
	for _, name := range []string{"count", "bytes", "zero limits", "empty", "missing identity", "duplicate identity", "unverified", "forged status", "forged evidence", "allocation", "target", "pool", "cancel"} {
		t.Run(name, func(t *testing.T) {
			e, _, _, _ := executableFixture(t)
			defer e.Close()
			options, limits := registerOptions(t), registerLimits()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch name {
			case "count":
				limits.MaxCandidates = 2
			case "bytes":
				options[2].Candidate.Resources.PeakLive.Reason = strings.Repeat("x", limits.MaxResourceBytes)
			case "zero limits":
				limits = RegisterSearchLimits{}
			case "empty":
				options = nil
			case "missing identity":
				options[2].Identity = ""
			case "duplicate identity":
				options[2].Identity = options[0].Identity
			case "unverified":
				options[2].Candidate.Resources.DecodedTarget = ""
				options[2].Candidate, _ = cudareg.PublishCandidate(options[2].Candidate.Resources, options[2].Candidate.Schedule)
			case "forged status":
				options[2].Candidate.Status = cudareg.PublicationUnverified
			case "forged evidence":
				options[2].Candidate.TargetReason = "suffix proven"
			case "allocation":
				options[2].Candidate.Resources.Allocated.Count++
			case "target":
				options[2].Candidate.Resources.Target = "sm_100a"
			case "pool":
				options[2].Candidate.Schedule.CTARegisterPool = 1
			case "cancel":
				cancel()
			}
			calls := 0
			r, err := e.SearchRegisters(ctx, &egraph.Expr{Op: "fixture"}, options, limits, func(context.Context, *egraph.Expr, WorkloadBindings, RegisterOption) (Executable, error) {
				calls++
				return nil, nil
			})
			if err == nil || r != nil || calls != 0 {
				t.Fatalf("result=%v err=%v calls=%d", r, err, calls)
			}
			if name == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		})
	}
}

func TestRegisterSearchFailureAndOwnership(t *testing.T) {
	for _, name := range []string{"incorrect", "identity", "unmeasured", "cleanup", "compile owner", "cancel", "tie"} {
		t.Run(name, func(t *testing.T) {
			e, base, _, _ := executableFixture(t)
			defer e.Close()
			options := registerOptions(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			failure := errors.New("retained compiler owner")
			var owners []*testExecutable
			compile := func(_ context.Context, _ *egraph.Expr, _ WorkloadBindings, option RegisterOption) (Executable, error) {
				x := &testExecutable{id: option.Identity, value: 1, kind: base.kind, duration: 5 * time.Microsecond, order: base.order}
				owners = append(owners, x)
				if len(owners) == 2 {
					switch name {
					case "incorrect":
						x.value = 2
					case "identity":
						x.id = "wrong image"
					case "unmeasured":
						x.duration = 0
					case "cleanup":
						x.closeErr = failure
					case "compile owner":
						x.closeErr = failure
						return x, failure
					case "cancel":
						cancel()
					}
				}
				return x, nil
			}
			r, err := e.SearchRegisters(ctx, &egraph.Expr{Op: "fixture"}, options, registerLimits(), compile)
			if r == nil || r.Winner == nil || r.Winner.ID != options[0].Identity {
				t.Fatalf("lost admitted best: result=%+v err=%v", r, err)
			}
			defer r.Close()
			if name == "tie" {
				if err != nil || r.Evaluations != 3 {
					t.Fatalf("%+v %v", r, err)
				}
				return
			}
			if err == nil || r.Evaluations != 2 || len(owners) != 2 || owners[1].closed != 1 {
				t.Fatalf("result=%+v err=%v owners=%+v", r, err, owners)
			}
			if name == "cleanup" || name == "compile owner" {
				if !errors.Is(err, failure) {
					t.Fatal(err)
				}
				if err := e.Close(); !errors.Is(err, failure) {
					t.Fatalf("lost pending owner: %v", err)
				}
				owners[1].closeErr = nil
				if err := e.Close(); err != nil || owners[1].closed != 3 {
					t.Fatalf("close err=%v calls=%d", err, owners[1].closed)
				}
			}
			if name == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		})
	}
}
