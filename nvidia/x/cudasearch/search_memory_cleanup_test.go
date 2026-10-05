package cudasearch

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"testing"
)

// A failed drain retains its callback and prevents destruction of any resource
// below it. Successful callbacks are removed so Close can safely be retried.
type memoryFixtureOwner struct {
	steps []memoryFixtureClose
}

type memoryFixtureClose struct {
	stage int
	name  string
	close func() error
}

const (
	memoryCollectorStage = iota
	memoryEvaluatorStage
	memoryExecutableStage
	memoryModuleStage
	memoryDeviceStage
)

func (o *memoryFixtureOwner) add(stage int, name string, close func() error) {
	o.steps = append(o.steps, memoryFixtureClose{stage, name, close})
	sort.SliceStable(o.steps, func(i, j int) bool { return o.steps[i].stage < o.steps[j].stage })
}

func (o *memoryFixtureOwner) Close() error {
	for len(o.steps) > 0 {
		step := o.steps[0]
		if err := step.close(); err != nil {
			return fmt.Errorf("close %s: %w", step.name, err)
		}
		o.steps[0] = memoryFixtureClose{}
		o.steps = o.steps[1:]
	}
	return nil
}

func TestMemoryFixtureOwnerRepeatedFailure(t *testing.T) {
	for failing := memoryCollectorStage; failing <= memoryDeviceStage; failing++ {
		t.Run(fmt.Sprint(failing), func(t *testing.T) {
			var owner memoryFixtureOwner
			var calls []int
			failure := errors.New("drain incomplete")
			refuse := true
			// Deliberately register in reverse order, as allocation order need
			// not follow dependency order.
			for stage := memoryDeviceStage; stage >= memoryCollectorStage; stage-- {
				stage := stage
				owner.add(stage, fmt.Sprint(stage), func() error {
					calls = append(calls, stage)
					if stage == failing && refuse {
						return failure
					}
					return nil
				})
			}
			for attempt := 0; attempt < 3; attempt++ {
				if err := owner.Close(); !errors.Is(err, failure) {
					t.Fatalf("attempt %d: %v", attempt, err)
				}
				if len(owner.steps) != memoryDeviceStage-failing+1 {
					t.Fatal("failed owner or dependencies not retained")
				}
			}
			want := make([]int, failing+1)
			for stage := range want {
				want[stage] = stage
			}
			want = append(want, failing, failing)
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("destroyed dependencies or retried completed owner: %v want %v", calls, want)
			}
			refuse = false
			if err := owner.Close(); err != nil || len(owner.steps) != 0 {
				t.Fatalf("retry: %v, retained %d", err, len(owner.steps))
			}
			for stage := failing; stage <= memoryDeviceStage; stage++ {
				want = append(want, stage)
			}
			if err := owner.Close(); err != nil || !reflect.DeepEqual(calls, want) {
				t.Fatalf("completed close changed callbacks: %v, %v", calls, err)
			}
		})
	}
}
