package cudabpfdev

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestCompiledResourceGate(t *testing.T) {
	for _, tc := range []struct {
		name      string
		resources compiledResources
		limits    resourceLimits
		valid     bool
	}{
		{"fits", compiledResources{24, 0, 0, 32}, resourceLimits{32, 0}, true},
		{"register boundary", compiledResources{32, 0, 0, 32}, resourceLimits{32, 0}, true},
		{"register overrun", compiledResources{33, 0, 0, 32}, resourceLimits{32, 0}, false},
		{"unbudgeted", compiledResources{24, 0, 0, 32}, resourceLimits{}, false},
		{"shared boundary", compiledResources{24, 64, 0, 32}, resourceLimits{32, 64}, true},
		{"shared overrun", compiledResources{24, 65, 0, 32}, resourceLimits{32, 64}, false},
		{"spills", compiledResources{24, 0, 4, 32}, resourceLimits{32, 0}, false},
		{"threads", compiledResources{24, 0, 0, 31}, resourceLimits{32, 0}, false},
		{"invalid registers", compiledResources{-1, 0, 0, 32}, resourceLimits{32, 0}, false},
		{"invalid shared", compiledResources{24, -1, 0, 32}, resourceLimits{32, 0}, false},
		{"invalid local", compiledResources{24, 0, -1, 32}, resourceLimits{32, 0}, false},
		{"invalid threads", compiledResources{24, 0, 0, -1}, resourceLimits{32, 0}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeRun{}
			e, source := runFixture(t, f)
			e.limits = tc.limits
			e.ops.resources = func(driver.Function) (compiledResources, error) { return tc.resources, nil }
			err := e.execute(context.Background(), source, 2, 5)
			if (err == nil) != tc.valid {
				t.Fatalf("error=%v valid=%v", err, tc.valid)
			}
			if !tc.valid {
				for _, forbidden := range []string{"stream", "alloc1", "zero1", "launch"} {
					if slices.Contains(f.calls, forbidden) {
						t.Fatalf("resource rejection reached %s", forbidden)
					}
				}
				if _, err := e.Record(); err == nil {
					t.Fatal("rejected kernel produced success record")
				}
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if e.module != 0 {
				t.Fatal("resource rejection leaked module")
			}
		})
	}
}

func TestQueryResources(t *testing.T) {
	attrs := []driver.FunctionAttribute{driver.FuncAttributeNumRegs, driver.FuncAttributeSharedSizeBytes, driver.FuncAttributeLocalSizeBytes, driver.FuncAttributeMaxThreadsPerBlock}
	values := []int32{24, 64, 0, 128}
	for failed := -1; failed < len(attrs); failed++ {
		calls := 0
		r, err := queryResources(func(attr driver.FunctionAttribute) (int32, error) {
			i := calls
			calls++
			if i >= len(attrs) || attr != attrs[i] {
				t.Fatalf("unexpected resource attribute %v", attr)
			}
			if i == failed {
				return 0, errRun
			}
			return values[i], nil
		})
		if failed < 0 {
			if err != nil || r != (compiledResources{24, 64, 0, 128}) {
				t.Fatalf("resources=%+v error=%v", r, err)
			}
		} else if !errors.Is(err, errRun) || r != (compiledResources{}) || calls != failed+1 {
			t.Fatalf("partial attributes escaped: %+v %v calls=%d", r, err, calls)
		}
	}
}

func TestTranslationRetainsResourceBudget(t *testing.T) {
	p := Program{Version: 1, Registers: 1, Slots: 1, Code: []Instruction{{Op: Exit}}}
	b := Budget{MaxInstructions: 1, MaxSteps: 1, MaxRegisters: 32, MaxSlots: 1, MaxSharedBytes: 200000}
	for _, target := range []struct {
		name   string
		shared uint32
	}{{"sm_100a", 200000}, {"sm_121a", 102400}} {
		tr, err := Translate(p, b, target.name)
		if err != nil {
			t.Fatal(err)
		}
		// Public analysis fields cannot relax the private admission budget.
		tr.DeclaredRegisters = 1000
		tr.Analysis.Registers = 1000
		if tr.limits != (resourceLimits{32, target.shared}) {
			t.Fatalf("%s limits=%+v", target.name, tr.limits)
		}
	}
}
