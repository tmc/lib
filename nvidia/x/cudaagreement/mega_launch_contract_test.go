package cudaagreement

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamega"
)

// These controls use only inert Go ownership objects and query callbacks.
// They neither load the driver nor compile or launch a device kernel.
func TestMegaLaunchContract(t *testing.T) {
	for _, ctas := range []uint32{1, 2} {
		t.Run(string(rune('0'+ctas))+"cta", func(t *testing.T) {
			dev := new(cuda.Device)
			grid := cudamega.Grid{CTAs: ctas, ThreadsPerCTA: 64, MaxElements: 8}
			var calls []string
			current := func() (driver.Device, driver.Context, error) {
				calls = append(calls, "context")
				return dev.Raw(), 7, nil
			}
			q := megaLaunchQueries{current: current,
				cooperative: func() (int, error) { calls = append(calls, "cooperative"); return 1, nil },
				launchEx:    func() bool { calls = append(calls, "launch-ex"); return true },
			}
			a, reason, err := prepareMegaLaunch(grid, 3, 8, dev, dev, dev, q)
			if err != nil || reason != "" {
				t.Fatalf("prepare: %q, %v", reason, err)
			}
			if a.scratchWords != 24 || a.orderWords != 1+3*int(ctas) {
				t.Fatalf("sizes: %+v", a)
			}
			reason, err = a.resident(grid, dev, dev, dev, dev, current, func() (int32, int, error) {
				calls = append(calls, "occupancy")
				return 1, 2, nil
			})
			if err != nil || reason != "" {
				t.Fatalf("resident: %q, %v", reason, err)
			}
			// A bare inert function permits Describe without a module or driver.
			// Production passes OwnedKernel through this same configuration helper.
			l := megaLauncher(grid, cudalaunch.Kernel(driver.Function(1)))
			var count int
			err = a.launch(l, dev, dev, dev, dev, current, func(l *cudalaunch.Launcher) error {
				calls = append(calls, "launch")
				count++
				d, err := l.Describe()
				if err != nil {
					return err
				}
				if d.Grid != driver.Dims(ctas, 1, 1) || d.Block != driver.Dims(64, 1, 1) || d.SharedMem != 0 {
					t.Fatalf("geometry: %+v", d)
				}
				attrs := l.Attrs()
				if ctas == 1 {
					if len(attrs) != 0 {
						t.Fatalf("single CTA attributes: %v", attrs)
					}
				} else if len(attrs) != 1 || attrs[0].Id != driver.LaunchAttributeCooperative || *attrs[0].Value.Cooperative() != 1 {
					t.Fatalf("cooperative attributes: %v", attrs)
				}
				return nil
			})
			if err != nil || count != 1 {
				t.Fatalf("launch count %d: %v", count, err)
			}
			want := []string{"context"}
			if ctas > 1 {
				want = append(want, "cooperative", "launch-ex")
			}
			want = append(want, "context", "occupancy", "context", "launch")
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("calls %v, want %v", calls, want)
			}
		})
	}
}

func TestMegaPreflightRefusal(t *testing.T) {
	boom := errors.New("query failed")
	for _, tc := range []struct {
		name             string
		words            int
		max              uint32
		owner            int
		raw              driver.Device
		ctx              driver.Context
		cap              int
		ex               bool
		queryErr, capErr error
		want             string
		calls            int
	}{
		{name: "extent", words: 9, max: 8, want: "extent", calls: 0},
		{name: "uint32", words: int(uint64(math.MaxUint32) + 1), want: "extent", calls: 0},
		{name: "empty", words: 0, want: "extent", calls: 0},
		{name: "stream", words: 8, owner: 1, want: "differ", calls: 0},
		{name: "modules", words: 8, owner: 2, want: "differ", calls: 0},
		{name: "device", words: 8, raw: 1, ctx: 7, want: "wrong device", calls: 1},
		{name: "context", words: 8, ctx: 0, want: "absent", calls: 1},
		{name: "query", words: 8, queryErr: boom, want: "query failed", calls: 1},
		{name: "capability query", words: 8, ctx: 7, capErr: boom, want: "query failed", calls: 2},
		{name: "unsupported", words: 8, ctx: 7, want: "does not support", calls: 2},
		{name: "symbol", words: 8, ctx: 7, cap: 1, want: "cuLaunchKernelEx", calls: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev, other := new(cuda.Device), new(cuda.Device)
			streamDev, modulesDev := dev, dev
			if tc.owner == 1 {
				streamDev = other
			}
			if tc.owner == 2 {
				modulesDev = other
			}
			var calls int
			q := megaLaunchQueries{
				current:     func() (driver.Device, driver.Context, error) { calls++; return tc.raw, tc.ctx, tc.queryErr },
				cooperative: func() (int, error) { calls++; return tc.cap, tc.capErr },
				launchEx:    func() bool { calls++; return tc.ex },
			}
			_, reason, err := prepareMegaLaunch(cudamega.Grid{CTAs: 2, ThreadsPerCTA: 64, MaxElements: tc.max}, 3, tc.words, dev, streamDev, modulesDev, q)
			got := reason
			if err != nil {
				got = err.Error()
			}
			if !strings.Contains(got, tc.want) || calls != tc.calls {
				t.Fatalf("got %q, calls %d, want %q, calls %d", got, calls, tc.want, tc.calls)
			}
			if tc.queryErr != nil || tc.capErr != nil {
				if !errors.Is(err, boom) {
					t.Fatalf("lost query error: %v", err)
				}
			}
		})
	}
}

func TestMegaBufferSizes(t *testing.T) {
	for _, tc := range []struct {
		name                    string
		tasks, words, ctas, max uint64
		want                    bool
	}{
		{"valid", 3, 8, 2, math.MaxInt64, false},
		{"scratch int bytes", 1024, math.MaxUint32, 1, math.MaxInt32, true},
		{"order int bytes", 1024, 1, math.MaxUint32, math.MaxInt32, true},
		{"scratch uint overflow", 1024, math.MaxUint64, 1, math.MaxInt64, true},
		{"too many tasks", 1025, 1, 1, math.MaxInt64, true},
		{"order cta range", 1, 1, uint64(math.MaxUint32) + 1, math.MaxInt64, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := megaBufferSizes(tc.tasks, tc.words, tc.ctas, tc.max)
			if (err != nil) != tc.want {
				t.Fatalf("error %v, want refusal %v", err, tc.want)
			}
		})
	}
}

func TestMegaResidencyRefusal(t *testing.T) {
	boom := errors.New("occupancy query failed")
	for _, tc := range []struct {
		name                     string
		owner                    int
		raw                      driver.Device
		ctx                      driver.Context
		perSM                    int32
		sms                      int
		queryErr, errorOccupancy error
		want                     string
		calls                    int
	}{
		{name: "kernel", owner: 4, ctx: 7, want: "ownership", calls: 0},
		{name: "executor", owner: 1, ctx: 7, want: "ownership", calls: 0},
		{name: "stream", owner: 2, ctx: 7, want: "ownership", calls: 0},
		{name: "modules", owner: 3, ctx: 7, want: "ownership", calls: 0},
		{name: "changed device", raw: 1, ctx: 7, want: "changed", calls: 0},
		{name: "changed context", ctx: 8, want: "changed", calls: 0},
		{name: "context query", queryErr: boom, want: "query failed", calls: 0},
		{name: "occupancy query", ctx: 7, errorOccupancy: boom, want: "query failed", calls: 1},
		{name: "zero occupancy", ctx: 7, perSM: 0, sms: 2, want: "invalid", calls: 1},
		{name: "negative occupancy", ctx: 7, perSM: -1, sms: 2, want: "invalid", calls: 1},
		{name: "zero sms", ctx: 7, perSM: 1, want: "invalid", calls: 1},
		{name: "negative sms", ctx: 7, perSM: 1, sms: -1, want: "invalid", calls: 1},
		{name: "overflow", ctx: 7, perSM: 2, sms: math.MaxInt, want: "invalid", calls: 1},
		{name: "too wide", ctx: 7, perSM: 1, sms: 1, want: "exceeds residency", calls: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev, other := new(cuda.Device), new(cuda.Device)
			owners := []*cuda.Device{dev, dev, dev, dev}
			if tc.owner > 0 {
				owners[tc.owner-1] = other
			}
			a := megaAdmission{device: dev, context: 7}
			var calls int
			reason, err := a.resident(cudamega.Grid{CTAs: 2}, owners[0], owners[1], owners[2], owners[3], func() (driver.Device, driver.Context, error) { return tc.raw, tc.ctx, tc.queryErr }, func() (int32, int, error) { calls++; return tc.perSM, tc.sms, tc.errorOccupancy })
			got := reason
			if err != nil {
				got = err.Error()
			}
			if !strings.Contains(got, tc.want) || calls != tc.calls {
				t.Fatalf("got %q, calls %d, want %q, calls %d", got, calls, tc.want, tc.calls)
			}
		})
	}
}

func TestMegaLaunchRefusalAndError(t *testing.T) {
	boom := errors.New("launch failed")
	for _, tc := range []struct {
		name      string
		owner     int
		raw       driver.Device
		ctx       driver.Context
		submitErr error
		want      string
		count     int
	}{
		{name: "executor", owner: 1, ctx: 7, want: "ownership"},
		{name: "stream", owner: 2, ctx: 7, want: "ownership"},
		{name: "modules", owner: 3, ctx: 7, want: "ownership"},
		{name: "kernel", owner: 4, ctx: 7, want: "ownership"},
		{name: "changed device", raw: 1, ctx: 7, want: "changed"},
		{name: "changed context", ctx: 8, want: "changed"},
		{name: "launch error", ctx: 7, submitErr: boom, want: "launch failed", count: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev, other := new(cuda.Device), new(cuda.Device)
			owners := []*cuda.Device{dev, dev, dev, dev}
			if tc.owner > 0 {
				owners[tc.owner-1] = other
			}
			a := megaAdmission{device: dev, context: 7}
			var count int
			l := megaLauncher(cudamega.Grid{CTAs: 2, ThreadsPerCTA: 64}, cudalaunch.Kernel(1))
			err := a.launch(l, owners[0], owners[1], owners[2], owners[3], func() (driver.Device, driver.Context, error) { return tc.raw, tc.ctx, nil }, func(*cudalaunch.Launcher) error { count++; return tc.submitErr })
			if err == nil || !strings.Contains(err.Error(), tc.want) || count != tc.count {
				t.Fatalf("error %v, count %d, want %q, count %d", err, count, tc.want, tc.count)
			}
			if tc.submitErr != nil && !errors.Is(err, boom) {
				t.Fatalf("lost launch error: %v", err)
			}
		})
	}
}

func TestMegaRunExtentBeforeCompiler(t *testing.T) {
	dev := new(cuda.Device)
	var calls int
	e := MegaExecutor{Device: dev, Modules: NewModules(dev), Grid: cudamega.Grid{CTAs: 2, ThreadsPerCTA: 64, MaxElements: 8}, Compile: func(string, cudamega.Target) (cudakernel.Kind, []byte, error) {
		calls++
		return cudakernel.KindPTX, nil, nil
	}}
	_, reason, err := e.run(dev.DefaultStream(), nil, megaUnit{steps: []int{0}, words: 9}, nil)
	if err != nil || !strings.Contains(reason, "extent") || calls != 0 || len(e.scratch) != 0 || e.order.Len() != 0 {
		t.Fatalf("reason %q, error %v, compile calls %d, scratch %v, order %d", reason, err, calls, e.scratch, e.order.Len())
	}
}
