package hipmega_test

import (
	"errors"
	"math"
	"math/big"
	"reflect"
	"testing"

	"github.com/tmc/lib/rocm/driver"
	"github.com/tmc/lib/rocm/x/hipmega"
)

type hostOperations struct {
	limits                             hipmega.Limits
	active                             int32
	limitsErr, occupancyErr, launchErr error
	calls                              []string
	limitsFunction, occupancyFunction  driver.Function
	threads                            int32
	shared                             uint
	launched                           hipmega.Request
}

func newHostOperations() *hostOperations {
	return &hostOperations{
		limits: hipmega.Limits{
			Device: 0, CooperativeLaunch: true, Multiprocessors: 2,
			MaxGrid: [3]int64{1024, 1024, 64}, MaxBlock: [3]int64{1024, 1024, 64},
			MaxThreadsPerBlock: 1024, MaxSharedMemPerBlock: 65536,
			StaticSharedMemBytes: 1024, MaxDynamicSharedMemBytes: 32768,
		},
		active: 4,
	}
}

func (h *hostOperations) operations() hipmega.Operations {
	return hipmega.Operations{
		Limits: func(f driver.Function) (hipmega.Limits, error) {
			h.calls = append(h.calls, "limits")
			h.limitsFunction = f
			return h.limits, h.limitsErr
		},
		Occupancy: func(f driver.Function, threads int32, shared uint) (int32, error) {
			h.calls = append(h.calls, "occupancy")
			h.occupancyFunction, h.threads, h.shared = f, threads, shared
			return h.active, h.occupancyErr
		},
		CooperativeLaunch: func(r hipmega.Request) error {
			h.calls = append(h.calls, "launch")
			h.launched = r
			return h.launchErr
		},
	}
}

func hostRequest() hipmega.Request {
	return hipmega.Request{
		Device: 0, Function: driver.Function(123), Grid: [3]int64{8, 1, 1},
		Block: [3]int64{128, 1, 1}, SharedMemBytes: 1024,
		Stream: driver.Stream(456), KernelParams: 789, Progress: hipmega.CoResident,
	}
}

func TestLaunchUsesExactOccupancyAndCooperativeOperation(t *testing.T) {
	for _, dimensions := range []struct {
		name        string
		grid, block [3]int64
	}{
		{"linear", [3]int64{8, 1, 1}, [3]int64{128, 1, 1}},
		{"three dimensions", [3]int64{2, 2, 2}, [3]int64{4, 4, 8}},
	} {
		t.Run(dimensions.name, func(t *testing.T) {
			h := newHostOperations()
			r := hostRequest()
			r.Grid, r.Block = dimensions.grid, dimensions.block
			got, err := hipmega.Launch(h.operations(), r)
			if err != nil {
				t.Fatal(err)
			}
			want := hipmega.Admission{Device: 0, GridBlocks: 8, ThreadsPerBlock: 128, ActiveBlocksPerMultiprocessor: 4, Multiprocessors: 2, ResidentBlocks: 8, Progress: hipmega.CoResident}
			if got != want || h.limitsFunction != r.Function || h.occupancyFunction != r.Function || h.threads != 128 || h.shared != 1024 || h.launched != r || !reflect.DeepEqual(h.calls, []string{"limits", "occupancy", "launch"}) {
				t.Fatalf("admission %+v; operations %+v", got, h)
			}
		})
	}
}

func TestOccupancyResultGatesLaunch(t *testing.T) {
	// Identical request and limits, different occupancy result. This is a host
	// callback model, not an observed HIP residency or device execution.
	for _, active := range []int32{1, 4} {
		h := newHostOperations()
		h.active = active
		got, err := hipmega.Launch(h.operations(), hostRequest())
		if active == 1 {
			if err == nil || got != (hipmega.Admission{}) || !reflect.DeepEqual(h.calls, []string{"limits", "occupancy"}) {
				t.Fatalf("oversubscribed launch: admission %+v error %v calls %v", got, err, h.calls)
			}
		} else if err != nil || got.ResidentBlocks != 8 || h.launched.Function == 0 {
			t.Fatalf("admitted launch: admission %+v error %v calls %v", got, err, h.calls)
		}
	}
}

func TestRefusalsNeverLaunch(t *testing.T) {
	tests := []struct {
		name      string
		change    func(*hipmega.Request, *hostOperations)
		wantCalls int
	}{
		{"missing progress", func(r *hipmega.Request, h *hostOperations) { r.Progress = hipmega.NoProgress }, 0},
		{"unknown progress", func(r *hipmega.Request, h *hostOperations) { r.Progress = 255 }, 0},
		{"cluster", func(r *hipmega.Request, h *hostOperations) { r.RequiresCluster = true }, 0},
		{"tmem", func(r *hipmega.Request, h *hostOperations) { r.RequiresTMEM = true }, 0},
		{"negative device", func(r *hipmega.Request, h *hostOperations) { r.Device = -1 }, 0},
		{"zero function", func(r *hipmega.Request, h *hostOperations) { r.Function = 0 }, 0},
		{"zero grid", func(r *hipmega.Request, h *hostOperations) { r.Grid[1] = 0 }, 0},
		{"negative grid", func(r *hipmega.Request, h *hostOperations) { r.Grid[2] = -1 }, 0},
		{"zero block", func(r *hipmega.Request, h *hostOperations) { r.Block[0] = 0 }, 0},
		{"negative block", func(r *hipmega.Request, h *hostOperations) { r.Block[1] = -1 }, 0},
		{"grid binding width", func(r *hipmega.Request, h *hostOperations) { r.Grid[0] = math.MaxUint32 + 1 }, 0},
		{"block binding width", func(r *hipmega.Request, h *hostOperations) { r.Block[0] = math.MaxUint32 + 1 }, 0},
		{"grid product overflow", func(r *hipmega.Request, h *hostOperations) { r.Grid = [3]int64{math.MaxUint32, math.MaxUint32, 2} }, 0},
		{"block product overflow", func(r *hipmega.Request, h *hostOperations) { r.Block = [3]int64{math.MaxUint32, math.MaxUint32, 2} }, 0},
		{"occupancy block width", func(r *hipmega.Request, h *hostOperations) { r.Block = [3]int64{1 << 30, 2, 1} }, 0},
		{"negative shared", func(r *hipmega.Request, h *hostOperations) { r.SharedMemBytes = -1 }, 0},
		{"shared binding width", func(r *hipmega.Request, h *hostOperations) { r.SharedMemBytes = math.MaxUint32 + 1 }, 0},
		{"wrong current device", func(r *hipmega.Request, h *hostOperations) { h.limits.Device = 1 }, 1},
		{"no cooperative capability", func(r *hipmega.Request, h *hostOperations) { h.limits.CooperativeLaunch = false }, 1},
		{"zero multiprocessors", func(r *hipmega.Request, h *hostOperations) { h.limits.Multiprocessors = 0 }, 1},
		{"negative multiprocessors", func(r *hipmega.Request, h *hostOperations) { h.limits.Multiprocessors = -1 }, 1},
		{"missing grid limit", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxGrid[0] = 0 }, 1},
		{"missing block limit", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxBlock[2] = -1 }, 1},
		{"grid axis", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxGrid[0] = 7 }, 1},
		{"block axis", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxBlock[0] = 127 }, 1},
		{"thread count", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxThreadsPerBlock = 127 }, 1},
		{"missing threads", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxThreadsPerBlock = 0 }, 1},
		{"missing shared limit", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxSharedMemPerBlock = 0 }, 1},
		{"negative static", func(r *hipmega.Request, h *hostOperations) { h.limits.StaticSharedMemBytes = -1 }, 1},
		{"negative dynamic limit", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxDynamicSharedMemBytes = -1 }, 1},
		{"static too large", func(r *hipmega.Request, h *hostOperations) { h.limits.StaticSharedMemBytes = 65537 }, 1},
		{"static plus dynamic", func(r *hipmega.Request, h *hostOperations) { h.limits.StaticSharedMemBytes = 65000 }, 1},
		{"dynamic function limit", func(r *hipmega.Request, h *hostOperations) { h.limits.MaxDynamicSharedMemBytes = 1023 }, 1},
		{"shared sum overflow", func(r *hipmega.Request, h *hostOperations) {
			h.limits.MaxSharedMemPerBlock = math.MaxInt64
			h.limits.StaticSharedMemBytes = math.MaxInt64 - 1
			h.limits.MaxDynamicSharedMemBytes = math.MaxInt64
		}, 1},
		{"zero occupancy", func(r *hipmega.Request, h *hostOperations) { h.active = 0 }, 2},
		{"negative occupancy", func(r *hipmega.Request, h *hostOperations) { h.active = -1 }, 2},
		{"residency overflow", func(r *hipmega.Request, h *hostOperations) { h.limits.Multiprocessors = math.MaxInt64; h.active = 3 }, 2},
		{"too many workers", func(r *hipmega.Request, h *hostOperations) { r.Grid[0] = 9 }, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newHostOperations()
			r := hostRequest()
			tc.change(&r, h)
			got, err := hipmega.Launch(h.operations(), r)
			if err == nil || got != (hipmega.Admission{}) || len(h.calls) != tc.wantCalls || h.launched.Function != 0 {
				t.Fatalf("admission %+v error %v calls %v launched %+v", got, err, h.calls, h.launched)
			}
		})
	}
}

func TestOperationErrorsAndMissingCallbacks(t *testing.T) {
	failure := errors.New("synthetic operation error")
	for _, stage := range []string{"limits", "occupancy", "launch"} {
		t.Run(stage, func(t *testing.T) {
			h := newHostOperations()
			switch stage {
			case "limits":
				h.limitsErr = failure
			case "occupancy":
				h.occupancyErr = failure
			case "launch":
				h.launchErr = failure
			}
			got, err := hipmega.Launch(h.operations(), hostRequest())
			if !errors.Is(err, failure) || got != (hipmega.Admission{}) || h.calls[len(h.calls)-1] != stage {
				t.Fatalf("admission %+v error %v calls %v", got, err, h.calls)
			}
		})
	}
	for _, stage := range []string{"limits", "occupancy", "launch"} {
		t.Run("missing/"+stage, func(t *testing.T) {
			h := newHostOperations()
			ops := h.operations()
			switch stage {
			case "limits":
				ops.Limits = nil
			case "occupancy":
				ops.Occupancy = nil
			case "launch":
				ops.CooperativeLaunch = nil
			}
			if _, err := hipmega.Launch(ops, hostRequest()); err == nil || len(h.calls) != 0 {
				t.Fatalf("error %v calls %v", err, h.calls)
			}
		})
	}
}

func TestHIPOperationsConstructionDoesNotInvokeBindings(t *testing.T) {
	// The test process must set CODEGEN_SKIP_BINDINGS=1 before import init.
	// Only build the adapter; none of the returned HIP operations is invoked.
	ops := hipmega.HIPOperations()
	if ops.Limits == nil || ops.Occupancy == nil || ops.CooperativeLaunch == nil {
		t.Fatal("incomplete HIP binding adapter")
	}
}

func TestZeroDynamicSharedAndDefaultStream(t *testing.T) {
	h := newHostOperations()
	h.limits.MaxDynamicSharedMemBytes = 0
	r := hostRequest()
	r.SharedMemBytes, r.Stream, r.KernelParams = 0, 0, 0
	if _, err := hipmega.Launch(h.operations(), r); err != nil || h.launched != r {
		t.Fatalf("error %v launched %+v", err, h.launched)
	}
}

func FuzzLaunchGeometry(f *testing.F) {
	f.Add(int64(4), int64(1), int64(1), int64(64), int64(1), int64(1))
	f.Add(int64(math.MaxUint32), int64(math.MaxUint32), int64(2), int64(1), int64(1), int64(1))
	f.Add(int64(1), int64(1), int64(1), int64(1<<30), int64(2), int64(1))
	f.Add(int64(0), int64(-1), int64(1), int64(1), int64(1), int64(1))
	f.Fuzz(func(t *testing.T, gx, gy, gz, bx, by, bz int64) {
		h := newHostOperations()
		h.active, h.limits.Multiprocessors = 1, math.MaxInt64
		h.limits.MaxGrid = [3]int64{math.MaxUint32, math.MaxUint32, math.MaxUint32}
		h.limits.MaxBlock = h.limits.MaxGrid
		h.limits.MaxThreadsPerBlock = math.MaxInt32
		r := hostRequest()
		r.Grid, r.Block = [3]int64{gx, gy, gz}, [3]int64{bx, by, bz}
		// Independent unbounded integer products are the arithmetic oracle.
		valid := true
		grid, block := big.NewInt(1), big.NewInt(1)
		for i := range r.Grid {
			if r.Grid[i] <= 0 || r.Grid[i] > math.MaxUint32 || r.Block[i] <= 0 || r.Block[i] > math.MaxUint32 {
				valid = false
			}
			grid.Mul(grid, big.NewInt(r.Grid[i]))
			block.Mul(block, big.NewInt(r.Block[i]))
		}
		valid = valid && grid.Cmp(big.NewInt(math.MaxInt64)) <= 0 && block.Cmp(big.NewInt(math.MaxInt32)) <= 0
		got, err := hipmega.Launch(h.operations(), r)
		if valid {
			if err != nil || got.GridBlocks != grid.Uint64() || int64(got.ThreadsPerBlock) != block.Int64() || h.launched != r {
				t.Fatalf("valid geometry: admission %+v error %v launched %+v", got, err, h.launched)
			}
		} else if err == nil || h.launched.Function != 0 {
			t.Fatalf("invalid geometry launched: admission %+v error %v", got, err)
		}
	})
}
