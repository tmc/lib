package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

const source = `extern "C" __global__ void capture_work(unsigned *out, unsigned trips) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 unsigned v = i + 1;
 #pragma unroll 1
 for (unsigned j = 0; j < trips; j++) v = v * 1664525u + 1013904223u;
 out[i] = v;
}`
const elements = 64
const trips = uint32(4096)
const byteLimit = 32 << 20

type runtimeBudget struct {
	records       uint64
	bytes, passes int
	deadline      int64
}

func runtimeLimits(plan cudaprof.CapturePlan) (runtimeBudget, error) {
	if plan.Budget.MaxRecords == 0 || plan.Budget.MaxBytes <= 0 || plan.Budget.MaxPasses <= 0 || plan.Budget.MaxWallNanos <= 0 {
		return runtimeBudget{}, errors.New("positive original plan budgets required")
	}
	read := func(name string) (uint64, error) {
		n, e := strconv.ParseUint(os.Getenv(name), 10, 63)
		if e != nil || n == 0 {
			return 0, fmt.Errorf("positive controller budget %s required", name)
		}
		return n, nil
	}
	records, e := read("CUDAPROF_CAPTURE_MAX_RECORDS")
	if e != nil {
		return runtimeBudget{}, e
	}
	bytes, e := read("CUDAPROF_CAPTURE_MAX_BYTES")
	if e != nil {
		return runtimeBudget{}, e
	}
	passes, e := read("CUDAPROF_CAPTURE_MAX_PASSES")
	if e != nil {
		return runtimeBudget{}, e
	}
	deadline, e := read("CUDAPROF_CAPTURE_DEADLINE_UNIX_NANOS")
	if e != nil {
		return runtimeBudget{}, e
	}
	if int64(deadline) <= time.Now().UnixNano() {
		return runtimeBudget{}, errors.New("controller wall budget exhausted before collection")
	}
	return runtimeBudget{min(records, plan.Budget.MaxRecords), int(min(bytes, uint64(plan.Budget.MaxBytes))), int(min(passes, uint64(plan.Budget.MaxPasses))), min(int64(deadline), time.Now().Add(time.Duration(plan.Budget.MaxWallNanos)).UnixNano())}, nil
}

type limitedOutput struct {
	bytes.Buffer
	remaining int
}

func (w *limitedOutput) Write(b []byte) (int, error) {
	if len(b) > w.remaining {
		return 0, errors.New("tool output byte budget exceeded")
	}
	w.remaining -= len(b)
	return w.Buffer.Write(b)
}
func boundedCommand(ctx context.Context, limit int, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	out := &limitedOutput{remaining: limit}
	diagnostic := &limitedOutput{remaining: 4096}
	cmd.Stdout = out
	cmd.Stderr = diagnostic
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%s: %w: %s", name, err, diagnostic.String())
	}
	return out.Bytes(), nil
}

func verifyWorkload(w cudacache.WorkloadIdentity, text string) error {
	if text != source {
		return errors.New("fixture source receipt mismatch")
	}
	if w.ProgramID != "capture-plan-native/lcg-v1" || w.NumericalMode != "uint32-wrap" || len(w.ShapeDims) != 2 || w.ShapeDims[0] != elements || w.ShapeDims[1] != int(trips) {
		return errors.New("fixture workload receipt mismatch")
	}
	return nil
}

func boundedPMConfig(cfg cuptipmsampling.Config, limits runtimeBudget) (cuptipmsampling.Config, error) {
	if cfg.DeviceIndex != 0 || limits.bytes < 4 || limits.records == 0 {
		return cuptipmsampling.Config{}, errors.New("PM device or remaining budget mismatch")
	}
	if cfg.MaxSamples == 0 || cfg.HardwareBufferSize == 0 {
		return cuptipmsampling.Config{}, errors.New("PM sample and hardware budgets must be positive")
	}
	cfg.MaxSamples = uint32(min(uint64(cfg.MaxSamples), limits.records))
	cap := uint(limits.bytes / 4)
	cfg.HardwareBufferSize = min(cfg.HardwareBufferSize, cap)
	if cfg.MaxCounterDataBytes == 0 {
		cfg.MaxCounterDataBytes = cap
	} else {
		cfg.MaxCounterDataBytes = min(cfg.MaxCounterDataBytes, cap)
	}
	return cfg, nil
}

// rangeCounterBudget reserves the other quarters for retained and encoded copies.
func rangeCounterBudget(limits runtimeBudget) (int, error) {
	if limits.bytes < 4 {
		return 0, errors.New("range counter image byte budget exhausted")
	}
	return limits.bytes / 4, nil
}

func retainObservedResult(plan cudaprof.CapturePlan, pass cudaprof.CapturePass, r cudaprof.CaptureProviderResult) cudaprof.CaptureProviderResult {
	if r.Status != "complete" {
		return r
	}
	var reasons []string
	if r.Identity != plan.Identity {
		reasons = append(reasons, "observed identity differs from plan")
	}
	if r.Environment != plan.Environment {
		reasons = append(reasons, "observed environment differs from plan")
	}
	cfg, ok := pass.Configs[r.Provider]
	if !ok || !reflect.DeepEqual(r.Config, cfg) {
		reasons = append(reasons, "observed provider configuration differs from plan")
	}
	if len(reasons) > 0 {
		r.Status = "partial"
		if r.Reason != "" {
			reasons = append([]string{r.Reason}, reasons...)
		}
		r.Reason = strings.Join(reasons, "; ")
	}
	return r
}

// publicationStorageBudget checks actual transport bytes and the controller's
// publication/code JSON byte fields, then their enclosing manifest Raw field.
// Future journal records and manifest metadata remain controller-budgeted.
func publicationStorageBudget(limit, publication, code, existing int) error {
	if limit <= 0 || publication < 0 || code < 0 || existing < 0 {
		return errors.New("invalid publication byte budget")
	}
	available := uint64(limit)
	spend := func(n uint64) bool {
		if n > available {
			return false
		}
		available -= n
		return true
	}
	if !spend(uint64(publication)) || !spend(uint64(code)) || !spend(uint64(existing)) {
		return errors.New("transport exceeds aggregate byte budget")
	}
	encoded := func(n int) uint64 { return 4 * ((uint64(n) + 2) / 3) }
	// Exact JSON for two nonnil []byte fields, including quotes and braces.
	packet := uint64(len(`{"publication":"","code_capture":""}`)) + encoded(publication) + encoded(code)
	outer := 2 + 4*((packet+2)/3)
	if outer > uint64(limit) || uint64(existing) > uint64(limit)-outer {
		return errors.New("publication exceeds nested encoded byte budget")
	}
	return nil
}
