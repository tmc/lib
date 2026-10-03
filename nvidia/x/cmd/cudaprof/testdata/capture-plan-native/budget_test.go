package main

import (
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

func TestRuntimeBudgetRequiresObservedControllerLimits(t *testing.T) {
	plan := cudaprof.CapturePlan{Budget: cudaprof.CaptureBudget{MaxRecords: 8, MaxBytes: 1024, MaxPasses: 4, MaxWallNanos: int64(time.Minute)}}
	for k, v := range map[string]string{"CUDAPROF_CAPTURE_MAX_RECORDS": "3", "CUDAPROF_CAPTURE_MAX_BYTES": "512", "CUDAPROF_CAPTURE_MAX_PASSES": "2", "CUDAPROF_CAPTURE_DEADLINE_UNIX_NANOS": strconv.FormatInt(time.Now().Add(time.Minute).UnixNano(), 10)} {
		t.Setenv(k, v)
	}
	got, err := runtimeLimits(plan)
	if err != nil || got.records != 3 || got.bytes != 512 || got.passes != 2 {
		t.Fatalf("actual remaining limits: %+v %v", got, err)
	}
	t.Setenv("CUDAPROF_CAPTURE_MAX_PASSES", "0")
	if _, err := runtimeLimits(plan); err == nil {
		t.Fatal("zero remaining native pass budget admitted")
	}
	t.Setenv("CUDAPROF_CAPTURE_MAX_PASSES", "2")
	t.Setenv("CUDAPROF_CAPTURE_DEADLINE_UNIX_NANOS", "1")
	if _, err := runtimeLimits(plan); err == nil {
		t.Fatal("expired deadline admitted")
	}
}

func TestToolOutputRefusesBeforeAllocation(t *testing.T) {
	w := &limitedOutput{remaining: 3}
	if n, err := w.Write([]byte("abcd")); err == nil || n != 0 || w.Len() != 0 {
		t.Fatal("oversized tool output retained")
	}
	if n, err := w.Write([]byte("abc")); err != nil || n != 3 {
		t.Fatalf("bounded tool output refused: %d %v", n, err)
	}
}

func TestFixedWorkloadAdmission(t *testing.T) {
	valid := cudacache.WorkloadIdentity{ProgramID: "capture-plan-native/lcg-v1", ShapeDims: []int{64, 4096}, NumericalMode: "uint32-wrap"}
	for _, mode := range []string{"healthy", "program", "mode", "elements", "trips", "extra dimension", "source"} {
		t.Run(mode, func(t *testing.T) {
			w := valid
			w.ShapeDims = append([]int(nil), valid.ShapeDims...)
			text := source
			switch mode {
			case "program":
				w.ProgramID = "other"
			case "mode":
				w.NumericalMode = "float32"
			case "elements":
				w.ShapeDims[0]++
			case "trips":
				w.ShapeDims[1]--
			case "extra dimension":
				w.ShapeDims = append(w.ShapeDims, 1)
			case "source":
				text += " "
			}
			if err := verifyWorkload(w, text); (err == nil) != (mode == "healthy") {
				t.Fatalf("workload %s: %v", mode, err)
			}
		})
	}
}

func TestBoundedPMConfig(t *testing.T) {
	for _, tt := range []struct {
		name                      string
		bytes                     int
		records                   uint64
		device                    uint
		hardware, counter         uint
		samples                   uint32
		wantErr                   bool
		wantHardware, wantCounter uint
		wantSamples               uint32
	}{
		{"clamp", 400, 5, 0, 200, 200, 20, false, 100, 100, 5},
		{"smaller explicit caps", 400, 5, 0, 16, 8, 2, false, 16, 8, 2},
		{"unspecified counter cap", 400, 5, 0, 16, 0, 2, false, 16, 100, 2},
		{"minimum", 4, 1, 0, 16, 0, 2, false, 1, 1, 1},
		{"tiny bytes", 3, 1, 0, 16, 0, 2, true, 0, 0, 0},
		{"negative bytes", -1, 1, 0, 16, 0, 2, true, 0, 0, 0},
		{"device mismatch", 400, 5, 1, 16, 0, 2, true, 0, 0, 0},
		{"zero hardware", 400, 5, 0, 0, 0, 2, true, 0, 0, 0},
		{"zero samples", 400, 5, 0, 16, 0, 0, true, 0, 0, 0},
		{"zero records", 400, 0, 0, 16, 0, 2, true, 0, 0, 0},
		{"large record budget", 400, 1 << 40, 0, 16, 0, 7, false, 16, 100, 7},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := cuptipmsampling.Config{DeviceIndex: uint32(tt.device), HardwareBufferSize: tt.hardware, MaxCounterDataBytes: tt.counter, MaxSamples: tt.samples}
			got, err := boundedPMConfig(cfg, runtimeBudget{bytes: tt.bytes, records: tt.records})
			if (err != nil) != tt.wantErr {
				t.Fatalf("config=%+v error=%v", got, err)
			}
			if err == nil && (got.HardwareBufferSize != tt.wantHardware || got.MaxCounterDataBytes != tt.wantCounter || got.MaxSamples != tt.wantSamples) {
				t.Fatalf("caps changed: %+v", got)
			}
		})
	}
}
