package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/nccltest"
)

// ncclOps are the collectives a sweep runs: all-reduce for tensor
// parallelism and all-to-all for expert-parallel dispatch and combine.
var ncclOps = []nccltest.Operation{nccltest.AllReduce, nccltest.AllToAll}

// executable is the program runNCCL starts as the child probe.
var executable = os.Executable

// runNCCL sweeps each collective over the local GPUs in a child process,
// so a hung communicator can be killed.
func runNCCL(ctx context.Context, gpus, maxBytes int, timeout time.Duration) ([]Collective, error) {
	if gpus == 0 {
		return nil, fmt.Errorf("no GPUs")
	}
	exe, err := executable()
	if err != nil {
		return nil, err
	}
	var all []Collective
	for _, op := range ncclOps {
		ctx, cancel := context.WithTimeout(ctx, timeout)
		cmd := exec.CommandContext(ctx, exe, "-probe=nccl", "-nccl-op="+string(op), fmt.Sprintf("-nccl-gpus=%d", gpus), fmt.Sprintf("-nccl-max-bytes=%d", maxBytes))
		cmd.WaitDelay = 5 * time.Second
		// NCCL writes NCCL_DEBUG output to standard output, which
		// carries the results.
		if os.Getenv("NCCL_DEBUG_FILE") == "" {
			cmd.Env = append(os.Environ(), "NCCL_DEBUG_FILE=/dev/stderr")
		}
		var stderr strings.Builder
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		timedOut := ctx.Err() == context.DeadlineExceeded
		cancel()
		if timedOut {
			return all, fmt.Errorf("nccl %s: timed out after %v", op, timeout)
		}
		if err != nil {
			return all, fmt.Errorf("nccl %s: %v: %s", op, err, strings.TrimSpace(stderr.String()))
		}
		var rs []Collective
		if err := json.Unmarshal(out, &rs); err != nil {
			return all, fmt.Errorf("nccl %s: %v", op, err)
		}
		all = append(all, rs...)
	}
	return all, nil
}

// ncclProbe is the child process: it runs one sweep and writes the results
// as JSON. A failing size is recorded and ends the sweep.
func ncclProbe(op string, gpus, maxBytes int, out io.Writer) error {
	sizes, err := nccltest.Sizes(1<<10, maxBytes, 8)
	if err != nil {
		return err
	}
	o := nccltest.Options{Warmup: 5, Iterations: 20, Checks: 1}
	for i := range gpus {
		o.Devices = append(o.Devices, int32(i))
	}
	c := nccltest.Case{Operation: nccltest.Operation(op)}
	var rs []Collective
	err = nccltest.RunSweep(c, o, sizes, 1, func(r nccltest.Result) error {
		rs = append(rs, Collective{
			Op:          op,
			Ranks:       r.Ranks,
			Bytes:       r.Case.Bytes,
			TimeNanos:   r.GPUTime.Nanoseconds(),
			AlgGBs:      r.AlgorithmGBs,
			BusGBs:      r.BusGBs,
			Comparisons: r.Comparisons,
			Wrong:       r.Wrong,
			NCCLVersion: r.NCCLVersion,
		})
		return nil
	})
	if err != nil {
		rs = append(rs, Collective{Op: op, Ranks: gpus, Error: err.Error()})
	}
	return json.NewEncoder(out).Encode(rs)
}
