// cudatracehost records Linux CUDA host observations as JSON lines.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/tmc/lib/nvidia/x/cudatracehost"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() (err error) {
	var cfg cudatracehost.Config
	var symbols string
	flag.StringVar(&cfg.Backend, "backend", "bpf", "bpf or tracefs; no automatic fallback")
	flag.IntVar(&cfg.PID, "pid", 0, "target process ID")
	flag.StringVar(&cfg.Library, "library", "", "absolute mapped CUDA library path")
	flag.StringVar(&cfg.Version, "version", "", "caller-reported CUDA library version")
	flag.StringVar(&cfg.Layer, "layer", "driver", "driver or runtime; exactly one layer")
	flag.StringVar(&symbols, "symbols", "cuLaunchKernel,cuMemcpyHtoD_v2,cuMemcpyDtoH_v2,cuCtxSynchronize,cuMemAlloc_v2,cuMemFree_v2", "comma-separated exact symbols")
	flag.StringVar(&cfg.TraceRoot, "tracefs", "/sys/kernel/tracing", "tracefs mount")
	flag.BoolVar(&cfg.Scheduler, "scheduler", true, "capture scheduler evidence for current target threads")
	flag.BoolVar(&cfg.UserStack, "user-stack", false, "capture bounded frame-pointer user stacks (bpf only)")
	flag.Parse()
	cfg.Symbols = strings.Split(symbols, ",")
	c, err := cudatracehost.Open(cfg)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	out := json.NewEncoder(os.Stdout)
	var writtenNative uint64
	defer func() {
		stats, statsErr := c.StopAndStats()
		closeErr := c.Close()
		err = errors.Join(err, statsErr, closeErr)
		status := "closed"
		if closeErr != nil {
			status = "error"
		}
		reason := "stopped"
		if err != nil {
			reason = err.Error()
		}
		err = errors.Join(err, out.Encode(map[string]any{"schema": "cudatracehost/v1", "kind": "completion", "complete": false, "reason": reason, "cleanup_status": status, "ring": stats, "written_native": writtenNative, "lifecycle_loss_status": "unknown"}))
	}()
	allocation := "unavailable"
	ring := "unavailable"
	if cfg.Backend == "bpf" {
		allocation = "per_record_status"
		ring = "kernel_counters"
	}
	if err := out.Encode(map[string]any{"manifest": c.Manifest(), "schema": "cudatracehost/v1", "backend": cfg.Backend, "clock": "linux_monotonic", "late_attach": true, "complete": false, "loss_accounting": ring, "lifecycle_loss_status": "unknown", "allocation_output": allocation, "scheduler_thread_scope": "attachment_snapshot"}); err != nil {
		return err
	}
	for {
		r, err := c.Next(ctx)
		if r.Raw != "" || r.Native != nil || r.Capture != nil || r.Call != nil {
			if writeErr := out.Encode(r); writeErr != nil {
				return writeErr
			}
			if r.Native != nil {
				writtenNative++
			}
		}
		if err != nil {
			return err
		}
	}
}
