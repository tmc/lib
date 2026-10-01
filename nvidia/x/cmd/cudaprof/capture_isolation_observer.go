package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
)

// isolationWatch adapts the existing GPU-window policy: only live descendants
// of a verified local lock owner are owned. It acquires no lock. Containers
// and ancestry that cannot be read remain unresolved.
type isolationWatch struct {
	receipt isolation.Receipt
	stop    chan struct{}
	done    chan struct{}
	owner   isolation.Process
	query   func(string) ([][]string, error)
}

func startIsolationWatch() *isolationWatch {
	w := &isolationWatch{stop: make(chan struct{}), done: make(chan struct{})}
	w.receipt = isolation.Receipt{Version: 1, Observer: "cudaprof-gpu-window", ObserverVersion: "1", Producer: "cudaprof", Method: "nvidia-smi-compute-residency", OwnershipMethod: "live-process-generation-ancestry", RequestedInterval: int64(250 * time.Millisecond), Window: isolation.Window{Domain: "CLOCK_MONOTONIC"}}
	if now, ok := monotonicNanos(); ok {
		w.receipt.Window.Start = int64(now)
	} else {
		w.receipt.Errors = append(w.receipt.Errors, "CLOCK_MONOTONIC unavailable")
	}
	dir := os.Getenv("GPU_LOCKDIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "ml-explore/.gpu-lock/lock")
	}
	if b, err := os.ReadFile(filepath.Join(dir, "owner")); err == nil {
		fields := strings.Fields(string(b))
		var pid int
		var host, token string
		for _, f := range fields {
			key, value, _ := strings.Cut(f, "=")
			switch key {
			case "pid":
				pid, _ = strconv.Atoi(value)
			case "host":
				host = value
			case "token":
				token = value
			}
		}
		owner := captureProcessIdentity(pid)
		if owner.Host == host && owner.Start != "" && token != "" {
			owned, known := isolationDescendant(os.Getpid(), owner)
			if known && owned {
				w.owner = owner
				w.receipt.Lock = &isolation.Lock{Owner: owner, Token: token}
				w.receipt.OwnershipResolved = true
			}
		}
	}
	if !w.receipt.OwnershipResolved {
		w.receipt.Errors = append(w.receipt.Errors, "live lock owner ancestry unavailable")
	}
	w.sample()
	go func() {
		defer close(w.done)
		timer := time.NewTicker(250 * time.Millisecond)
		defer timer.Stop()
		for {
			select {
			case <-timer.C:
				w.sample()
			case <-w.stop:
				return
			}
		}
	}()
	return w
}
func (w *isolationWatch) finish(result *captureResult) {
	close(w.stop)
	<-w.done
	w.sample()
	if now, ok := monotonicNanos(); ok {
		w.receipt.Window.End = int64(now)
	}
	w.receipt.BoundarySamples = true
	b := result.IsolationBinding
	if phase := result.PhaseReceipt; phase != nil && phase.Status == "MEASURED" && phase.StartTimestamp <= phase.StopTimestamp && phase.StopTimestamp <= uint64(^uint64(0)>>1) {
		b.Window = isolation.Window{Domain: "CLOCK_MONOTONIC", Start: int64(phase.StartTimestamp), End: int64(phase.StopTimestamp)}
		b.Scope = "capture_phase"
	}
	// Process termination alone cannot prove completion of every followed or
	// orphaned GPU context. This deliberately remains false.
	b.CompletionKnown = false
	if len(w.receipt.Devices) == 1 {
		b.GPUs = []string{w.receipt.Devices[0].GPU}
	} else {
		b.GPUs = nil
		w.receipt.Errors = appendBounded(w.receipt.Errors, "multiple physical GPUs: target CUDA ordinal to UUID mapping unavailable")
	}
	result.IsolationBinding = b
	w.receipt.Binding = b
	raw, err := json.Marshal(&w.receipt)
	if err != nil || len(raw) > isolation.MaxBytes {
		out := isolation.Evaluate(nil, "", b)
		out.Reasons = []string{"observer receipt exceeded bounded retention"}
		result.Isolation = &out
		return
	}
	result.isolationRaw = raw
	r, hash, _ := isolation.Read(strings.NewReader(string(raw)))
	out := isolation.Evaluate(r, hash, b)
	result.Isolation = &out
}
func (w *isolationWatch) sample() {
	now, ok := monotonicNanos()
	if !ok {
		return
	}
	if w.owner.Start != "" && captureProcessIdentity(w.owner.PID) != w.owner {
		w.receipt.OwnershipResolved = false
		w.receipt.Errors = appendBounded(w.receipt.Errors, "lock owner generation changed")
	}
	at := int64(now)
	query := w.query
	if query == nil {
		query = isolationQuery
	}
	rows, err := query("--query-gpu=uuid")
	if err != nil {
		w.receipt.Errors = appendBounded(w.receipt.Errors, "device query: "+err.Error())
		return
	}
	for _, row := range rows {
		if len(row) != 1 || !strings.HasPrefix(row[0], "GPU-") {
			w.receipt.Errors = appendBounded(w.receipt.Errors, "physical GPU UUID unavailable")
			continue
		}
		gpu := row[0]
		found := -1
		for i := range w.receipt.Devices {
			if w.receipt.Devices[i].GPU == gpu {
				found = i
			}
		}
		if found < 0 {
			w.receipt.Devices = append(w.receipt.Devices, isolation.DeviceCoverage{GPU: gpu, First: at})
			found = len(w.receipt.Devices) - 1
		}
		d := &w.receipt.Devices[found]
		gap := at - d.Last
		if d.Samples > 0 && gap > d.MaxGap {
			d.MaxGap = gap
		}
		d.Last = at
		d.Samples++
	}
	queryStart, clockOK := monotonicNanos()
	rows, err = query("--query-compute-apps=gpu_uuid,pid,used_gpu_memory")
	queryEnd, endClockOK := monotonicNanos()
	if !clockOK || !endClockOK {
		w.receipt.Errors = appendBounded(w.receipt.Errors, "process query clock unavailable")
		return
	}
	queryWindow := isolation.Window{Domain: "CLOCK_MONOTONIC", Start: int64(queryStart), End: int64(queryEnd)}
	at = int64(queryEnd)
	if err != nil {
		w.receipt.Errors = appendBounded(w.receipt.Errors, "process query: "+err.Error())
		return
	}
	w.receipt.Samples++
	for _, row := range rows {
		if len(row) != 3 {
			w.receipt.Errors = appendBounded(w.receipt.Errors, "malformed process observation")
			continue
		}
		pid, err := strconv.Atoi(row[1])
		if err != nil {
			w.receipt.Errors = appendBounded(w.receipt.Errors, "invalid process PID")
			continue
		}
		p := captureProcessIdentity(pid)
		owned, known := isolationDescendant(pid, w.owner)
		if !known {
			w.receipt.OwnershipResolved = false
			w.receipt.Errors = appendBounded(w.receipt.Errors, "unresolved process ancestry")
			continue
		}
		if owned {
			continue
		}
		memory, err := strconv.ParseUint(row[2], 10, 64)
		if err != nil || memory > ^uint64(0)/(1<<20) {
			w.receipt.Errors = appendBounded(w.receipt.Errors, "resident memory unavailable")
			memory = 0
		}
		if len(w.receipt.Outside) >= 4096 {
			w.receipt.Errors = appendBounded(w.receipt.Errors, "outside observation bound exceeded")
			continue
		}
		exe, _ := os.Readlink(fmt.Sprintf("/proc/%d/exe", pid))
		w.receipt.Outside = append(w.receipt.Outside, isolation.Observation{QueryWindow: &queryWindow, Process: p, GPU: row[0], First: at, Last: at, ResidentBytes: memory * (1 << 20), Executable: exe})
	}
	for _, d := range w.receipt.Devices {
		if d.MaxGap > w.receipt.ObservedInterval {
			w.receipt.ObservedInterval = d.MaxGap
		}
	}
}
func appendBounded(a []string, s string) []string {
	if len(a) < 64 {
		return append(a, s)
	}
	return a
}
func isolationDescendant(pid int, owner isolation.Process) (bool, bool) {
	if owner.PID <= 0 || owner.Start == "" {
		return false, false
	}
	for range 64 {
		p := captureProcessIdentity(pid)
		if p.Start == "" {
			return false, false
		}
		if pid == owner.PID {
			return p == owner, true
		}
		b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
		if err != nil {
			return false, false
		}
		end := strings.LastIndexByte(string(b), ')')
		if end < 0 {
			return false, false
		}
		fields := strings.Fields(string(b)[end+1:])
		if len(fields) < 2 {
			return false, false
		}
		parent, err := strconv.Atoi(fields[1])
		if err != nil {
			return false, false
		}
		if parent <= 1 {
			return false, true
		}
		pid = parent
	}
	return false, false
}

type isolationOutput struct{ bytes []byte }

func (w *isolationOutput) Write(p []byte) (int, error) {
	if len(w.bytes)+len(p) > 1<<20 {
		return 0, fmt.Errorf("observer output exceeds limit")
	}
	w.bytes = append(w.bytes, p...)
	return len(p), nil
}
func isolationQuery(query string) ([][]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi", query, "--format=csv,noheader,nounits")
	var out isolationOutput
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	if err := cmd.Run(); err != nil {
		return nil, err
	}
	rows, err := csv.NewReader(strings.NewReader(string(out.bytes))).ReadAll()
	for _, r := range rows {
		for i := range r {
			r[i] = strings.TrimSpace(r[i])
		}
	}
	return rows, err
}
