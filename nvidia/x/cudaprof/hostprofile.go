package cudaprof

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sync"
	"time"
)

// HostProfile owns process-wide runtime tracing and CPU profiling started by
// StartHostProfile. Stop must be called before reading either output.
// CPU profile samples are aggregate samples; their individual timestamps are
// not retained. The runtime trace retains timestamped scheduler observations.
type HostProfile struct {
	mu                 sync.Mutex
	stopped            bool
	tracing, profiling bool
	finish             func() error
	err                error
	writers            []*hostProfileWriter
}

// StartHostProfile starts runtime tracing and CPU profiling into separate
// writers. A nil writer disables that output. Both nil is an error.
// Existing tracing or profiling owners cause an error and are left running.
// Callers must coordinate direct runtime/trace and runtime/pprof start/stop
// calls with this helper; those APIs do not expose ownership tokens.
func StartHostProfile(traceOutput, cpuOutput io.Writer) (*HostProfile, error) {
	if traceOutput == nil && cpuOutput == nil {
		return nil, fmt.Errorf("start host profile: no output")
	}
	p := new(HostProfile)
	wrap := func(w io.Writer) io.Writer {
		tracker := &hostProfileWriter{writer: w}
		p.writers = append(p.writers, tracker)
		return tracker
	}
	if traceOutput != nil {
		if err := trace.Start(wrap(traceOutput)); err != nil {
			return nil, fmt.Errorf("start runtime trace: %w", err)
		}
		p.tracing = true
	}
	if cpuOutput != nil {
		if err := pprof.StartCPUProfile(wrap(cpuOutput)); err != nil {
			if p.tracing {
				trace.Stop()
			}
			return nil, fmt.Errorf("start CPU profile: %w", err)
		}
		p.profiling = true
	}
	return p, nil
}

// Stop stops only the collectors started by StartHostProfile and waits for
// their writes to finish. Repeated calls are harmless. A zero HostProfile is
// safe to stop. Stop reports retained writer failures after the runtime
// collectors have finished writing.
func (p *HostProfile) Stop() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.stopped {
		return p.err
	}
	p.stopped = true
	if p.profiling {
		pprof.StopCPUProfile()
	}
	if p.tracing {
		trace.Stop()
	}
	for _, w := range p.writers {
		p.err = errors.Join(p.err, w.Err())
	}
	if p.finish != nil {
		p.err = errors.Join(p.err, p.finish())
	}
	return p.err
}

// StartCaptureHostProfile retains a runtime trace, CPU profile, and clock
// receipt in CUDAPROF_STACK_DIR for the capture launcher to attach. Collection
// starts only when called explicitly. Stop must complete before process exit.
// The receipt records wall-clock bounds, not a qualified mapping from runtime
// trace timestamps to CUPTI timestamps.
func StartCaptureHostProfile() (*HostProfile, error) {
	dir := os.Getenv("CUDAPROF_STACK_DIR")
	if dir == "" || stackProcess == "" {
		return nil, fmt.Errorf("start capture host profile: missing output directory or process identity")
	}
	prefix := filepath.Join(dir, "go-"+stackProcess)
	tracePath, cpuPath, receiptPath := prefix+".trace", prefix+".pprof", prefix+"-host.json"
	tr, err := os.OpenFile(tracePath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, fmt.Errorf("create runtime trace: %w", err)
	}
	cpu, err := os.OpenFile(cpuPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		tr.Close()
		os.Remove(tracePath)
		return nil, fmt.Errorf("create CPU profile: %w", err)
	}
	start := time.Now().UTC()
	p, err := StartHostProfile(tr, cpu)
	if err != nil {
		tr.Close()
		cpu.Close()
		os.Remove(tracePath)
		os.Remove(cpuPath)
		return nil, err
	}
	p.finish = func() error {
		closeErr := errors.Join(tr.Close(), cpu.Close())
		receipt := struct {
			Version     int       `json:"version"`
			PID         int       `json:"pid"`
			Process     string    `json:"process"`
			Session     string    `json:"session"`
			Generation  uint64    `json:"generation"`
			Start       time.Time `json:"start_wall_utc"`
			End         time.Time `json:"end_wall_utc"`
			Alignment   string    `json:"gpu_clock_alignment"`
			CPUSampling string    `json:"cpu_sample_timestamps"`
			GOOS        string    `json:"goos"`
			GOARCH      string    `json:"goarch"`
			GoVersion   string    `json:"go_version"`
		}{1, os.Getpid(), stackProcess, os.Getenv("CUDAPROF_SESSION"), stackGeneration(os.Getenv("CUDAPROF_SESSION")), start, time.Now().UTC(), "unavailable", "not_retained", runtime.GOOS, runtime.GOARCH, runtime.Version()}
		data, marshalErr := json.Marshal(receipt)
		if marshalErr != nil {
			return errors.Join(closeErr, marshalErr)
		}
		f, openErr := os.OpenFile(receiptPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if openErr != nil {
			return errors.Join(closeErr, openErr)
		}
		_, writeErr := f.Write(append(data, '\n'))
		return errors.Join(closeErr, writeErr, f.Close())
	}
	return p, nil
}

// hostProfileWriter remembers errors discarded by the runtime collectors.
type hostProfileWriter struct {
	mu     sync.Mutex
	writer io.Writer
	err    error
}

func (w *hostProfileWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.err != nil {
		return 0, w.err
	}
	n, err := w.writer.Write(p)
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	if err != nil {
		w.err = fmt.Errorf("write host profile: %w", err)
	}
	return n, err
}

func (w *hostProfileWriter) Err() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.err
}
