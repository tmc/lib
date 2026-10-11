package ebpfhost

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Probe describes a file-offset uprobe. FetchArgs is tracefs fetch syntax.
type Probe struct {
	Path      string
	Offset    uint64
	Name      string
	Return    bool
	FetchArgs string
}

// Config describes an isolated tracefs capture. PID selects one process and its
// existing threads; Scheduler also filters scheduler payload PIDs for that process.
type Config struct {
	PID       int
	TraceRoot string
	Probes    []Probe
	Scheduler bool
	Lifecycle bool
	BufferKB  int
}

// Record is a raw tracefs line; its timestamp belongs to the selected mono clock.
// Raw records do not prove loss-free capture or application workload identity.
type Record struct{ Line string }

// Collector owns one tracefs instance and its uniquely named global probes.
// Read may have only one concurrent caller. Close interrupts Read.
type Collector struct {
	mu                    sync.Mutex
	root, instance, group string
	probes                []string
	fd                    int
	hasFD                 bool
	pending               []byte
	closed                bool
	cleanupDone           bool
	firstCleanupErr       error
	tracingDisabled       bool
	eventsDisabled        bool
}

var eventName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// Open installs probes and enables an isolated tracefs instance.
func Open(cfg Config) (_ *Collector, err error) {
	if runtime.GOOS != "linux" || (runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64") {
		return nil, errors.New("tracefs collector requires linux/amd64 or linux/arm64")
	}
	if cfg.PID <= 0 {
		return nil, errors.New("invalid process ID")
	}
	if cfg.TraceRoot == "" {
		cfg.TraceRoot = "/sys/kernel/tracing"
	}
	if cfg.BufferKB == 0 {
		cfg.BufferKB = 1024
	}
	if cfg.BufferKB < 1 || cfg.BufferKB > 65536 {
		return nil, errors.New("invalid buffer size")
	}
	if len(cfg.Probes) == 0 && !cfg.Scheduler && !cfg.Lifecycle {
		return nil, errors.New("no events requested")
	}
	for _, p := range cfg.Probes {
		if !eventName.MatchString(p.Name) || !filepath.IsAbs(p.Path) || strings.ContainsAny(p.Path, " \t\r\n:") || strings.ContainsAny(p.FetchArgs, "\r\n") {
			return nil, errors.New("invalid probe definition")
		}
	}
	instance, err := os.MkdirTemp(filepath.Join(cfg.TraceRoot, "instances"), "libhost_")
	if err != nil {
		return nil, fmt.Errorf("create trace instance: %w", err)
	}
	c := &Collector{root: cfg.TraceRoot, instance: instance, group: filepath.Base(instance)}
	defer func() {
		if err != nil {
			err = errors.Join(err, c.Close())
		}
	}()
	write := func(name, value string) error {
		return os.WriteFile(filepath.Join(instance, name), []byte(value), 0600)
	}
	if err = write("tracing_on", "0"); err != nil {
		return nil, err
	}
	if err = write("trace_clock", "mono"); err != nil {
		return nil, err
	}
	if err = write("buffer_size_kb", strconv.Itoa(cfg.BufferKB)); err != nil {
		return nil, err
	}
	tids, err := os.ReadDir(fmt.Sprintf("/proc/%d/task", cfg.PID))
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, tid := range tids {
		if _, e := strconv.Atoi(tid.Name()); e == nil {
			ids = append(ids, tid.Name())
		}
	}
	if len(ids) == 0 {
		return nil, errors.New("process has no threads")
	}
	if err = write("set_event_pid", strings.Join(ids, " ")); err != nil {
		return nil, err
	}
	if err = write("options/event-fork", "0"); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	for _, p := range cfg.Probes {
		if seen[p.Name] {
			return nil, errors.New("duplicate probe name")
		}
		seen[p.Name] = true
		kind := "p"
		if p.Return {
			kind = "r"
		}
		cmd := fmt.Sprintf("%s:%s/%s %s:0x%x %s\n", kind, c.group, p.Name, p.Path, p.Offset, p.FetchArgs)
		if err = appendControl(filepath.Join(c.root, "uprobe_events"), cmd); err != nil {
			return nil, err
		}
		c.probes = append(c.probes, p.Name)
		if err = write("events/"+c.group+"/"+p.Name+"/enable", "1"); err != nil {
			return nil, err
		}
	}
	if cfg.Scheduler || cfg.Lifecycle {
		// Per-event payload filters include switches away from and wakeups of target threads.
		if err = write("set_event_pid", ""); err != nil {
			return nil, err
		}
		pidExpr := func(field string) string {
			parts := make([]string, len(ids))
			for i, id := range ids {
				parts[i] = field + " == " + id
			}
			return "(" + strings.Join(parts, " || ") + ")"
		}
		var events []string
		if cfg.Scheduler {
			events = append(events, "sched_switch", "sched_wakeup", "sched_wakeup_new")
		}
		if cfg.Lifecycle {
			events = append(events, "sched_process_exec", "sched_process_exit")
		}
		for _, ev := range events {
			filter := pidExpr("pid")
			if ev == "sched_switch" {
				filter = pidExpr("prev_pid") + " || " + pidExpr("next_pid")
			}
			if err = write("events/sched/"+ev+"/filter", filter); err != nil {
				return nil, err
			}
			if err = write("events/sched/"+ev+"/enable", "1"); err != nil {
				return nil, err
			}
		}
		for _, p := range cfg.Probes {
			if err = write("events/"+c.group+"/"+p.Name+"/filter", pidExpr("common_pid")); err != nil {
				return nil, err
			}
		}
	}
	c.fd, err = syscall.Open(filepath.Join(instance, "trace_pipe"), syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}

	c.hasFD = true
	if err = write("tracing_on", "1"); err != nil {
		return nil, err
	}
	return c, nil
}
func appendControl(path, value string) error {
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if e != nil {
		return e
	}
	n, e := io.WriteString(f, value)
	if e == nil && n != len(value) {
		e = io.ErrShortWrite
	}
	return errors.Join(e, f.Close())
}

// Read waits for one raw line. It bounds each line to 1 MiB and respects cancellation.
func (c *Collector) Read(ctx context.Context) (Record, error) {
	var buf [65536]byte
	for {
		if err := ctx.Err(); err != nil {
			return Record{}, err
		}
		c.mu.Lock()
		if c.closed {
			c.mu.Unlock()
			return Record{}, os.ErrClosed
		}
		if i := strings.IndexByte(string(c.pending), '\n'); i >= 0 {
			line := string(c.pending[:i+1])
			c.pending = c.pending[i+1:]
			c.mu.Unlock()
			return Record{Line: line}, nil
		}
		n, err := syscall.Read(c.fd, buf[:])
		if n > 0 {
			c.pending = append(c.pending, buf[:n]...)
		}
		over := len(c.pending) > 1<<20
		c.mu.Unlock()
		if over {
			return Record{}, errors.New("trace line exceeds limit")
		}
		if err != nil && !errors.Is(err, syscall.EAGAIN) && !errors.Is(err, syscall.EINTR) {
			return Record{}, err
		}
		if n > 0 {
			continue
		}
		select {
		case <-ctx.Done():
			return Record{}, ctx.Err()
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// TraceClock reports the explicitly selected tracefs clock.
func (c *Collector) TraceClock() string { return "mono" }

// Close disables capture, closes the reader, removes owned probes and instance.
// Failed cleanup is retried on subsequent calls. The first cleanup error remains
// visible even after retry removes the remaining resources.
func (c *Collector) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	if c.cleanupDone {
		return c.firstCleanupErr
	}
	var errs []error
	if !c.tracingDisabled {
		err := os.WriteFile(filepath.Join(c.instance, "tracing_on"), []byte("0"), 0600)
		if err == nil {
			c.tracingDisabled = true
		}
		errs = append(errs, err)
	}
	if c.hasFD {
		c.hasFD = false
		errs = append(errs, syscall.Close(c.fd))
	} // Never retry close: fd may be reused, including EINTR.
	if !c.eventsDisabled {
		err := os.WriteFile(filepath.Join(c.instance, "events/enable"), []byte("0"), 0600)
		if err == nil {
			c.eventsDisabled = true
		}
		errs = append(errs, err)
	}
	if c.eventsDisabled {
		var remaining []string
		for _, name := range c.probes {
			err := appendControl(filepath.Join(c.root, "uprobe_events"), "-:"+c.group+"/"+name+"\n")
			if err != nil {
				remaining = append(remaining, name)
			}
			errs = append(errs, err)
		}
		c.probes = remaining
	}
	if c.eventsDisabled && c.tracingDisabled && len(c.probes) == 0 {
		err := os.Remove(c.instance)
		if err == nil || errors.Is(err, os.ErrNotExist) {
			c.cleanupDone = true
		} else {
			errs = append(errs, err)
		}
	}
	current := errors.Join(errs...)
	if c.firstCleanupErr == nil {
		c.firstCleanupErr = current
	}
	return errors.Join(c.firstCleanupErr, current)
}
