package cudatracehost

import (
	"context"
	"debug/elf"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmc/lib/ebpfhost/bpfprobe"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"

	"github.com/tmc/lib/ebpfhost"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
)

// Config identifies a running target and exact CUDA library version. Library
// must be the target's mapped image, not an unrelated host library. Layer is
// either runtime or driver; combining layers would double count nested APIs.
type Config struct {
	Backend                            string
	PID                                int
	Library, Version, Layer, TraceRoot string
	Symbols                            []string
	Scheduler                          bool
	UserStack                          bool
	MaxPending                         int
}

// Collector owns probe lifetime and bounded host-call correlation. It exposes
// raw scheduler events without claiming an exact request/goroutine join.
type Collector struct {
	execKnown   bool
	delivered   atomic.Uint64
	execID      uint64
	cleanupErr  error
	done        chan struct{}
	stopOnce    sync.Once
	stateMu     sync.Mutex
	terminalErr error
	closeMu     sync.Mutex
	held        *os.File
	group       *bpfprobe.Group
	events      chan transportEvent
	cancel      context.CancelFunc
	wait        sync.WaitGroup
	tagNames    map[uint64]string
	trace       *ebpfhost.Collector
	calls       *Correlator
	process     cudaprobe.Process
	pid         int
	manifest    Manifest
}

// Result preserves raw transport even when a line cannot be decoded. Scheduler
// is an observed host scheduling line, not GPU evidence.
type Result struct {
	Call      *CallInterval
	Native    *bpfprobe.Event
	Raw       string
	Capture   *cudaprobe.Capture
	Scheduler bool
}

// Open resolves and attaches entry/return probes. Permission, missing symbols,
// unsupported signatures and tracefs failures return errors without fallback.
func Open(cfg Config) (*Collector, error) {
	if runtime.GOOS != "linux" || (runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64") {
		return nil, fmt.Errorf("cudatracehost: linux/amd64 or linux/arm64 required")
	}
	if !filepath.IsAbs(cfg.Library) {
		return nil, fmt.Errorf("cudatracehost: library path must be absolute")
	}
	if cfg.PID <= 0 || cfg.Library == "" || cfg.Version == "" || (cfg.Layer != "runtime" && cfg.Layer != "driver") {
		return nil, fmt.Errorf("cudatracehost: missing target/library/layer identity")
	}
	generation, err := ebpfhost.ProcessStartTime(cfg.PID)
	if err != nil {
		return nil, err
	}
	// Resolve the target mount namespace path and verify that the file is mapped.
	maps, err := os.ReadFile(fmt.Sprintf("/proc/%d/maps", cfg.PID))
	if err != nil {
		return nil, err
	}
	mapped := false
	var mappedInode uint64
	var mappedDev string
	for _, line := range strings.Split(string(maps), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 6 && fields[5] == cfg.Library {
			mapped = true
			mappedDev = fields[3]
			mappedInode, err = strconv.ParseUint(fields[4], 10, 64)
			if err != nil {
				return nil, err
			}
			break
		}
	}
	if !mapped {
		return nil, fmt.Errorf("cudatracehost: library is not mapped by target")
	}
	path := fmt.Sprintf("/proc/%d/root%s", cfg.PID, cfg.Library)
	held, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	releaseHeld := true
	defer func() {
		if releaseHeld {
			held.Close()
		}
	}()
	info, err := held.Stat()
	if err != nil {
		return nil, err
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat.Ino != mappedInode || !sameDevice(uint64(stat.Dev), mappedDev) {
		return nil, fmt.Errorf("cudatracehost: mapped library inode changed")
	}
	imageELF, err := elf.NewFile(held)
	if err != nil {
		return nil, fmt.Errorf("cudatracehost: ELF image: %w", err)
	}
	if err := validateImageABI(imageELF.Class, imageELF.Data, imageELF.Machine, runtime.GOARCH); err != nil {
		return nil, err
	}
	path = fmt.Sprintf("/proc/self/fd/%d", held.Fd())
	if cfg.MaxPending == 0 {
		cfg.MaxPending = 4096
	}
	var images []ebpfhost.Image
	var targets []Target
	var probes []ebpfhost.Probe
	tagNames := map[uint64]string{}
	seenSymbols := map[string]bool{}
	for i, symbol := range cfg.Symbols {
		if seenSymbols[symbol] {
			return nil, fmt.Errorf("cudatracehost: duplicate symbol %s", symbol)
		}
		seenSymbols[symbol] = true
		n, layer, ok := signature(symbol)
		if !ok || layer != cfg.Layer {
			return nil, fmt.Errorf("cudatracehost: unsupported %s symbol %s", cfg.Layer, symbol)
		}
		image, err := ebpfhost.ResolveImage(path, symbol)
		if err != nil {
			return nil, err
		}
		image.Path = cfg.Library
		images = append(images, image)
		tagNames[uint64(2*i)] = fmt.Sprintf("cuda%d_in", i)
		tagNames[uint64(2*i+1)] = fmt.Sprintf("cuda%d_out", i)
		name := fmt.Sprintf("cuda%d", i)
		id := cudaprobe.Identity{Process: cudaprobe.Process{PID: uint32(cfg.PID), Generation: generation}, Provenance: "linux_tracefs_uprobe", Library: cfg.Library, BuildID: image.BuildID, Version: cfg.Version, Symbol: symbol, ABI: captureABI()}
		targets = append(targets, Target{Name: name, Identity: id, Arguments: n})
		var fetch []string
		regs, stackOffset := captureRegisters()
		for j := 0; j < n; j++ {
			arg := fmt.Sprintf("+%d(%%sp)", stackOffset+(j-len(regs))*8)
			if j < len(regs) {
				arg = regs[j]
			}
			fetch = append(fetch, fmt.Sprintf("a%d=%s:u64", j, arg))
		}
		probes = append(probes, ebpfhost.Probe{Path: path, Offset: image.Offset, Name: name + "_in", FetchArgs: strings.Join(fetch, " ")}, ebpfhost.Probe{Path: path, Offset: image.Offset, Name: name + "_out", Return: true, FetchArgs: "ret=$retval:u64"})
	}
	calls, err := NewCorrelator(targets, cfg.MaxPending)
	if err != nil {
		return nil, err
	}
	backend := cfg.Backend
	if backend == "" {
		backend = "tracefs"
	}
	if cfg.UserStack && backend != "bpf" {
		return nil, fmt.Errorf("cudatracehost: user stacks require bpf backend")
	}
	if backend != "tracefs" && backend != "bpf" {
		return nil, fmt.Errorf("cudatracehost: unsupported backend %s", backend)
	}
	var trace *ebpfhost.Collector
	var group *bpfprobe.Group
	if backend == "tracefs" {
		trace, err = ebpfhost.Open(ebpfhost.Config{PID: cfg.PID, TraceRoot: cfg.TraceRoot, Probes: probes, Scheduler: cfg.Scheduler, Lifecycle: true})
	} else {
		tids, readErr := os.ReadDir(fmt.Sprintf("/proc/%d/task", cfg.PID))
		if readErr != nil {
			return nil, readErr
		}
		if len(tids)*len(probes) > 256 {
			return nil, fmt.Errorf("cudatracehost: attachment snapshot exceeds 256 probes")
		}
		var configs []bpfprobe.Config
		for _, tid := range tids {
			id, e := strconv.Atoi(tid.Name())
			if e != nil {
				return nil, e
			}
			for i, p := range probes {
				configs = append(configs, bpfprobe.Config{Path: p.Path, Offset: p.Offset, PID: id, Tag: uint64(i), Return: p.Return, Capacity: 65536, OutputPointer: outputPointer(targets[i/2].Identity.Symbol), UserStack: cfg.UserStack, CorrelationTag: uint64(i/2 + 1), TGID: uint32(cfg.PID)})
			}
		}
		group, err = bpfprobe.AttachGroup(configs)
		if err == nil {
			trace, err = ebpfhost.Open(ebpfhost.Config{PID: cfg.PID, TraceRoot: cfg.TraceRoot, Scheduler: cfg.Scheduler, Lifecycle: true})
			if err != nil {
				group.Close()
			}
		}
	}
	if err != nil {
		return nil, err
	}
	cleanup := func() {
		if trace != nil {
			trace.Close()
		}
		if group != nil {
			group.Close()
		}
	}

	after, err := ebpfhost.ProcessStartTime(cfg.PID)
	if err != nil || after != generation {
		cleanup()
		return nil, fmt.Errorf("cudatracehost: process generation changed during attachment")
	}
	boot, err := os.ReadFile("/proc/sys/kernel/random/boot_id")
	if err != nil {
		cleanup()
		return nil, err
	}
	for i, symbol := range cfg.Symbols {
		current, e := ebpfhost.ResolveImage(path, symbol)
		if e != nil || current.SHA256 != images[i].SHA256 || current.BuildID != images[i].BuildID || current.Offset != images[i].Offset {
			cleanup()
			return nil, fmt.Errorf("cudatracehost: library changed during attachment")
		}
	}
	if backend == "bpf" {
		for k, t := range calls.targets {
			t.Identity.Provenance = "linux_bpf_uprobe"
			calls.targets[k] = t
		}
	}
	m := Manifest{Backend: backend, Clock: "linux_monotonic", Epoch: strings.TrimSpace(string(boot)), Process: cudaprobe.Process{PID: uint32(cfg.PID), Generation: generation}, Images: images, Complete: false, LossStatus: "unknown", ThreadScope: "attachment_snapshot", LibraryVersionStatus: "caller_reported"}
	if backend == "bpf" {
		m.ThreadScope = "attachment_snapshot_new_threads_unknown"
		m.KernelBTFHash = group.KernelBTFHash
		m.TaskStartOffset = group.TaskStartOffset
		m.TaskExecOffset = group.TaskExecOffset
		m.TaskExecSize = group.TaskExecSize
	}
	c := &Collector{done: make(chan struct{}), held: held, trace: trace, group: group, calls: calls, process: m.Process, pid: cfg.PID, manifest: m, tagNames: tagNames}
	if group != nil {
		c.startReaders()
	}
	releaseHeld = false
	return c, nil
}

func signature(s string) (int, string, bool) {
	driver := map[string]int{"cuMemAllocAsync": 3, "cuMemFreeAsync": 2, "cuCtxCreate_v2": 3, "cuCtxGetCurrent": 1, "cuCtxSetCurrent": 1, "cuCtxDestroy_v2": 1, "cuStreamCreate": 2, "cuStreamDestroy_v2": 1, "cuLaunchKernel": 11, "cuMemAlloc_v2": 2, "cuMemFree_v2": 1, "cuMemcpyHtoD_v2": 3, "cuMemcpyDtoH_v2": 3, "cuMemcpyDtoD_v2": 3, "cuMemcpyHtoDAsync_v2": 4, "cuMemcpyDtoHAsync_v2": 4, "cuMemcpyDtoDAsync_v2": 4, "cuCtxSynchronize": 0, "cuStreamSynchronize": 1, "cuGraphLaunch": 2}
	if n, ok := driver[s]; ok {
		return n, "driver", true
	}
	rt := map[string]int{"cudaMallocAsync": 3, "cudaFreeAsync": 2, "cudaStreamCreate": 1, "cudaStreamDestroy": 1, "cudaLaunchKernel": 8, "cudaMalloc": 2, "cudaFree": 1, "cudaMemcpy": 4, "cudaMemcpyAsync": 5, "cudaDeviceSynchronize": 0, "cudaStreamSynchronize": 1, "cudaGraphLaunch": 2}
	n, ok := rt[s]
	return n, "runtime", ok
}

func outputPointer(symbol string) bool {
	switch symbol {
	case "cudaMalloc", "cuMemAlloc_v2", "cudaMallocAsync", "cuMemAllocAsync", "cuCtxCreate_v2", "cuCtxGetCurrent", "cuStreamCreate", "cudaStreamCreate":
		return true
	}
	return false
}

// Next returns the next raw event and optional completed call. Unknown transport
// or return pairing marks the capture incomplete and returns the original line.
func (c *Collector) Next(ctx context.Context) (result Result, err error) {
	if c == nil || c.done == nil {
		return Result{}, os.ErrClosed
	}
	select {
	case <-c.done:
		return Result{}, c.closedError()
	default:
	}
	defer func() {
		if err != nil {
			c.markTerminal(err)
		}
	}()
	generation, err := ebpfhost.ProcessStartTime(c.pid)
	if err != nil || generation != c.process.Generation {
		c.calls.Gap()
		return Result{}, fmt.Errorf("cudatracehost: target process lifetime ended or changed")
	}
	if c.group != nil {
		return c.nextBPF(ctx)
	}
	r, err := c.trace.Read(ctx)
	if err != nil {
		c.calls.Gap()
		return Result{}, err
	}
	result = Result{Raw: r.Line}
	generation, err = ebpfhost.ProcessStartTime(c.pid)
	if err != nil || generation != c.process.Generation {
		c.calls.Gap()
		return result, fmt.Errorf("cudatracehost: target process lifetime changed during read")
	}
	if strings.Contains(r.Line, ": sched_process_exec:") || strings.Contains(r.Line, ": sched_process_exit:") {
		c.calls.Gap()
		return result, fmt.Errorf("cudatracehost: observed target execution or exit boundary")
	}
	if strings.Contains(r.Line, ": sched_switch:") || strings.Contains(r.Line, ": sched_wakeup:") || strings.Contains(r.Line, ": sched_wakeup_new:") {
		result.Scheduler = true
		return result, nil
	}
	o, err := Parse(r.Line)
	if err != nil {
		c.calls.Gap()
		return result, err
	}
	if _, memberErr := os.Stat(fmt.Sprintf("/proc/%d/task/%d", c.pid, o.TID)); memberErr != nil {
		c.calls.Gap()
		return result, fmt.Errorf("cudatracehost: thread membership unavailable: %w", memberErr)
	}
	result.Capture, err = c.calls.Observe(o)
	if result.Capture != nil {
		result.Call = c.callInterval(o, result.Capture)
	}
	return result, err
}

// Close detaches owned probes. Missing return and pre-attachment history remain
// incomplete; Close never converts cancellation into successful completion.
func (c *Collector) Close() error {
	c.closeMu.Lock()
	defer c.closeMu.Unlock()
	c.markTerminal(os.ErrClosed)
	if c.cancel != nil {
		c.cancel()
	}
	var errs []error
	if c.cleanupErr != nil {
		errs = append(errs, c.cleanupErr)
	}
	if c.group != nil {
		errs = append(errs, c.group.Close())
	}
	if c.trace != nil {
		errs = append(errs, c.trace.Close())
	}
	if c.held != nil {
		errs = append(errs, c.held.Close())
		c.held = nil
	}
	c.wait.Wait()
	err := errors.Join(errs...)
	if c.cleanupErr == nil {
		c.cleanupErr = err
	}
	return err
}

// Manifest records attachment provenance and explicit coverage limits. Epoch
// is the kernel boot ID for the monotonic clock, not a sensor clock mapping.
type Manifest struct {
	TaskExecOffset, TaskExecSize                  uint32
	KernelBTFHash                                 string
	TaskStartOffset                               uint32
	Backend, Clock, Epoch                         string
	Process                                       cudaprobe.Process
	Images                                        []ebpfhost.Image
	Complete                                      bool
	LossStatus, ThreadScope, LibraryVersionStatus string
}

// Manifest returns a copy of immutable attachment provenance.
func (c *Collector) Manifest() Manifest {
	m := c.manifest
	m.Images = append([]ebpfhost.Image(nil), m.Images...)
	return m
}

type transportEvent struct {
	native *bpfprobe.Event
	raw    string
	err    error
}

func (c *Collector) startReaders() {
	ctx, cancel := context.WithCancel(context.Background())
	c.cancel = cancel
	c.events = make(chan transportEvent, 256)
	send := func(e transportEvent) bool {
		select {
		case c.events <- e:
			return true
		case <-ctx.Done():
			return false
		}
	}
	c.wait.Add(1)
	go func() {
		defer c.wait.Done()
		for {
			e, err := c.group.Read(ctx)
			if err != nil {
				send(transportEvent{err: err})
				return
			}
			if !send(transportEvent{native: &e}) {
				return
			}
		}
	}()
	if c.trace != nil {
		c.wait.Add(1)
		go func() {
			defer c.wait.Done()
			for {
				e, err := c.trace.Read(ctx)
				if err != nil {
					send(transportEvent{err: err})
					return
				}
				if !send(transportEvent{raw: e.Line}) {
					return
				}
			}
		}()
	}
}
func (c *Collector) nextBPF(ctx context.Context) (Result, error) {
	select {
	case <-c.done:
		return Result{}, c.closedError()
	case <-ctx.Done():
		c.calls.Gap()
		return Result{}, ctx.Err()
	case e := <-c.events:
		generation, generationErr := ebpfhost.ProcessStartTime(c.pid)
		if generationErr != nil || generation != c.process.Generation {
			c.calls.Gap()
			return Result{Raw: e.raw}, fmt.Errorf("cudatracehost: process lifetime changed during BPF read")
		}
		if e.err != nil {
			c.calls.Gap()
			return Result{}, e.err
		}
		if e.native == nil {
			if strings.Contains(e.raw, ": sched_process_exec:") || strings.Contains(e.raw, ": sched_process_exit:") {
				c.calls.Gap()
				return Result{Raw: e.raw}, fmt.Errorf("cudatracehost: observed execution or exit boundary")
			}
			return Result{Raw: e.raw, Scheduler: true}, nil
		}
		c.delivered.Add(1)
		data, err := json.Marshal(e.native)
		r := Result{Raw: string(data), Native: e.native}
		if err != nil {
			return r, err
		}
		_, dropped, err := c.group.Counts()
		if err != nil {
			return r, err
		}
		if dropped != 0 {
			c.calls.Gap()
			return r, fmt.Errorf("cudatracehost: BPF ring dropped %d observations", dropped)
		}
		if err := c.validateTaskEvent(*e.native); err != nil {
			c.calls.Gap()
			return r, err
		}
		tid := uint32(e.native.PIDTGID)
		name, ok := c.tagNames[e.native.Tag]
		if !ok {
			c.calls.Gap()
			return r, fmt.Errorf("cudatracehost: unknown BPF tag")
		}
		if strings.HasSuffix(name, "_out") != e.native.ReturnKnown || (e.native.ReturnKnown && e.native.ArgumentsMask != 0) {
			c.calls.Gap()
			return r, fmt.Errorf("cudatracehost: BPF tag/return flags contradict")
		}
		o := Observation{ExecID: e.native.ExecID, ThreadGenerationKnown: e.native.ThreadGenerationKnown, ThreadGeneration: e.native.ThreadGeneration, Raw: r.Raw, TID: tid, Timestamp: e.native.Timestamp, Name: name, Args: e.native.Arguments, Return: int32(e.native.ReturnValue), ReturnKnown: e.native.ReturnKnown}
		for i := range o.ArgsKnown {
			o.ArgsKnown[i] = e.native.ArgumentsMask&(1<<i) != 0
		}
		r.Capture, err = c.calls.Observe(o)
		if r.Capture != nil && e.native.OutputKnown {
			binary.LittleEndian.PutUint16(r.Capture.Data[2:], 7)
			binary.LittleEndian.PutUint64(r.Capture.Data[120:], e.native.Output)
		}
		if r.Capture != nil {
			r.Call = c.callInterval(o, r.Capture)
		}
		return r, err
	}
}

// CallInterval is observed host API entry-to-return timing. It is not device
// execution time and does not imply application request identity.
type CallInterval struct {
	ExecID                          uint64
	ThreadGenerationKnown           bool
	ThreadGeneration                uint64
	PairingStatus                   string
	TID                             uint32
	EntryTimestamp, ReturnTimestamp uint64
	Clock, Epoch                    string
	Process                         cudaprobe.Process
}

func (c *Collector) callInterval(o Observation, cap *cudaprobe.Capture) *CallInterval {
	status := "unverified_thread_lifetime"
	if o.ThreadGenerationKnown {
		status = "verified_thread_lifetime"
	}
	return &CallInterval{ExecID: o.ExecID, ThreadGenerationKnown: o.ThreadGenerationKnown, ThreadGeneration: o.ThreadGeneration, PairingStatus: status, TID: o.TID, EntryTimestamp: binary.LittleEndian.Uint64(cap.Data[16:]), ReturnTimestamp: o.Timestamp, Clock: c.manifest.Clock, Epoch: c.manifest.Epoch, Process: c.process}
}

func (c *Collector) markTerminal(err error) {
	c.stateMu.Lock()
	if c.terminalErr == nil {
		c.terminalErr = err
	}
	c.stateMu.Unlock()
	c.stopOnce.Do(func() {
		if c.done != nil {
			close(c.done)
		}
		if c.cancel != nil {
			c.cancel()
		}
	})
}
func (c *Collector) closedError() error {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	if c.terminalErr != nil {
		return c.terminalErr
	}
	return os.ErrClosed
}

func sameDevice(dev uint64, text string) bool {
	a, b, ok := strings.Cut(text, ":")
	if !ok {
		return false
	}
	major, e := strconv.ParseUint(a, 16, 32)
	if e != nil {
		return false
	}
	minor, e := strconv.ParseUint(b, 16, 32)
	if e != nil {
		return false
	}
	return major == ((dev>>8)&0xfff|(dev>>32)&0xfffff000) && minor == (dev&0xff|(dev>>12)&0xffffff00)
}

// StopStats reports known BPF ring counts after disabling attachments. The
// capture remains incomplete: pre-attachment history and tracefs loss are
// unknown. Delivered counts include records returned before caller writes.
type StopStats struct {
	Submitted, Dropped, Delivered, Undelivered uint64
	RingStatus                                 string
	Output                                     bpfprobe.OutputFailures
}

func (c *Collector) StopAndStats() (StopStats, error) {
	s := StopStats{Delivered: c.delivered.Load(), RingStatus: "unavailable"}
	if c.group == nil {
		return s, nil
	}
	if err := c.group.Disable(); err != nil {
		return s, err
	}
	submitted, dropped, err := c.group.Counts()
	if err != nil {
		return s, err
	}
	s.Submitted = submitted
	s.Dropped = dropped
	s.RingStatus = "measured"
	if s.Delivered > submitted {
		return s, fmt.Errorf("cudatracehost: delivered count exceeds kernel submissions")
	}
	s.Undelivered = submitted - s.Delivered
	s.Output, err = c.group.OutputFailures()
	return s, err
}

// validateTaskEvent uses identity measured by the kernel at the event, so an
// already-exited task can still be decoded from the ordered ring. Current /proc
// membership cannot prove historical identity. Process start-time checks remain
// separate; this does not establish complete coverage of newly created tasks.
func (c *Collector) validateTaskEvent(e bpfprobe.Event) error {
	if uint32(e.PIDTGID>>32) != c.process.PID {
		return fmt.Errorf("cudatracehost: BPF process identity mismatch")
	}
	if !e.ThreadGenerationKnown {
		return fmt.Errorf("cudatracehost: BPF thread generation unavailable")
	}
	if !e.ExecIDKnown {
		return fmt.Errorf("cudatracehost: BPF exec identity unavailable")
	}
	if c.execKnown && c.execID != e.ExecID {
		return fmt.Errorf("cudatracehost: target exec epoch changed")
	}
	c.execKnown = true
	c.execID = e.ExecID
	return nil
}

func captureABI() string {
	if runtime.GOARCH == "arm64" {
		return "linux-arm64-aapcs64-v1"
	}
	return "linux-amd64-sysv-v1"
}
func captureRegisters() ([]string, int) {
	if runtime.GOARCH == "arm64" {
		return []string{"%x0", "%x1", "%x2", "%x3", "%x4", "%x5", "%x6", "%x7"}, 0
	}
	return []string{"%di", "%si", "%dx", "%cx", "%r8", "%r9"}, 8
}

func validateImageABI(class elf.Class, data elf.Data, machine elf.Machine, arch string) error {
	want := elf.EM_X86_64
	if arch == "arm64" {
		want = elf.EM_AARCH64
	}
	if class != elf.ELFCLASS64 || data != elf.ELFDATA2LSB || machine != want {
		return fmt.Errorf("cudatracehost: ELF architecture does not match %s capture ABI", arch)
	}
	return nil
}
