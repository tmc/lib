package bpfprobe

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/tmc/lib/ebpfhost/bpfload"
)

// Config identifies one uprobe file offset on a single Linux task (thread).
// Callers must explicitly attach each thread they intend to observe. PID -1
// and system-wide capture are refused. Offset is an ELF file offset, not a
// runtime virtual address. Return requests a uretprobe.
type Config struct {
	// UserStack captures up to eight user instruction addresses in a bounded
	// 1024-entry stack map. Symbols and Go goroutine identities are not inferred.
	UserStack bool
	Path      string
	Offset    uint64
	PID       int
	Tag       uint64
	Return    bool
	Capacity  int
	// OutputPointer captures an eight-byte value through argument zero on
	// successful (low 32-bit status zero) return. Entry and return probes must
	// share a nonzero CorrelationTag. Per-task nesting is bounded to eight;
	// maps are bounded to 4096 bindings and state failures remain explicit.
	OutputPointer  bool
	CorrelationTag uint64
	// InheritThreads is unsupported and rejected before attachment. Linux perf
	// reinitializes inherited uprobes in the child, which can prevent thread
	// creation. Attach existing tasks explicitly; future tasks remain unknown.
	InheritThreads       bool
	TGID                 uint32
	generationOffset     uint32
	execOffset, execSize uint32
}

// OutputFailures reports missing entry state, nesting overflow, user-memory
// read refusal, and map operation failures. Counts include failed observations
// even when the event ring was full; they are not deduplicated request counts.
type OutputFailures struct{ Missing, Overflow, Unreadable, MapFailure, GenerationUnknown, ExecUnknown, StackFailure uint64 }

func (p *Probe) OutputFailures() (OutputFailures, error) {
	if p == nil {
		return OutputFailures{}, errors.New("bpf probe: nil probe")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return OutputFailures{}, errors.New("bpf probe: closed")
	}
	var values [7]uint64
	key := make([]byte, 4)
	value := make([]byte, 8)
	for i := range values {
		binary.LittleEndian.PutUint32(key, uint32(i+2))
		if err := p.counts.Lookup(key, value); err != nil {
			return OutputFailures{}, err
		}
		values[i] = binary.LittleEndian.Uint64(value)
	}
	return OutputFailures{Missing: values[0], Overflow: values[1], Unreadable: values[2], MapFailure: values[3], GenerationUnknown: values[4], ExecUnknown: values[5], StackFailure: values[6]}, nil
}

// Event records host uprobe register observations, not device execution.
// Timestamp is kernel monotonic nanoseconds; PIDTGID contains tgid in the
// upper 32 bits and tid in the lower 32 bits. On amd64 Arguments are the first
// six SysV integer argument registers followed by five stack slots. On arm64
// they are x0..x7 followed by three stack slots under AAPCS64.
// ArgumentsMask marks successfully acquired slots. Return probes have no
// arguments; ReturnValue alone is the return register observation.
type Event struct {
	UserStackRequested    bool
	StackID               int64
	Stack                 []uint64
	StackUnavailable      string
	ExecID                uint64
	ExecIDKnown           bool
	ThreadGeneration      uint64
	ThreadGenerationKnown bool
	OutputKnown           bool
	Output                uint64
	// OutputError bits mark missing state (1), depth overflow (2), unreadable
	// user memory (4), and map operation failure (8).
	OutputError                                   uint32
	ReturnKnown                                   bool
	ArgumentsMask                                 uint16
	Timestamp, PIDTGID, Tag                       uint64
	Arguments                                     [11]uint64
	ReturnValue, StackPointer, InstructionPointer uint64
}

// DecodeEvent decodes the fixed little-endian normalized uprobe wire record.
func DecodeEvent(data []byte) (Event, error) {
	if len(data) != 168 {
		return Event{}, errors.New("bpf probe: invalid event size")
	}
	var words [21]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(data[i*8:])
	}
	if words[17]&^uint64(0x3ff07ff) != 0 {
		return Event{}, errors.New("bpf probe: invalid argument flags")
	}
	e := Event{Timestamp: words[0], PIDTGID: words[1], Tag: words[2], ReturnValue: words[14], StackPointer: words[15], InstructionPointer: words[16], ArgumentsMask: uint16(words[17]), ReturnKnown: words[17]&(1<<16) != 0}
	if e.ReturnKnown && e.ArgumentsMask != 0 {
		return Event{}, errors.New("bpf probe: return record claims arguments")
	}
	e.UserStackRequested = words[17]&(1<<24) != 0
	e.StackID = int64(words[20])
	if words[17]&(1<<25) != 0 && (!e.UserStackRequested || e.StackID >= 0) {
		return Event{}, errors.New("bpf probe: contradictory stack flags")
	}
	e.ExecID = words[19]
	e.ExecIDKnown = words[17]&(1<<23) == 0
	e.ThreadGeneration = words[18]
	e.ThreadGenerationKnown = words[17]&(1<<22) == 0 && e.ThreadGeneration != 0
	e.OutputKnown = words[17]&(1<<17) != 0
	e.OutputError = uint32(words[17]>>18) & 15
	if e.OutputKnown {
		if !e.ReturnKnown || e.OutputError != 0 {
			return Event{}, errors.New("bpf probe: contradictory output flags")
		}
		e.Output = words[3]
	}
	if !e.ReturnKnown {
		copy(e.Arguments[:], words[3:14])
	}
	return e, nil
}

// Probe owns a perf-event attachment, loaded BPF program, and ring buffer.
// Closing disables capture and unmaps buffers; Close is idempotent.
type Probe struct {
	closeErr                     error
	stacks                       *bpfload.Object
	TaskExecOffset, TaskExecSize uint32
	KernelBTFHash                string
	TaskStartOffset              uint32
	mu                           sync.Mutex
	ring                         *Ring
	perf                         *os.File
	program, events, counts      *bpfload.Object
	extras                       []*bpfload.Object
	closed                       bool
}

// Read waits for and decodes the next record. It may be canceled by ctx or
// Close. It does not infer device timestamps or request identities.
func (p *Probe) Read(ctx context.Context) (Event, error) {
	if p == nil {
		return Event{}, errors.New("bpf probe: nil probe")
	}
	if p.ring == nil {
		return Event{}, errors.New("bpf probe: uninitialized probe")
	}
	raw, err := p.ring.Read(ctx)
	if err != nil {
		return Event{}, err
	}
	e, err := DecodeEvent(raw)
	if err != nil {
		return Event{}, err
	}
	if e.UserStackRequested {
		if e.StackID < 0 {
			e.StackUnavailable = fmt.Sprintf("kernel_stack_error:%d", e.StackID)
		} else {
			p.mu.Lock()
			if p.closed || p.stacks == nil {
				e.StackUnavailable = "stack_map_closed"
			} else {
				key := make([]byte, 4)
				binary.LittleEndian.PutUint32(key, uint32(e.StackID))
				value := make([]byte, 64)
				if err := p.stacks.Lookup(key, value); err != nil {
					e.StackUnavailable = fmt.Sprintf("stack_lookup:%v", err)
				} else {
					for i := 0; i < 8; i++ {
						address := binary.LittleEndian.Uint64(value[i*8:])
						if address == 0 {
							break
						}
						e.Stack = append(e.Stack, address)
					}
					if len(e.Stack) == 0 {
						e.StackUnavailable = "empty_stack"
					}
				}
			}
			p.mu.Unlock()
		}
	}
	return e, nil
}

// Counts reports successful ring submissions and failed submissions. Counts
// are monotonic kernel counters; callers should sample after disabling capture
// when they need a final completeness decision.
func (p *Probe) Counts() (submitted, dropped uint64, err error) {
	if p == nil {
		return 0, 0, errors.New("bpf probe: nil probe")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return 0, 0, errors.New("bpf probe: closed")
	}
	key := make([]byte, 4)
	value := make([]byte, 8)
	if err = p.counts.Lookup(key, value); err != nil {
		return
	}
	submitted = binary.LittleEndian.Uint64(value)
	binary.LittleEndian.PutUint32(key, 1)
	if err = p.counts.Lookup(key, value); err != nil {
		return
	}
	dropped = binary.LittleEndian.Uint64(value)
	return
}

func (p *Probe) Close() error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return p.closeErr
	}
	p.closed = true
	var errs []error
	if p.perf != nil {
		errs = append(errs, p.perf.Close())
	}
	if p.ring != nil {
		errs = append(errs, p.ring.Close())
	}
	errs = append(errs, p.program.Close(), p.events.Close(), p.counts.Close(), p.stacks.Close())
	for _, o := range p.extras {
		errs = append(errs, o.Close())
	}
	p.closeErr = errors.Join(errs...)
	return p.closeErr
}

func validateConfig(c Config) (Config, error) {
	if c.InheritThreads {
		return c, errors.New("bpf probe: perf uprobe inheritance unsupported: child initialization can fail")
	}
	if c.OutputPointer && c.CorrelationTag == 0 {
		return c, errors.New("bpf probe: output correlation tag required")
	}
	if c.PID <= 0 {
		return c, errors.New("bpf probe: positive task id required")
	}
	if c.Path == "" {
		return c, errors.New("bpf probe: missing path")
	}
	info, err := os.Stat(c.Path)
	if err != nil {
		return c, fmt.Errorf("bpf probe: stat binary: %w", err)
	}
	if !info.Mode().IsRegular() || c.Offset >= uint64(info.Size()) {
		return c, errors.New("bpf probe: invalid binary file offset")
	}
	if c.Capacity == 0 {
		c.Capacity = 1 << 20
	}
	return c, nil
}
