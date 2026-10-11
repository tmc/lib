package cudashadow

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"strconv"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// --- FIFO Eviction Policy ---

type fifoPolicy struct {
	info  PolicyInfo
	limit int
	mu    sync.Mutex
	// Track chunk activation order: earliest activated is first in list
	chunks []uint64
}

// NewFIFOPolicy constructs a FIFO UVM eviction policy modeled after gpu_ext eviction_fifo.bpf.c.
func NewFIFOPolicy(info PolicyInfo) (Policy, error) {
	limit, err := policyLimit(info)
	if err != nil {
		return nil, err
	}
	if info.Domain != DomainEviction {
		return nil, fmt.Errorf("cudashadow: fifo policy requires domain %s, got %s: %w",
			DomainEviction, info.Domain, cudaerrors.ErrUnsupported)
	}
	return &fifoPolicy{
		info:   clonePolicyInfo(info),
		limit:  limit,
		chunks: make([]uint64, 0),
	}, nil
}

func (p *fifoPolicy) Info() PolicyInfo { return clonePolicyInfo(p.info) }

func (p *fifoPolicy) Evaluate(ctx context.Context, in Input) (Decision, error) {
	if err := policyInput(ctx, in, p.info.Domain); err != nil {
		return Decision{}, err
	}
	if in.Eviction == nil {
		return Decision{}, fmt.Errorf("cudashadow fifo: missing eviction payload")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	switch in.Provenance.Hook {
	case HookBlockActivate:
		for _, chunk := range p.chunks {
			if chunk == in.Eviction.ChunkAddr {
				return Decision{}, errors.New("cudashadow fifo: duplicate activation")
			}
		}
		if len(p.chunks) >= p.limit {
			return Decision{}, ErrCapacity
		}
		// When activated, record chunk at tail
		p.chunks = append(p.chunks, in.Eviction.ChunkAddr)
		return Decision{
			Type:   DecisionBypass,
			Reason: "fifo activate: chunk tracked in fifo list, default lru positioning",
		}, nil

	case HookBlockAccess:
		found := false
		for _, chunk := range p.chunks {
			if chunk == in.Eviction.ChunkAddr {
				found = true
				break
			}
		}
		if !found {
			return Decision{}, errors.New("cudashadow fifo: access before activation")
		}
		// FIFO does not promote accessed chunks to tail
		return Decision{
			Type:   DecisionBypass,
			Reason: "fifo access: retain insertion order without promoting accessed chunk",
		}, nil

	case HookEvictPrepare:
		if len(p.chunks) == 0 {
			return Decision{
				Type:   DecisionSkip,
				Reason: "fifo evict prepare: empty chunk candidate pool",
			}, nil
		}
		// Candidate is oldest chunk (head of slice)
		target := p.chunks[0]
		p.chunks = p.chunks[1:]
		return Decision{
			Type:       DecisionEvict,
			Parameters: map[string]int64{"target_chunk": int64(target)},
			Reason:     fmt.Sprintf("fifo evict prepare: proposed oldest chunk 0x%x for eviction", target),
		}, nil

	default:
		return Decision{}, fmt.Errorf("cudashadow fifo: unsupported hook %q: %w", in.Provenance.Hook, cudaerrors.ErrUnsupported)
	}
}

// --- LFU Eviction Policy ---

type lfuPolicy struct {
	info        PolicyInfo
	limit       int
	threshold   uint64
	priorityPID int
	mu          sync.Mutex
	accesses    map[uint64]uint64
}

// NewLFUPolicy constructs a Least-Frequently-Used eviction policy with PID priority quota.
func NewLFUPolicy(info PolicyInfo) (Policy, error) {
	limit, err := policyLimit(info)
	if err != nil {
		return nil, err
	}
	if info.Domain != DomainEviction {
		return nil, fmt.Errorf("cudashadow: lfu policy requires domain %s, got %s: %w",
			DomainEviction, info.Domain, cudaerrors.ErrUnsupported)
	}
	threshold := uint64(5)
	if valStr, ok := info.Parameters["threshold"]; ok {
		v, err := strconv.ParseUint(valStr, 10, 63)
		if err != nil || v == 0 {
			return nil, errors.New("cudashadow: invalid threshold")
		}
		threshold = v
	}
	priorityPID := 0
	if pidStr, ok := info.Parameters["priority_pid"]; ok {
		pid, err := strconv.Atoi(pidStr)
		if err != nil || pid <= 0 {
			return nil, errors.New("cudashadow: invalid priority pid")
		}
		priorityPID = pid
	}
	return &lfuPolicy{
		info:        clonePolicyInfo(info),
		limit:       limit,
		threshold:   threshold,
		priorityPID: priorityPID,
		accesses:    make(map[uint64]uint64),
	}, nil
}

func (p *lfuPolicy) Info() PolicyInfo { return clonePolicyInfo(p.info) }

func (p *lfuPolicy) Evaluate(ctx context.Context, in Input) (Decision, error) {
	if err := policyInput(ctx, in, p.info.Domain); err != nil {
		return Decision{}, err
	}
	if in.Eviction == nil {
		return Decision{}, fmt.Errorf("cudashadow lfu: missing eviction payload")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	chunk := in.Eviction.ChunkAddr

	switch in.Provenance.Hook {
	case HookBlockActivate:
		if _, ok := p.accesses[chunk]; ok {
			return Decision{}, errors.New("cudashadow lfu: duplicate activation")
		}
		if len(p.accesses) >= p.limit {
			return Decision{}, ErrCapacity
		}
		p.accesses[chunk] = 1
		return Decision{
			Type:   DecisionBypass,
			Reason: "lfu activate: initialized access count to 1",
		}, nil

	case HookBlockAccess:
		if _, ok := p.accesses[chunk]; !ok {
			return Decision{}, errors.New("cudashadow lfu: access before activation")
		}
		if p.accesses[chunk] == math.MaxInt64 {
			return Decision{}, errors.New("cudashadow lfu: access count overflow")
		}
		p.accesses[chunk]++
		cnt := p.accesses[chunk]
		if p.priorityPID > 0 && in.Eviction.PID == p.priorityPID || cnt >= p.threshold {
			return Decision{
				Type:       DecisionProtect,
				Parameters: map[string]int64{"access_count": int64(cnt)},
				Reason:     fmt.Sprintf("lfu access: chunk 0x%x protected (count=%d >= threshold=%d or priority pid)", chunk, cnt, p.threshold),
			}, nil
		}
		return Decision{
			Type:       DecisionBypass,
			Parameters: map[string]int64{"access_count": int64(cnt)},
			Reason:     fmt.Sprintf("lfu access: chunk 0x%x below frequency threshold (%d < %d)", chunk, cnt, p.threshold),
		}, nil

	case HookEvictPrepare:
		if len(p.accesses) == 0 {
			return Decision{
				Type:   DecisionSkip,
				Reason: "lfu evict prepare: empty candidate pool",
			}, nil
		}
		var minChunk uint64
		var minCount uint64 = ^uint64(0)
		for c, cnt := range p.accesses {
			if cnt < minCount || cnt == minCount && c < minChunk {
				minCount = cnt
				minChunk = c
			}
		}
		delete(p.accesses, minChunk)
		return Decision{
			Type:       DecisionEvict,
			Parameters: map[string]int64{"target_chunk": int64(minChunk), "frequency": int64(minCount)},
			Reason:     fmt.Sprintf("lfu evict prepare: chunk 0x%x has lowest frequency %d", minChunk, minCount),
		}, nil

	default:
		return Decision{}, fmt.Errorf("cudashadow lfu: unsupported hook %q: %w", in.Provenance.Hook, cudaerrors.ErrUnsupported)
	}
}

// --- MRU Eviction Policy ---

type mruPolicy struct {
	info  PolicyInfo
	limit int
	mu    sync.Mutex
	// Track chunk access order: latest accessed is last in slice
	stack []uint64
}

// NewMRUPolicy constructs a Most-Recently-Used eviction policy.
func NewMRUPolicy(info PolicyInfo) (Policy, error) {
	limit, err := policyLimit(info)
	if err != nil {
		return nil, err
	}
	if info.Domain != DomainEviction {
		return nil, fmt.Errorf("cudashadow: mru policy requires domain %s, got %s: %w",
			DomainEviction, info.Domain, cudaerrors.ErrUnsupported)
	}
	return &mruPolicy{
		info:  clonePolicyInfo(info),
		limit: limit,
		stack: make([]uint64, 0),
	}, nil
}

func (p *mruPolicy) Info() PolicyInfo { return clonePolicyInfo(p.info) }

func (p *mruPolicy) Evaluate(ctx context.Context, in Input) (Decision, error) {
	if err := policyInput(ctx, in, p.info.Domain); err != nil {
		return Decision{}, err
	}
	if in.Eviction == nil {
		return Decision{}, fmt.Errorf("cudashadow mru: missing eviction payload")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	chunk := in.Eviction.ChunkAddr

	switch in.Provenance.Hook {
	case HookBlockActivate, HookBlockAccess:
		found := false
		for _, c := range p.stack {
			if c == chunk {
				found = true
				break
			}
		}
		if in.Provenance.Hook == HookBlockAccess && !found {
			return Decision{}, errors.New("cudashadow mru: access before activation")
		}
		if in.Provenance.Hook == HookBlockActivate && found {
			return Decision{}, errors.New("cudashadow mru: duplicate activation")
		}
		if !found && len(p.stack) >= p.limit {
			return Decision{}, ErrCapacity
		}
		// Remove existing occurrence if present
		newStack := make([]uint64, 0, len(p.stack)+1)
		for _, c := range p.stack {
			if c != chunk {
				newStack = append(newStack, c)
			}
		}
		newStack = append(newStack, chunk)
		p.stack = newStack
		return Decision{
			Type:   DecisionBypass,
			Reason: fmt.Sprintf("mru update: placed chunk 0x%x at top of mru stack", chunk),
		}, nil

	case HookEvictPrepare:
		if len(p.stack) == 0 {
			return Decision{
				Type:   DecisionSkip,
				Reason: "mru evict prepare: empty candidate pool",
			}, nil
		}
		// Most recently used is at top of stack
		target := p.stack[len(p.stack)-1]
		p.stack = p.stack[:len(p.stack)-1]
		return Decision{
			Type:       DecisionEvict,
			Parameters: map[string]int64{"target_chunk": int64(target)},
			Reason:     fmt.Sprintf("mru evict prepare: proposed most-recently-used chunk 0x%x for eviction", target),
		}, nil

	default:
		return Decision{}, fmt.Errorf("cudashadow mru: unsupported hook %q: %w", in.Provenance.Hook, cudaerrors.ErrUnsupported)
	}
}

// --- Adaptive Prefetch Policy ---

type adaptivePrefetchPolicy struct {
	info       PolicyInfo
	limit      int
	mu         sync.Mutex
	lastFault  map[int]uint64
	lastStride map[int]int64
}

// NewAdaptivePrefetchPolicy constructs a sequential/stride prefetch policy modeled after gpu_ext prefetch.
func NewAdaptivePrefetchPolicy(info PolicyInfo) (Policy, error) {
	limit, err := policyLimit(info)
	if err != nil {
		return nil, err
	}
	if info.Domain != DomainPrefetch {
		return nil, fmt.Errorf("cudashadow: adaptive prefetch requires domain %s, got %s: %w",
			DomainPrefetch, info.Domain, cudaerrors.ErrUnsupported)
	}
	return &adaptivePrefetchPolicy{
		info:       clonePolicyInfo(info),
		limit:      limit,
		lastFault:  make(map[int]uint64),
		lastStride: make(map[int]int64),
	}, nil
}

func (p *adaptivePrefetchPolicy) Info() PolicyInfo { return clonePolicyInfo(p.info) }

func (p *adaptivePrefetchPolicy) Evaluate(ctx context.Context, in Input) (Decision, error) {
	if err := policyInput(ctx, in, p.info.Domain); err != nil {
		return Decision{}, err
	}
	if in.Prefetch == nil {
		return Decision{}, fmt.Errorf("cudashadow prefetch: missing prefetch payload")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	pid := in.Prefetch.PID
	addr := in.Prefetch.FaultAddr
	prevFault, hasPrev := p.lastFault[pid]
	if !hasPrev && len(p.lastFault) >= p.limit {
		return Decision{}, ErrCapacity
	}
	p.lastFault[pid] = addr

	if !hasPrev {
		return Decision{
			Type:   DecisionThrottle,
			Reason: "prefetch cold start: first observed fault for pid",
		}, nil
	}

	stride := int64(addr) - int64(prevFault)
	prevStride := p.lastStride[pid]
	p.lastStride[pid] = stride

	// Constant sequential forward stride detection
	if stride > 0 && stride == prevStride {
		const pageSize = 4096
		pagesToPrefetch := uint16(8)
		outer := in.Prefetch.OuterPage + pagesToPrefetch
		if outer > 512 {
			outer = 512
		}
		return Decision{
			Type: DecisionPrefetch,
			Parameters: map[string]int64{
				"first_page":   int64(in.Prefetch.OuterPage),
				"outer_page":   int64(outer),
				"stride_bytes": stride,
			},
			Reason: fmt.Sprintf("adaptive prefetch: detected steady stride %d bytes, prefetching pages [%d, %d)",
				stride, in.Prefetch.OuterPage, outer),
		}, nil
	}

	return Decision{
		Type:   DecisionThrottle,
		Reason: fmt.Sprintf("irregular access stride (%d vs prev %d): prefetch suppressed", stride, prevStride),
	}, nil
}

// --- Priority Timeslice Scheduling Policy ---

type priorityTimeslicePolicy struct {
	info PolicyInfo
}

// NewPriorityTimeslicePolicy constructs a timeslice scheduling policy based on process priority.
func NewPriorityTimeslicePolicy(info PolicyInfo) (Policy, error) {
	if _, err := policyLimit(info); err != nil {
		return nil, err
	}
	if info.Domain != DomainScheduling {
		return nil, fmt.Errorf("cudashadow: scheduling policy requires domain %s, got %s: %w",
			DomainScheduling, info.Domain, cudaerrors.ErrUnsupported)
	}
	return &priorityTimeslicePolicy{info: clonePolicyInfo(info)}, nil
}

func (p *priorityTimeslicePolicy) Info() PolicyInfo { return clonePolicyInfo(p.info) }

func (p *priorityTimeslicePolicy) Evaluate(ctx context.Context, in Input) (Decision, error) {
	if err := policyInput(ctx, in, p.info.Domain); err != nil {
		return Decision{}, err
	}
	if in.Scheduling == nil {
		return Decision{}, fmt.Errorf("cudashadow scheduling: missing scheduling payload")
	}

	baseTimesliceUS := uint64(1000)
	priority := in.Scheduling.Priority
	if priority < 1 {
		priority = 1
	}
	if priority > 10 {
		priority = 10
	}

	// Quantum scales with priority and queue length
	quantumUS := baseTimesliceUS * uint64(priority)
	if in.Scheduling.QueueLength > 4 {
		// Under high queue contention, reduce quantum by half to improve latency
		quantumUS /= 2
	}

	return Decision{
		Type: DecisionTimeslice,
		Parameters: map[string]int64{
			"channel_id":   int64(in.Scheduling.ChannelID),
			"timeslice_us": int64(quantumUS),
		},
		Reason: fmt.Sprintf("timeslice calculated: priority %d, queue length %d -> %d us",
			priority, in.Scheduling.QueueLength, quantumUS),
	}, nil
}

// --- PTX Shadow Policy ---

type ptxShadowPolicy struct {
	info PolicyInfo
}

// NewPTXShadowPolicy constructs an observation-only policy for inspecting PTX kernels.
// It verifies that zero PTX mutations, code injections, or rewrites are emitted.
func NewPTXShadowPolicy(info PolicyInfo) (Policy, error) {
	if _, err := policyLimit(info); err != nil {
		return nil, err
	}
	if info.Domain != DomainPTXInstrument {
		return nil, fmt.Errorf("cudashadow: ptx policy requires domain %s, got %s: %w",
			DomainPTXInstrument, info.Domain, cudaerrors.ErrUnsupported)
	}
	return &ptxShadowPolicy{info: clonePolicyInfo(info)}, nil
}

func (p *ptxShadowPolicy) Info() PolicyInfo { return clonePolicyInfo(p.info) }

func (p *ptxShadowPolicy) Evaluate(ctx context.Context, in Input) (Decision, error) {
	if err := policyInput(ctx, in, p.info.Domain); err != nil {
		return Decision{}, err
	}
	if in.PTX == nil {
		return Decision{}, fmt.Errorf("cudashadow ptx: missing ptx payload")
	}

	// Shadow observation only: plan entry and exit telemetry without mutating PTX
	return Decision{
		Type: DecisionSkip,
		Parameters: map[string]int64{
			"observed_size": int64(in.PTX.CodeSize),
			"offset":        int64(in.PTX.FunctionOffset),
		},
		Reason: fmt.Sprintf("ptx shadow observation: kernel %q (arch %s) recorded; mutation skipped in shadow mode",
			in.PTX.KernelName, in.PTX.TargetArch),
	}, nil
}

// ErrCapacity indicates that policy state reached its configured bound.
var ErrCapacity = errors.New("cudashadow: policy capacity exhausted")

func policyLimit(info PolicyInfo) (int, error) {
	if err := info.Validate(); err != nil {
		return 0, err
	}
	limit := 4096
	if text, ok := info.Parameters["max_entries"]; ok {
		n, err := strconv.Atoi(text)
		if err != nil || n <= 0 || n > 1<<20 {
			return 0, errors.New("cudashadow: invalid max_entries")
		}
		limit = n
	}
	return limit, nil
}

func clonePolicyInfo(info PolicyInfo) PolicyInfo {
	info.Parameters = maps.Clone(info.Parameters)
	return info
}

func policyInput(ctx context.Context, in Input, domain Domain) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := in.Validate(); err != nil {
		return err
	}
	if in.Domain != domain {
		return fmt.Errorf("cudashadow: %w: domain mismatch", cudaerrors.ErrUnsupported)
	}
	return nil
}

// RegisteredCorpusPolicies returns a deterministic catalog of standard policy configurations.
func RegisteredCorpusPolicies() []PolicyInfo {
	return []PolicyInfo{
		{
			Name:    "gpu_ext-eviction-fifo",
			Version: "1.0.0",
			Domain:  DomainEviction,
			Seed:    42,
		},
		{
			Name:    "gpu_ext-eviction-lfu",
			Version: "1.0.0",
			Domain:  DomainEviction,
			Parameters: map[string]string{
				"threshold":    "3",
				"priority_pid": "1001",
			},
			Seed: 43,
		},
		{
			Name:    "gpu_ext-eviction-mru",
			Version: "1.0.0",
			Domain:  DomainEviction,
			Seed:    44,
		},
		{
			Name:    "gpu_ext-prefetch-adaptive",
			Version: "1.0.0",
			Domain:  DomainPrefetch,
			Seed:    45,
		},
		{
			Name:    "gpu_ext-sched-priority",
			Version: "1.0.0",
			Domain:  DomainScheduling,
			Seed:    46,
		},
		{
			Name:    "bpftime-ptx-shadow",
			Version: "1.0.0",
			Domain:  DomainPTXInstrument,
			Seed:    47,
		},
	}
}
