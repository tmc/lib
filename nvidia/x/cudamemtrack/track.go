package cudamemtrack

import (
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// Allocation records the lifecycle metadata and stack trace of a tracked GPU allocation.
type Allocation struct {
	Address     uintptr
	Size        uint64
	Tag         string
	AllocatedAt time.Time
	Callers     []uintptr
}

// StackFrames formats the recorded call stack frames as readable function and file locations.
func (a *Allocation) StackFrames() []string {
	if len(a.Callers) == 0 {
		return nil
	}
	frames := runtime.CallersFrames(a.Callers)
	var out []string
	for {
		frame, more := frames.Next()
		// Omit runtime internal frames
		if !strings.Contains(frame.Function, "runtime.") && !strings.Contains(frame.Function, "cudamemtrack.") {
			out = append(out, fmt.Sprintf("%s (%s:%d)", frame.Function, frame.File, frame.Line))
		}
		if !more {
			break
		}
	}
	return out
}

// Summary provides aggregate allocation and memory footprint statistics.
type Summary struct {
	ActiveAllocations   int
	ActiveBytes         uint64
	TotalAllocations    uint64
	TotalAllocatedBytes uint64
	TotalFreedBytes     uint64
	PeakBytes           uint64
}

// Tracker monitors GPU allocations and detects memory leaks.
type Tracker struct {
	mu           sync.Mutex
	allocations  map[uintptr]*Allocation
	recordStacks bool
	summary      Summary
}

// Option configures a Tracker.
type Option func(*Tracker)

// WithRecordStacks enables call stack capture for each allocation.
func WithRecordStacks(record bool) Option {
	return func(t *Tracker) {
		t.recordStacks = record
	}
}

// New creates a new GPU memory allocation tracker.
func New(opts ...Option) *Tracker {
	t := &Tracker{
		allocations:  make(map[uintptr]*Allocation),
		recordStacks: true,
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Alloc records a newly allocated device memory block at addr with the given byte
// size and tag. It returns an error, having recorded nothing, when addr or size is
// zero, or when addr is already live in the tracker.
//
// A live address cannot be handed out by a second allocation, so a repeat means a
// Free went unrecorded. Alloc refuses it rather than silently retiring the first
// block: retiring it would drop that block from Live, clear it from CheckLeaks, and
// add its bytes to Summary.TotalFreedBytes, reporting a leak as freed memory.
func (t *Tracker) Alloc(addr uintptr, size uint64, tag string) error {
	if addr == 0 {
		return errors.New("cudamemtrack: allocation at address 0")
	}
	if size == 0 {
		return fmt.Errorf("cudamemtrack: zero-size allocation at 0x%x", addr)
	}

	var callers []uintptr
	if t.recordStacks {
		callers = make([]uintptr, 32)
		n := runtime.Callers(2, callers)
		callers = callers[:n]
	}

	alloc := &Allocation{
		Address:     addr,
		Size:        size,
		Tag:         tag,
		AllocatedAt: time.Now(),
		Callers:     callers,
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	if prev, exists := t.allocations[addr]; exists {
		return fmt.Errorf("cudamemtrack: address 0x%x is already tracked (%d bytes, tag %q); a Free was missed",
			addr, prev.Size, prev.Tag)
	}

	t.allocations[addr] = alloc
	t.summary.ActiveAllocations++
	t.summary.TotalAllocations++
	t.summary.TotalAllocatedBytes += size
	t.summary.ActiveBytes += size
	if t.summary.ActiveBytes > t.summary.PeakBytes {
		t.summary.PeakBytes = t.summary.ActiveBytes
	}
	return nil
}

// Free marks the device memory block at addr as freed.
// It returns an error if addr was not tracked or already freed.
func (t *Tracker) Free(addr uintptr) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	alloc, exists := t.allocations[addr]
	if !exists {
		return fmt.Errorf("cudamemtrack: untracked or double-free address 0x%x", addr)
	}

	t.summary.ActiveBytes -= alloc.Size
	t.summary.TotalFreedBytes += alloc.Size
	t.summary.ActiveAllocations--
	delete(t.allocations, addr)
	return nil
}

// Live returns a snapshot of all currently active allocations sorted by address.
func (t *Tracker) Live() []Allocation {
	t.mu.Lock()
	defer t.mu.Unlock()

	result := make([]Allocation, 0, len(t.allocations))
	for _, a := range t.allocations {
		result = append(result, *a)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Address < result[j].Address
	})
	return result
}

// Summary returns the current aggregate allocation metrics.
func (t *Tracker) Summary() Summary {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.summary
}

// CheckLeaks verifies that all tracked allocations have been freed.
// If active allocations remain, it returns a formatted error detailing the leaked blocks.
func (t *Tracker) CheckLeaks() error {
	live := t.Live()
	if len(live) == 0 {
		return nil
	}

	var totalLeaked uint64
	var details []string
	for _, a := range live {
		totalLeaked += a.Size
		tag := a.Tag
		if tag == "" {
			tag = "unlabeled"
		}
		details = append(details, fmt.Sprintf("0x%x (%d bytes, tag %q)", a.Address, a.Size, tag))
	}

	return fmt.Errorf("cudamemtrack: %d memory leaks detected (total %d bytes): %s",
		len(live), totalLeaked, strings.Join(details, ", "))
}

// WriteReport formats an audit summary of memory usage to w.
func (t *Tracker) WriteReport(w io.Writer) error {
	s := t.Summary()
	live := t.Live()

	fmt.Fprintf(w, "GPU Memory Allocation Summary:\n")
	fmt.Fprintf(w, "  Active Allocations: %d (%d bytes)\n", s.ActiveAllocations, s.ActiveBytes)
	fmt.Fprintf(w, "  Peak Usage:         %d bytes\n", s.PeakBytes)
	fmt.Fprintf(w, "  Total Lifetime:     %d allocs (%d bytes allocated, %d bytes freed)\n",
		s.TotalAllocations, s.TotalAllocatedBytes, s.TotalFreedBytes)

	if len(live) > 0 {
		fmt.Fprintf(w, "\nActive Allocations:\n")
		for i, a := range live {
			fmt.Fprintf(w, "  [%d] 0x%x - %d bytes (%s)\n", i+1, a.Address, a.Size, a.Tag)
			for _, frame := range a.StackFrames() {
				fmt.Fprintf(w, "      at %s\n", frame)
			}
		}
	}
	return nil
}

// Block represents a contiguous memory slice in virtual or physical address space.
type Block struct {
	Address uintptr
	Size    uint64
	Free    bool
}

// Fragmentation calculates the memory fragmentation ratio:
//
//	Fragmentation = 1.0 - (LargestContiguousFree / TotalFree)
//
// A value of 0.0 indicates zero fragmentation (all free memory is in one contiguous block).
// A value approaching 1.0 indicates severe fragmentation (free memory is split into tiny disjoint islands).
//
// Fragmentation returns NaN when largestFree exceeds totalFree. The largest contiguous
// free block is a part of the free memory, so that input is self-contradictory and says
// the caller's block accounting is wrong. Returning 0.0 for it would be indistinguishable
// from a perfectly unfragmented heap.
//
// When totalFree is 0 there is no free memory to fragment and the ratio is 0.0.
func Fragmentation(totalFree, largestFree uint64) float64 {
	if largestFree > totalFree {
		return math.NaN()
	}
	if totalFree == 0 {
		return 0.0
	}
	return 1.0 - (float64(largestFree) / float64(totalFree))
}

// ComputeFragmentation evaluates the fragmentation ratio across a slice of memory blocks.
func ComputeFragmentation(blocks []Block) float64 {
	var totalFree uint64
	var largestFree uint64

	for _, b := range blocks {
		if b.Free && b.Size > 0 {
			totalFree += b.Size
			if b.Size > largestFree {
				largestFree = b.Size
			}
		}
	}

	return Fragmentation(totalFree, largestFree)
}

// ValidateBlocks verifies that blocks are sorted by address and do not overlap.
func ValidateBlocks(blocks []Block) error {
	for i := 1; i < len(blocks); i++ {
		prev := blocks[i-1]
		curr := blocks[i]
		if curr.Address < prev.Address+uintptr(prev.Size) {
			return errors.New("cudamemtrack: overlapping or unsorted memory blocks")
		}
	}
	return nil
}
