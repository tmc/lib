package cudatracehost

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprobe"
)

// Target binds an event name to verified library provenance. Arguments is the
// number of ABI argument slots collected, including stack slots.
type Target struct {
	Name      string
	Identity  cudaprobe.Identity
	Arguments int
}
type frame struct {
	target Target
	entry  Observation
}

// Correlator pairs entry/return events by observed TID and nested call order.
// It bounds outstanding calls and never pairs a return across an observation
// gap. AddressSpace remains zero unless independently supplied by the caller.
type Correlator struct {
	targets      map[string]Target
	stacks       map[uint32][]frame
	limit, count int
	sequence     uint64
	Incomplete   bool
}

func NewCorrelator(targets []Target, maxPending int) (*Correlator, error) {
	if maxPending <= 0 {
		return nil, fmt.Errorf("cudatracehost: invalid pending limit")
	}
	c := &Correlator{targets: make(map[string]Target), stacks: make(map[uint32][]frame), limit: maxPending, Incomplete: true} // late attachment is always incomplete
	for _, t := range targets {
		if t.Name == "" || strings.ContainsAny(t.Name, ": \n") || t.Arguments < 0 || t.Arguments > 11 || t.Identity.Process.PID == 0 || t.Identity.Process.Generation == 0 || t.Identity.BuildID == "" {
			return nil, fmt.Errorf("cudatracehost: invalid target identity")
		}
		if _, ok := c.targets[t.Name]; ok {
			return nil, fmt.Errorf("cudatracehost: duplicate target")
		}
		c.targets[t.Name] = t
	}
	return c, nil
}

// Gap invalidates outstanding entry evidence. Lost observations cannot bridge
// an exact host call. Sequence advances so downstream trackers observe a gap.
func (c *Correlator) Gap() {
	c.stacks = make(map[uint32][]frame)
	c.count = 0
	c.sequence++
	c.Incomplete = true
}

// Observe returns a completed capture when a matching return arrives. Raw
// entry and return lines should also be retained by the consumer independently.
func (c *Correlator) Observe(o Observation) (*cudaprobe.Capture, error) {
	name, isReturn := strings.CutSuffix(o.Name, "_out")
	if !isReturn {
		if stack := c.stacks[o.TID]; len(stack) > 0 && (stack[len(stack)-1].entry.ThreadGeneration != o.ThreadGeneration || stack[len(stack)-1].entry.ExecID != o.ExecID || stack[len(stack)-1].entry.ThreadGenerationKnown != o.ThreadGenerationKnown) {
			c.count -= len(stack)
			delete(c.stacks, o.TID)
			c.sequence++
			c.Incomplete = true
		}
		var ok bool
		name, ok = strings.CutSuffix(o.Name, "_in")
		if !ok {
			return nil, fmt.Errorf("cudatracehost: unknown event %s", o.Name)
		}
	}
	t, ok := c.targets[name]
	if !ok {
		return nil, fmt.Errorf("cudatracehost: unknown target %s", name)
	}
	if !isReturn {
		if stack := c.stacks[o.TID]; len(stack) > 0 && (stack[len(stack)-1].entry.ThreadGeneration != o.ThreadGeneration || stack[len(stack)-1].entry.ExecID != o.ExecID || stack[len(stack)-1].entry.ThreadGenerationKnown != o.ThreadGenerationKnown) {
			c.count -= len(stack)
			delete(c.stacks, o.TID)
			c.sequence++
			c.Incomplete = true
		}
		for i := 0; i < t.Arguments; i++ {
			if !o.ArgsKnown[i] {
				c.Gap()
				return nil, fmt.Errorf("cudatracehost: missing argument %d", i)
			}
		}
		if c.count >= c.limit {
			c.Gap()
			return nil, fmt.Errorf("cudatracehost: pending calls exceed limit")
		}
		c.stacks[o.TID] = append(c.stacks[o.TID], frame{t, o})
		c.count++
		return nil, nil
	}
	stack := c.stacks[o.TID]
	if len(stack) == 0 || stack[len(stack)-1].target.Name != name || (stack[len(stack)-1].entry.ThreadGeneration != o.ThreadGeneration || stack[len(stack)-1].entry.ExecID != o.ExecID || stack[len(stack)-1].entry.ThreadGenerationKnown != o.ThreadGenerationKnown) || !o.ReturnKnown {
		c.Gap()
		return nil, fmt.Errorf("cudatracehost: unmatched return")
	}
	f := stack[len(stack)-1]
	if o.Timestamp < f.entry.Timestamp {
		c.Gap()
		return nil, fmt.Errorf("cudatracehost: reversed host clock")
	}
	if len(stack) == 1 {
		delete(c.stacks, o.TID)
	} else {
		// Clear the popped frame so the backing array cannot retain raw lines.
		stack[len(stack)-1] = frame{}
		c.stacks[o.TID] = stack[:len(stack)-1]
	}
	c.count--
	c.sequence++
	cap := &cudaprobe.Capture{Identity: f.target.Identity}
	cap.Identity.ThreadID = o.TID
	cap.Identity.ThreadGeneration = o.ThreadGeneration
	cap.Identity.ThreadGenerationKnown = o.ThreadGenerationKnown
	u := binary.LittleEndian
	u.PutUint16(cap.Data[:], 1)
	u.PutUint16(cap.Data[2:], 3)
	u.PutUint64(cap.Data[8:], c.sequence)
	u.PutUint64(cap.Data[16:], f.entry.Timestamp)
	u.PutUint32(cap.Data[24:], uint32(o.Return))
	for i, a := range f.entry.Args {
		u.PutUint64(cap.Data[32+8*i:], a)
	}
	// This layer does not acquire output pointers. The BPF adapter sets flag 4
	// only when the kernel reports a successful output read; tracefs cannot.
	return cap, nil
}
