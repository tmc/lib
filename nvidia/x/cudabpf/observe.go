package cudabpf

import "fmt"

// An Observer receives one bounded record per hook invocation. It is called
// after the handler has returned, never from inside one: a fault handler does
// no logging beyond a fixed-size counter or ring slot, and formatting a
// string is not that.
type Observer interface {
	Observe(Outcome)
}

// A Counters is the fixed-size observation a fault path can afford. It
// allocates nothing and its size does not depend on how many hooks fire.
type Counters struct {
	Consulted uint64
	Opinions  uint64
	Submitted uint64
	Fallbacks uint64
	Conflicts uint64
	Attempted uint64
}

// Observe folds one outcome into the counters.
func (c *Counters) Observe(o Outcome) {
	if o.Consulted {
		c.Consulted++
	}
	if o.HadOpinion {
		c.Opinions++
	}
	if o.Submitted {
		c.Submitted++
	}
	if o.Fallback {
		c.Fallbacks++
	}
	if o.DriverConflict {
		c.Conflicts++
	}
	if o.DriverAttempted {
		c.Attempted++
	}
}

func (c Counters) String() string {
	return fmt.Sprintf("consulted %d, opinions %d, submitted %d, fallbacks %d, driver attempted %d, conflicts %d",
		c.Consulted, c.Opinions, c.Submitted, c.Fallbacks, c.Attempted, c.Conflicts)
}

// A RecordSink receives a flight-recorder-shaped record. It is an interface
// rather than a direct dependency so the policy layer stays importable by
// anything, and so a record's shape is fixed here rather than by whoever
// happens to consume it.
type RecordSink interface {
	PushPolicyRecord(timestampNs int64, hook string, labels map[string]string)
}

// A SinkObserver forwards outcomes to a [RecordSink] as labelled records.
// Labels carry the four things worth separating after the fact: whether the
// policy had an opinion, whether it was submitted, whether the driver
// recorded taking it, and why the driver default stands if it does.
type SinkObserver struct {
	Sink  RecordSink
	Clock func() int64
}

// Observe forwards one outcome.
func (s SinkObserver) Observe(o Outcome) {
	if s.Sink == nil {
		return
	}
	labels := map[string]string{
		"hook":             o.Hook,
		"opinion":          fmt.Sprintf("%t", o.HadOpinion),
		"submitted":        fmt.Sprintf("%t", o.Submitted),
		"driver_attempted": fmt.Sprintf("%t", o.DriverAttempted),
		"driver_conflict":  fmt.Sprintf("%t", o.DriverConflict),
		"fallback":         fmt.Sprintf("%t", o.Fallback),
	}
	if o.Reason != "" {
		labels["reason"] = o.Reason
	}
	if o.HadOpinion && o.Reorder.Destination != DestinationUnset {
		labels["reorder"] = o.Reorder.String()
	}
	var ts int64
	if s.Clock != nil {
		ts = s.Clock()
	}
	s.Sink.PushPolicyRecord(ts, o.Hook, labels)
}
