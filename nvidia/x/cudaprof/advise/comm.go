package advise

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// CategoryCommunication holds findings about collective and peer-to-peer
// communication between devices.
const CategoryCommunication Category = "Communication"

// commKind names the communication mechanism of a device activity, or
// returns "" for one that is not communication.
//
// A collective kernel is not a compute hotspot. Its duration is the slowest
// peer's arrival plus the transfer, and on a single rank's timeline the two
// are indistinguishable: an NCCL all-reduce that "takes" 2 ms may have moved
// its bytes in 100 us and spun for the rest. Ranking it beside GEMMs, or
// telling its author to batch it, misreads both halves.
func commKind(ev timeline.Event) string {
	if strings.EqualFold(ev.Subtype, "memcpy") {
		if cupti.ActivityMemcpyKind(ev.CopyKind) == cupti.ActivityMemcpyKindPtop {
			return "peer copy"
		}
		return ""
	}
	return commKindName(ev.Name)
}

// commKindName classifies a kernel by its name.
//
// The NCCL and NVSHMEM prefixes are the libraries' own. multimem and NVLS
// name the NVLink SHARP multicast path, which is how cudacollective and the
// e-graph collective emitters spell it. The collective verbs catch kernels
// that fuse a collective into compute -- allreduce_residual_rmsnorm,
// gemm_allreduce -- which communicate whatever else they do.
func commKindName(name string) string {
	n := strings.ToLower(name)
	switch {
	case n == "":
		return ""
	case strings.Contains(n, "nccl"):
		return "NCCL"
	case strings.Contains(n, "nvshmem"):
		return "NVSHMEM"
	case strings.Contains(n, "multimem"), strings.Contains(n, "nvls"):
		return "NVLink multicast"
	}
	for _, verb := range []string{"allreduce", "all_reduce", "allgather", "all_gather", "reduce_scatter", "reducescatter", "alltoall", "all_to_all"} {
		if strings.Contains(n, verb) {
			return "collective"
		}
	}
	return ""
}

// commSummary is how one device's timeline divides between communication
// and compute.
type commSummary struct {
	span    uint64
	comm    uint64 // union of communication activity
	compute uint64 // union of every other device activity

	// exposed is communication time during which no compute ran on any
	// stream: the part of the collective the step waits for.
	exposed uint64

	// idleBeforeComm is idle time that ended at a communication activity.
	// On a single rank it is the rank waiting to enter a collective -- a
	// late host, or NCCL's proxy, or a peer -- and no launch record on this
	// rank says which.
	idleBeforeComm uint64

	kernels []commKernel // by time, largest first
}

type commKernel struct {
	name    string
	kind    string
	count   int
	nanos   uint64
	exposed uint64
}

type interval struct{ start, end uint64 }

// union merges intervals, which it sorts in place.
func union(ivs []interval) []interval {
	if len(ivs) == 0 {
		return nil
	}
	sort.Slice(ivs, func(i, j int) bool { return ivs[i].start < ivs[j].start })
	out := []interval{ivs[0]}
	for _, x := range ivs[1:] {
		last := &out[len(out)-1]
		if x.start > last.end {
			out = append(out, x)
			continue
		}
		last.end = max(last.end, x.end)
	}
	return out
}

func total(ivs []interval) uint64 {
	var n uint64
	for _, iv := range ivs {
		n += iv.end - iv.start
	}
	return n
}

// uncovered returns how much of x no interval of cover (a union) overlaps.
func uncovered(x interval, cover []interval) uint64 {
	n := x.end - x.start
	i := sort.Search(len(cover), func(i int) bool { return cover[i].end > x.start })
	for ; i < len(cover) && cover[i].start < x.end; i++ {
		lo, hi := max(x.start, cover[i].start), min(x.end, cover[i].end)
		if hi > lo {
			n -= hi - lo
		}
	}
	return n
}

// summarizeCommunication divides a timeline between communication and
// compute. It returns ok=false when the timeline holds no timed
// communication.
func summarizeCommunication(streams map[uint32][]timeline.Event) (commSummary, bool) {
	var comm, compute, all []interval
	byName := map[string]*commKernel{}
	type commEvent struct {
		iv interval
		k  *commKernel
	}
	var commEvents []commEvent
	var first, last uint64
	haveFirst := false
	for _, evs := range streams {
		for _, ev := range evs {
			if !deviceWorkSubtype(ev.Subtype, true) || ev.End <= ev.Start {
				continue
			}
			iv := interval{ev.Start, ev.End}
			all = append(all, iv)
			if !haveFirst || ev.Start < first {
				first, haveFirst = ev.Start, true
			}
			last = max(last, ev.End)
			kind := commKind(ev)
			if kind == "" {
				compute = append(compute, iv)
				continue
			}
			comm = append(comm, iv)
			k := byName[ev.Name]
			if k == nil {
				k = &commKernel{name: ev.Name, kind: kind}
				byName[ev.Name] = k
			}
			k.count++
			k.nanos += ev.End - ev.Start
			commEvents = append(commEvents, commEvent{iv, k})
		}
	}
	if len(comm) == 0 {
		return commSummary{}, false
	}
	starts := map[uint64]bool{}
	for _, iv := range comm {
		starts[iv.start] = true
	}
	compute = union(compute)
	s := commSummary{span: last - first, compute: total(compute)}
	for _, ce := range commEvents {
		ce.k.exposed += uncovered(ce.iv, compute)
	}
	commUnion := union(comm)
	s.comm = total(commUnion)
	for _, iv := range commUnion {
		s.exposed += uncovered(iv, compute)
	}
	// A gap ends where the next busy interval starts; that gap preceded a
	// collective when a communication activity starts there.
	merged := union(all)
	for i := 1; i < len(merged); i++ {
		if starts[merged[i].start] {
			s.idleBeforeComm += merged[i].start - merged[i-1].end
		}
	}
	for _, k := range byName {
		s.kernels = append(s.kernels, *k)
	}
	sort.Slice(s.kernels, func(i, j int) bool {
		if s.kernels[i].nanos != s.kernels[j].nanos {
			return s.kernels[i].nanos > s.kernels[j].nanos
		}
		return s.kernels[i].name < s.kernels[j].name
	})
	return s, true
}

// describeKernels names the largest communication activities.
func (s commSummary) describeKernels() string {
	const named = 4
	var parts []string
	for i, k := range s.kernels {
		if i == named {
			parts = append(parts, fmt.Sprintf("and %d more", len(s.kernels)-named))
			break
		}
		parts = append(parts, fmt.Sprintf("%s (%s, %d launches, %.3f ms, %.3f ms exposed)", cleanKernelName(k.name), k.kind, k.count, float64(k.nanos)/1e6, float64(k.exposed)/1e6))
	}
	return strings.Join(parts, ", ")
}

// peerWaitNote says, for a finding about device busy or idle time, how much
// of either belongs to communication. It is empty for a capture without
// communication.
func (s commSummary) peerWaitNote() string {
	if s.comm == 0 {
		return ""
	}
	note := fmt.Sprintf(" Of the device work, %.3f ms was communication, whose duration includes waiting for the slowest peer; compute ran %.3f ms.", float64(s.comm)/1e6, float64(s.compute)/1e6)
	if s.idleBeforeComm > 0 {
		note += fmt.Sprintf(" %.3f ms of the idle ended at a communication op: this rank waiting to enter a collective, which a single rank's capture cannot attribute to the host, the proxy or a peer. It is peer wait, not an unexplained stall; analyze all ranks together to name the straggler.", float64(s.idleBeforeComm)/1e6)
	}
	return note
}

// auditCommunication reports the communication a capture ran and how much
// of it compute did not hide.
//
// The finding is about overlap, not about the collective's speed. A
// collective that overlaps compute on another stream costs the step nothing
// however long it runs; one that runs alone costs all of it. Which part of an
// exposed collective was transfer and which was waiting for a peer cannot be
// read off one rank, and the finding says so rather than guessing.
func auditCommunication(cap *cudacapture.Capture, report *Report) {
	streams, _ := timeline.Events(cap)
	s, ok := summarizeCommunication(streams)
	if !ok || s.span == 0 {
		return
	}
	exposedPct := float64(s.exposed) / float64(s.span) * 100
	hiddenPct := 0.0
	if s.comm > 0 {
		hiddenPct = float64(s.comm-s.exposed) / float64(s.comm) * 100
	}
	severity := SeverityCost
	if exposedPct < 1 {
		severity = SeverityContext
	}
	report.Items = append(report.Items, Item{
		rule:            "exposed-comm",
		Severity:        severity,
		Category:        CategoryCommunication,
		EvidenceLevel:   EvidenceObserved,
		DeviceTimeNanos: int64(s.exposed),
		ShareOfProfile:  exposedPct,
		magnitude:       fmt.Sprintf("%.1f%% of the span exposed, %.3f ms", exposedPct, float64(s.exposed)/1e6),
		Finding: fmt.Sprintf("Exposed communication: %.3f ms of %.3f ms communication ran with no compute on any stream (%.1f%% of the %.3f ms span; %.1f%% of communication was overlapped). Communication: %s.",
			float64(s.exposed)/1e6, float64(s.comm)/1e6, exposedPct, float64(s.span)/1e6, hiddenPct, s.describeKernels()),
		ActionableRecommendation: "Overlap the exposed collectives with independent compute: issue them on their own stream and split the producer so the next layer's work runs during the transfer, or fuse the collective into the producing kernel's epilogue (multimem/NVLS reductions on NVLink). A collective's duration on one rank includes waiting for the slowest peer, so before shortening the transfer, analyze every rank's capture together to see whether this rank is waiting on a straggler.",
		NextMeasurement:          "cudaprof advise on every rank's capture together: per-rank skew, the straggler, and exposed communication per rank",
	})
}
