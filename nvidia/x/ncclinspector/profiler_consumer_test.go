package ncclinspector_test

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
	"github.com/tmc/lib/nvidia/x/nccljoin"
)

// The Inspector records and clock controls are the constructed two-rank
// fixture used by the existing joined-rank consumer tests. The profiler
// traces are constructed from the pinned example-profiler print formats.
// Nothing here is a same-run capture of both plugins.

const (
	fixtureComm = "0x1234567890abcdef"
	fixtureSeq  = 7
)

var fixturePID = []int{2100, 2101}

// fixtureRecord is the fixture record with an observed kernel event on
// channel 3 added, so that bindings of channels 0 and 3 are observed.
func fixtureRecord(t *testing.T, rank int) ncclinspector.Record {
	t.Helper()
	r := fixtureRecordRaw(t, rank)
	ts := *r.Collective.EventTraceTS
	ts.KernelEvents = append(append([]ncclinspector.KernelEventTS(nil), ts.KernelEvents...),
		ncclinspector.KernelEventTS{ChannelID: 3, KernelStartTS: 1100, KernelStopTS: 1300, KernelRecordTS: 1400})
	c := *r.Collective
	c.EventTraceTS = &ts
	r.Collective = &c
	return r
}

// fixtureRecordRaw is the fixture record as read: kernel events on channel 0 only.
func fixtureRecordRaw(t *testing.T, rank int) ncclinspector.Record {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(fixtureDir, []string{"rank0.delayed.jsonl", "rank1.delayed.jsonl"}[rank]))
	if err != nil {
		t.Fatal(err)
	}
	rs, err := ncclinspector.Read(bytes.NewReader(b), ncclinspector.Limits{})
	if err != nil || len(rs) != 1 {
		t.Fatalf("fixture rank %d: %v", rank, err)
	}
	return rs[0]
}

func rankColl(rank int) tcoll {
	c := goodColl()
	c.rank = rank
	if rank == 0 {
		// Rank 0's compute phase is shorter; its channel 3 is absent.
		c.channels = []tchan{{id: 0, start: 5000, stop: 6000, phases: []tphase{
			{0, 5000, 5200, "11.300000"}, {1, 5200, 5500, "11.400000"}, {2, 5500, 6000, "11.500000"},
		}}, c.channels[1]}
	}
	return c
}

func rankTrace(t *testing.T, rank int, p ncclinspector.ProfilerProfile, c tcoll) *ncclinspector.ProfilerTrace {
	t.Helper()
	tr, err := readTrace(traceOf(fixturePID[rank], c), p, ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

func binding(rank int) ncclinspector.ProfilerBinding {
	return ncclinspector.ProfilerBinding{
		Communicator: fixtureComm,
		Rank:         rank,
		Incarnation:  "constructed-communicator-epoch-17",
		Host:         []string{"constructed-node-r0", "constructed-node-r1"}[rank],
		PID:          fixturePID[rank],
		Operation:    "AllReduce",
		Sequence:     fixtureSeq,
		Channels:     []int{0, 3},
		Evidence:     "constructed-binding-17",
	}
}

func profClock(t *testing.T, rank int) ncclinspector.ClockMapping {
	t.Helper()
	m, err := readManifest()
	if err != nil {
		t.Fatal(err)
	}
	return clockFor(m, rank)
}

func TestBindProfilerJoinRoute(t *testing.T) {
	m, err := readManifest()
	if err != nil {
		t.Fatal(err)
	}
	var captures []nccljoin.Capture
	for rank := range 2 {
		captures = append(captures, nccljoin.Capture{ID: []string{"r0", "r1"}[rank], Records: []ncclinspector.Record{fixtureRecordRaw(t, rank)}})
	}
	joined, err := nccljoin.Join(captures, nccljoin.Options{
		Scope:    nccljoin.Scope{Capture: m.Scope.Capture, Job: m.Scope.Job, Epoch: m.Scope.Epoch},
		Expected: map[string][]int{m.Hash: m.ExpectedRanks},
	})
	if err != nil || len(joined.Collectives) != 1 {
		t.Fatalf("join: %v", err)
	}
	group := joined.Collectives[0]

	withPhases, baseline := ncclinspector.NewTwoRankAttributor(), ncclinspector.NewTwoRankAttributor()
	got := make([]*ncclinspector.ProfilerBound, 2)
	for _, o := range group.Observations {
		rank := o.Header.Rank
		rec := ncclinspector.Record{Header: o.Header, Metadata: o.Metadata, Collective: &o.Collective, Raw: o.Raw}
		clock := clockFor(m, rank)
		want, err := ncclinspector.ObservationFromRecord(rec, clock)
		if err != nil {
			t.Fatal(err)
		}
		bd := binding(rank)
		bd.Channels = []int{0} // the joined fixture record observes channel 0 only
		b, err := ncclinspector.BindProfiler(rankTrace(t, rank, ncclinspector.ProfilerNCCL2323, rankColl(rank)), bd, rec, clock)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(b.Observation, want) {
			t.Fatalf("rank %d: phase binding changed the Inspector observation", rank)
		}
		got[rank] = b
		for _, a := range []struct {
			*ncclinspector.TwoRankAttributor
			obs ncclinspector.CollectiveObservation
		}{{withPhases, b.Observation}, {baseline, want}} {
			obs := a.obs
			c, ok := completionFor(m, "delayed", rank, m.Scope.Epoch)
			if !ok {
				t.Fatal("missing completion control")
			}
			obs.Completion, obs.HasCompletion = c, true
			if err := a.AddObservation(obs); err != nil {
				t.Fatal(err)
			}
		}
	}
	r1, err := withPhases.Attribute(group.Communicator, group.Sequence)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := baseline.Attribute(group.Communicator, group.Sequence)
	if err != nil || !reflect.DeepEqual(r1, r2) || r1.Diagnosis != "host_enqueue_skew" || r1.DelayedRank != 1 {
		t.Fatalf("attribution %+v differs from baseline %+v (%v)", r1, r2, err)
	}

	// Per-rank, per-channel reported durations come from the raw GPU timers.
	type key struct{ rank, channel, phase int }
	want := map[key]int64{
		{0, 0, 0}: 200, {0, 0, 1}: 300, {0, 0, 2}: 500,
		{1, 0, 0}: 200, {1, 0, 1}: 700, {1, 0, 2}: 100,
	}
	have := make(map[key]int64)
	for rank, b := range got {
		if b.Asserted.Binding.Incarnation != "constructed-communicator-epoch-17" || b.Asserted.Build != "constructed-build" || b.Asserted.Clock.EvidenceRef == "" {
			t.Fatalf("rank %d asserted facts = %+v", rank, b.Asserted)
		}
		if b.Collective.Communicator() != fixtureComm || b.Collective.Rank != rank || !b.BytesKnown || b.Bytes != 4096 || b.ArtifactSHA256 == "" {
			t.Fatalf("rank %d observed facts = %+v", rank, b)
		}
		for _, ch := range b.Channels {
			for _, p := range ch.Phases {
				if p.Reported {
					have[key{rank, ch.Channel, p.ID}] = p.ReportedNS
				}
			}
		}
	}
	for k, ns := range want {
		if have[k] != ns {
			t.Errorf("%+v = %d, want %d", k, have[k], ns)
		}
	}
	if len(have) != 6 {
		t.Errorf("reported phases = %d, want 6", len(have))
	}
}

func TestBindProfilerRefusals(t *testing.T) {
	rec := func(rank int) ncclinspector.Record { return fixtureRecord(t, rank) }
	tests := []struct {
		name string
		mod  func(b *ncclinspector.ProfilerBinding, r *ncclinspector.Record, c *tcoll)
		want error
		text string
	}{
		{"hash", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) {
			b.Communicator = "0x1234567890abcdee"
		}, ncclinspector.ErrProfilerBinding, "differs from binding"},
		{"truncated hash", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) {
			b.Communicator = "0x1234567890abcd"
		}, ncclinspector.ErrProfilerBinding, "truncated"},
		{"trace hash", func(_ *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, c *tcoll) { c.hash++ },
			ncclinspector.ErrProfilerBinding, "no collective scope"},
		{"trace rank", func(_ *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, c *tcoll) { c.rank = 0 },
			ncclinspector.ErrProfilerBinding, "no collective scope"},
		{"binding rank", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) { b.Rank = 0 },
			ncclinspector.ErrProfilerBinding, "differs from binding"},
		{"trace sequence", func(_ *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, c *tcoll) { c.seq = 8 },
			ncclinspector.ErrProfilerBinding, "no collective scope"},
		{"binding sequence", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) { b.Sequence = 8 },
			ncclinspector.ErrProfilerBinding, "differs from binding"},
		{"operation", func(_ *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, c *tcoll) { c.name = "AllGather" },
			ncclinspector.ErrProfilerBinding, "no collective scope"},
		{"record operation", func(_ *ncclinspector.ProfilerBinding, r *ncclinspector.Record, _ *tcoll) {
			r.Collective.Name = "AllGather"
		}, ncclinspector.ErrProfilerBinding, "differs from binding"},
		{"pid", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) { b.PID++ },
			ncclinspector.ErrProfilerBinding, "process"},
		{"record pid", func(_ *ncclinspector.ProfilerBinding, r *ncclinspector.Record, _ *tcoll) { r.Metadata.PID++ },
			ncclinspector.ErrProfilerBinding, "differs from binding"},
		{"no incarnation", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) { b.Incarnation = "" },
			ncclinspector.ErrProfilerBinding, "incarnation"},
		{"host", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) {
			b.Host = "constructed-node-r0"
		},
			ncclinspector.ErrProfilerBinding, "differs from binding"},
		{"no host", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) { b.Host = "" },
			ncclinspector.ErrProfilerBinding, "host"},
		{"channel not observed by record", func(b *ncclinspector.ProfilerBinding, r *ncclinspector.Record, _ *tcoll) {
			*r = fixtureRecordRaw(t, 1)
		}, ncclinspector.ErrProfilerBinding, "observed kernel channels"},
		{"no evidence", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) { b.Evidence = "" },
			ncclinspector.ErrProfilerBinding, "evidence"},
		{"no channels", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) { b.Channels = nil },
			ncclinspector.ErrProfilerBinding, "channels"},
		{"missing channel", func(_ *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, c *tcoll) { c.channels = c.channels[:1] },
			ncclinspector.ErrProfilerBinding, "not in the collective"},
		{"repeated channel", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) { b.Channels = []int{3, 3} },
			ncclinspector.ErrProfilerBinding, "twice"},
		{"unknown need", func(b *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, _ *tcoll) {
			b.Require = []ncclinspector.ProfilerNeed{"liveness"}
		}, ncclinspector.ErrProfilerBinding, "unknown need"},
		{"count to bytes", func(_ *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, c *tcoll) { c.count = 512 },
			ncclinspector.ErrProfilerBinding, "bytes"},
		{"datatype size", func(_ *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, c *tcoll) { c.datatype = "ncclFloat64" },
			ncclinspector.ErrProfilerBinding, "bytes"},
		{"unknown datatype", func(_ *ncclinspector.ProfilerBinding, _ *ncclinspector.Record, c *tcoll) { c.datatype = "Unknown" },
			ncclinspector.ErrProfilerUnavailable, "no known size"},
		{"p2p record", func(_ *ncclinspector.ProfilerBinding, r *ncclinspector.Record, _ *tcoll) {
			r.Collective, r.PointToPoint = nil, &ncclinspector.PointToPoint{}
		}, ncclinspector.ErrProfilerBinding, "differs from binding"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, r, c := binding(1), rec(1), rankColl(1)
			tt.mod(&b, &r, &c)
			tr := rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, c)
			_, err := ncclinspector.BindProfiler(tr, b, r, profClock(t, 1))
			if !errors.Is(err, tt.want) || !strings.Contains(err.Error(), tt.text) {
				t.Fatalf("err = %v, want %v containing %q", err, tt.want, tt.text)
			}
		})
	}
	real := rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, rankColl(1))
	handBuilt := &ncclinspector.ProfilerTrace{PID: real.PID, Source: real.Source, Collectives: real.Collectives}
	if _, err := ncclinspector.BindProfiler(handBuilt, binding(1), rec(1), profClock(t, 1)); !errors.Is(err, ncclinspector.ErrProfilerBinding) {
		t.Fatalf("trace not from ReadProfilerTrace: %v", err)
	}
	if _, err := ncclinspector.BindProfiler(nil, binding(1), rec(1), profClock(t, 1)); !errors.Is(err, ncclinspector.ErrProfilerBinding) {
		t.Fatalf("nil trace: %v", err)
	}
	// A clock mapping that ObservationFromRecord refuses refuses the binding.
	bad := profClock(t, 1)
	bad.EvidenceRef = ""
	if _, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, rankColl(1)), binding(1), rec(1), bad); !errors.Is(err, ncclinspector.ErrProfilerBinding) {
		t.Fatalf("incomplete clock mapping: %v", err)
	}
}

func TestBindProfilerSizeUnverifiedIsExplicit(t *testing.T) {
	c := rankColl(1)
	c.datatype = "Unknown"
	b := binding(1)
	b.SizeUnverified = true
	got, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, c), b, fixtureRecord(t, 1), profClock(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	if got.BytesKnown || got.Bytes != 0 {
		t.Fatalf("size = %d known %t", got.Bytes, got.BytesKnown)
	}
}

func TestBindProfilerUnmetNeeds(t *testing.T) {
	complete := func(needs ...ncclinspector.ProfilerNeed) error {
		b := binding(1)
		b.Require = needs
		_, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, rankColl(1)), b, fixtureRecord(t, 1), profClock(t, 1))
		return err
	}
	if err := complete(ncclinspector.NeedPhases); err != nil {
		t.Fatalf("complete phases refused: %v", err)
	}
	for _, n := range []ncclinspector.ProfilerNeed{ncclinspector.NeedLoss, ncclinspector.NeedProgressAge, ncclinspector.NeedCrossRankTimeline} {
		if err := complete(n); !errors.Is(err, ncclinspector.ErrProfilerUnavailable) {
			t.Errorf("need %s: err = %v", n, err)
		}
	}

	// A missing phase and an empty interval refuse NeedPhases only.
	for name, mut := range map[string]func(*tcoll){
		"missing final_sync":   func(c *tcoll) { c.channels[0].phases = c.channels[0].phases[:2] },
		"missing initial_sync": func(c *tcoll) { c.channels[1].phases = c.channels[1].phases[1:] },
		"zero interval":        func(c *tcoll) { c.channels[0].phases[2].end = c.channels[0].phases[2].start },
		"reversed interval":    func(c *tcoll) { c.channels[0].phases[0].end = c.channels[0].phases[0].start - 1 },
		"absent GPU start":     func(c *tcoll) { c.channels[0].phases[0].start = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			c := rankColl(1)
			mut(&c)
			tr := rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, c)
			b := binding(1)
			got, err := ncclinspector.BindProfiler(tr, b, fixtureRecord(t, 1), profClock(t, 1))
			if err != nil {
				t.Fatalf("a missing or empty phase is not a binding failure: %v", err)
			}
			if len(got.Unavailable) == 0 || got.Unavailable[0].Family != "loss" {
				t.Fatalf("unavailable = %+v", got.Unavailable)
			}
			b.Require = []ncclinspector.ProfilerNeed{ncclinspector.NeedPhases}
			if _, err := ncclinspector.BindProfiler(tr, b, fixtureRecord(t, 1), profClock(t, 1)); !errors.Is(err, ncclinspector.ErrProfilerUnavailable) {
				t.Fatalf("NeedPhases err = %v", err)
			}
		})
	}
}

func TestBindProfilerMissingPhaseIsNotLoss(t *testing.T) {
	// before_close stamped at the kernel end: the writer skips final_sync.
	c := rankColl(1)
	c.channels[0].phases = c.channels[0].phases[:2]
	c.channels[0].phases[1].end = c.channels[0].stop
	got, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, c), binding(1), fixtureRecord(t, 1), profClock(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	ch := got.Channels[0]
	if !reflect.DeepEqual(ch.Absent, []string{"final_sync"}) || len(ch.Phases) != 2 || ch.Withheld != "" || !ch.Phases[1].Reported {
		t.Fatalf("channel = %+v", ch)
	}
	for _, u := range got.Unavailable {
		if u.Family == "loss" && !strings.Contains(u.Reason, "no drop") {
			t.Fatalf("loss reason = %q", u.Reason)
		}
	}
}

func TestBindProfilerHostCallbackIsNotGPUTime(t *testing.T) {
	// Changing every printed host callback time must not change any
	// reported duration or the bound Inspector observation.
	a := rankColl(1)
	b := rankColl(1)
	for _, ch := range b.channels {
		for i := range ch.phases {
			ch.phases[i].ts = "999.000001"
		}
	}
	ra, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, a), binding(1), fixtureRecord(t, 1), profClock(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	rb, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, b), binding(1), fixtureRecord(t, 1), profClock(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	for i := range ra.Channels {
		for j := range ra.Channels[i].Phases {
			pa, pb := ra.Channels[i].Phases[j], rb.Channels[i].Phases[j]
			if pa.ReportedNS != pb.ReportedNS || pa.HostCallbackMicros == pb.HostCallbackMicros || pa.ReportedNS == 0 {
				t.Fatalf("phase %d/%d = %+v %+v", i, j, pa, pb)
			}
		}
	}
	if !reflect.DeepEqual(ra.Observation, rb.Observation) {
		t.Fatal("host callback times changed the Inspector observation")
	}
	for _, u := range ra.Unavailable {
		if u.Family == "cross_rank_timeline" && !strings.Contains(u.Reason, "not GPU starts") {
			t.Fatalf("timeline reason = %q", u.Reason)
		}
	}
}

func TestBindProfilerChannelOwnership(t *testing.T) {
	// The same phase kinds on different channels are not interchangeable.
	c := rankColl(1)
	c.channels[1].phases = phases(8000)
	got, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, c), binding(1), fixtureRecord(t, 1), profClock(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	if got.Channels[0].Phases[0].Raw == nil || got.Channels[0].Channel != 0 || got.Channels[1].Channel != 3 ||
		got.Channels[0].Phases[0].ReportedNS != 200 || got.Channels[1].StartGpuClk != 5100 || !bytes.Contains(got.Channels[1].Phases[0].Raw, []byte(`"StartGpuClk": 8000`)) {
		t.Fatalf("channels = %+v", got.Channels)
	}
	// Binding only channel 3 leaves channel 0's phases out of the result.
	b := binding(1)
	b.Channels = []int{3}
	got, err = ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, c), b, fixtureRecord(t, 1), profClock(t, 1))
	if err != nil || len(got.Channels) != 1 || got.Channels[0].Channel != 3 {
		t.Fatalf("channel subset = %+v, %v", got, err)
	}
}

func TestBindProfiler2312(t *testing.T) {
	// Rank 0 is bound in 2.31; its COLL rank is not discarded. A regular
	// collective's phases are retained but not admitted as durations.
	c := rankColl(0)
	c.sym = 0
	b := binding(0)
	got, err := ncclinspector.BindProfiler(rankTrace(t, 0, ncclinspector.ProfilerNCCL2312, c), b, fixtureRecord(t, 0), profClock(t, 0))
	if err != nil {
		t.Fatal(err)
	}
	if got.Profile != ncclinspector.ProfilerNCCL2312 || got.Collective.Rank != 0 {
		t.Fatalf("bound = %+v", got.Collective)
	}
	for _, p := range got.Channels[0].Phases {
		if p.Reported || p.Reason == "" {
			t.Fatalf("regular phase admitted in 2.31: %+v", p)
		}
	}
	var note string
	for _, u := range got.Unavailable {
		if u.Family == "api_event_rank" {
			note = u.Reason
		}
	}
	if !strings.Contains(note, "leaves the descriptor rank zero") {
		t.Fatalf("api rank note = %q", note)
	}
	b.Require = []ncclinspector.ProfilerNeed{ncclinspector.NeedPhases}
	if _, err := ncclinspector.BindProfiler(rankTrace(t, 0, ncclinspector.ProfilerNCCL2312, c), b, fixtureRecord(t, 0), profClock(t, 0)); !errors.Is(err, ncclinspector.ErrProfilerUnavailable) {
		t.Fatalf("withheld phases satisfied NeedPhases: %v", err)
	}

	// The same symmetric collective under 2.32.3 admits its durations.
	sym := rankColl(0)
	got, err = ncclinspector.BindProfiler(rankTrace(t, 0, ncclinspector.ProfilerNCCL2323, sym), binding(0), fixtureRecord(t, 0), profClock(t, 0))
	if err != nil || !got.Channels[0].Phases[1].Reported {
		t.Fatalf("2.32.3 symmetric phases = %+v, %v", got, err)
	}
}

func TestBindProfilerRetainsRaw(t *testing.T) {
	got, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, rankColl(1)), binding(1), fixtureRecord(t, 1), profClock(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(got.Collective.Begin, []byte(`"CommHash": 1311768467294899695`)) {
		t.Fatalf("collective raw = %q", got.Collective.Begin)
	}
	if got.Channels[0].Phases[1].HostCallbackMicros != "11.400000" || !bytes.Contains(got.Channels[0].Phases[1].Raw, []byte(`"PhaseId": 1`)) {
		t.Fatalf("phase = %+v", got.Channels[0].Phases[1])
	}
	if p := got.Channels[0].Phases[0]; p.From != "begin" || p.To != "after_open" || got.Channels[0].Phases[2].To != "end" {
		t.Fatalf("endpoints = %+v", got.Channels[0].Phases)
	}
	if !got.Channels[0].KernelReported || got.Channels[0].KernelNS != 1000 {
		t.Fatalf("kernel = %+v", got.Channels[0])
	}
}

func TestBindProfilerStaleChannel(t *testing.T) {
	// A channel whose host begin precedes the enclosing COLL begin is a
	// reused pool slot's earlier occupant: retained, not attributed.
	c := rankColl(1)
	c.channels = append(c.channels, tchan{id: 5, start: 100, stop: 1100, phases: phases(100)})
	s := strings.Replace(string(traceOf(fixturePID[1], c)), `"ts": 11.250000, "args": {"Channel": 5`, `"ts": 1.250000, "args": {"Channel": 5`, 1)
	if !strings.Contains(s, `"ts": 1.250000, "args": {"Channel": 5`) {
		t.Fatal("fixture edit failed")
	}
	tr, err := readTrace([]byte(s), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	st := tr.Collectives[0].KernelChannels[2]
	if st.Channel != 5 || st.Withheld == "" || st.Phases[0].Withheld == "" || tr.Collectives[0].KernelChannels[0].Withheld != "" {
		t.Fatalf("channels = %+v", tr.Collectives[0].KernelChannels)
	}
	rec := fixtureRecord(t, 1)
	rec.Collective.EventTraceTS.KernelEvents = append(rec.Collective.EventTraceTS.KernelEvents, ncclinspector.KernelEventTS{ChannelID: 5, KernelStartTS: 1100, KernelStopTS: 1300, KernelRecordTS: 1400})
	b := binding(1)
	b.Channels = []int{0, 3, 5}
	got, err := ncclinspector.BindProfiler(tr, b, rec, profClock(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	ch := got.Channels[2]
	if ch.Withheld == "" || ch.KernelReported || ch.Phases[1].Reported || ch.Phases[1].Reason == "" {
		t.Fatalf("stale channel admitted: %+v", ch)
	}
	if !got.Channels[0].Phases[1].Reported {
		t.Fatal("current channel withheld")
	}
	b.Require = []ncclinspector.ProfilerNeed{ncclinspector.NeedPhases}
	if _, err := ncclinspector.BindProfiler(tr, b, rec, profClock(t, 1)); !errors.Is(err, ncclinspector.ErrProfilerUnavailable) {
		t.Fatalf("stale channel satisfied NeedPhases: %v", err)
	}
}

func TestBindProfilerAdjacentEndpointConflict(t *testing.T) {
	// initial_sync.stop and compute.start print the same ts[after_open]; a
	// mismatch in either direction cannot satisfy NeedPhases.
	tests := []struct {
		name  string
		delta int
	}{
		{"exact", 0},
		{"stop before start", 1},
		{"stop after start", -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := rankColl(1)
			c.channels = c.channels[:1]
			c.channels[0].phases[1].start = uint64(int(c.channels[0].phases[1].start) + tt.delta)
			tr := rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, c)
			b := binding(1)
			b.Channels = []int{0}
			got, err := ncclinspector.BindProfiler(tr, b, fixtureRecordRaw(t, 1), profClock(t, 1))
			if err != nil {
				t.Fatalf("a boundary conflict is not a binding failure: %v", err)
			}
			ch := got.Channels[0]
			if conflict := tt.delta != 0; conflict != (ch.Withheld != "") || conflict == ch.Phases[1].Reported || conflict == ch.KernelReported {
				t.Fatalf("channel = %+v", ch)
			}
			b.Require = []ncclinspector.ProfilerNeed{ncclinspector.NeedPhases}
			_, err = ncclinspector.BindProfiler(tr, b, fixtureRecordRaw(t, 1), profClock(t, 1))
			if tt.delta == 0 && err != nil {
				t.Fatalf("exact boundaries refused: %v", err)
			}
			if tt.delta != 0 && !errors.Is(err, ncclinspector.ErrProfilerUnavailable) {
				t.Fatalf("inconsistent boundaries satisfied NeedPhases: %v", err)
			}
		})
	}
}

func TestBindProfilerStaleSymmetricSlotInRegularCollective(t *testing.T) {
	c := rankColl(1)
	c.sym = 0
	c.channels = []tchan{{id: 0, start: 100, stop: 1100, phases: phases(100)}}
	s := strings.Replace(string(traceOf(fixturePID[1], c)), `"ts": 11.250000, "args": {"Channel": 0`, `"ts": 1.250000, "args": {"Channel": 0`, 1)
	tr, err := readTrace([]byte(s), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
	if err != nil {
		t.Fatalf("stale symmetric slot refused the 2.32.3 trace: %v", err)
	}
	if tr.Collectives[0].KernelChannels[0].Withheld == "" {
		t.Fatal("stale channel admitted")
	}
	// The same phases on a current channel remain impossible in 2.32.3.
	if _, err := readTrace(traceOf(fixturePID[1], c), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{}); err == nil {
		t.Fatal("current regular phase accepted")
	}
}

func TestBindProfilerRefusesModifiedTrace(t *testing.T) {
	tr := rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, rankColl(1))
	tr.Collectives[0].KernelChannels[0].Phases[0].ID = 7
	if _, err := ncclinspector.BindProfiler(tr, binding(1), fixtureRecord(t, 1), profClock(t, 1)); !errors.Is(err, ncclinspector.ErrProfilerBinding) {
		t.Fatalf("modified trace: %v", err)
	}
}

func TestBindProfilerCommunicatorCase(t *testing.T) {
	b := binding(1)
	b.Communicator = "0X1234567890ABCDEF"
	got, err := ncclinspector.BindProfiler(rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, rankColl(1)), b, fixtureRecord(t, 1), profClock(t, 1))
	if err != nil || got.Asserted.Binding.Communicator != fixtureComm {
		t.Fatalf("bound = %v, %v", got, err)
	}
}

func TestBindProfilerObservedChannelsUnionSequenceEvents(t *testing.T) {
	tr := rankTrace(t, 1, ncclinspector.ProfilerNCCL2323, rankColl(1))
	r := fixtureRecordRaw(t, 1)
	c := *r.Collective
	c.EventTraceSN = &ncclinspector.EventTraceSN{KernelEvents: []ncclinspector.KernelEventSN{{ChannelID: 3}}}
	r.Collective = &c
	if _, err := ncclinspector.BindProfiler(tr, binding(1), r, profClock(t, 1)); err != nil {
		t.Fatalf("channel observed by sequence events refused: %v", err)
	}
	// A record whose only listed kernel events are sequence events on other
	// channels still refuses an unlisted channel.
	c.EventTraceTS = nil
	c.EventTraceSN = &ncclinspector.EventTraceSN{KernelEvents: []ncclinspector.KernelEventSN{{ChannelID: 0}}}
	r.Collective = &c
	if _, err := ncclinspector.BindProfiler(tr, binding(1), r, profClock(t, 1)); !errors.Is(err, ncclinspector.ErrProfilerBinding) {
		t.Fatalf("unlisted channel: %v", err)
	}
}
