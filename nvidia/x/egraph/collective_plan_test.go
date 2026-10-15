package egraph_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func collectiveComms(t *testing.T, generation cudacollective.Generation, tr cudacollective.Transport) []*cudacollective.Communicator {
	t.Helper()
	m, err := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0, HostID: "h"}, {ID: 1, HostID: "h", DeviceID: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if tr == nil {
		tr = cudacollective.NewDeterministicTestTransport(2)
	}
	var comms []*cudacollective.Communicator
	for r := range 2 {
		c, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Rank: r, Membership: m, Generation: generation, Transport: tr})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { c.Close() })
		comms = append(comms, c)
	}
	return comms
}

// collectiveCalls returns a valid two-rank schedule of steps out-of-place
// f32 allreduces of 4 elements in generation 7.
func collectiveCalls(steps int, generation uint64) [][]egraph.CollectiveCall {
	calls := make([][]egraph.CollectiveCall, 2)
	for r := range 2 {
		var prev uint64
		for i := range steps {
			base := uint64(0x10000*(r+1) + 0x100*i)
			c := egraph.CollectiveCall{
				CollectiveIntent: egraph.CollectiveIntent{
					Communicator: "tp", Generation: generation, Rank: r, WorldSize: 2,
					Operation: "allreduce", DType: "f32", Count: 4, Stream: 3, EffectToken: uint64(100 + i),
				},
				Reduction:   "sum",
				Layout:      "contiguous",
				Partition:   2,
				Send:        egraph.CollectiveBuffer{Address: base, Bytes: 16},
				Recv:        egraph.CollectiveBuffer{Address: base, Offset: 16, Bytes: 16},
				Predecessor: prev,
				Completion:  uint64(1000*(r+1) + i + 1),
				StreamState: egraph.CollectiveStreamOrdered,
			}
			prev = c.Completion
			calls[r] = append(calls[r], c)
		}
	}
	return calls
}

func cloneCalls(calls [][]egraph.CollectiveCall) [][]egraph.CollectiveCall {
	out := make([][]egraph.CollectiveCall, len(calls))
	for r := range calls {
		out[r] = slices.Clone(calls[r])
	}
	return out
}

func collectiveInputs(steps int) [][][]float32 {
	in := make([][][]float32, steps)
	for i := range steps {
		for r := range 2 {
			v := make([]float32, 4)
			for j := range v {
				v[j] = float32((i+1)*10*(r+1) + j*(2*r+1))
			}
			in[i] = append(in[i], v)
		}
	}
	return in
}

func cloneInputs(in [][][]float32) [][][]float32 {
	out := make([][][]float32, len(in))
	for i := range in {
		for _, v := range in[i] {
			out[i] = append(out[i], slices.Clone(v))
		}
	}
	return out
}

// sumOracle sums rank inputs elementwise in float64. The inputs are small
// integers, so the float32 replay must match exactly.
func sumOracle(in [][]float32) []float32 {
	out := make([]float32, len(in[0]))
	for j := range out {
		var s float64
		for _, v := range in {
			s += float64(v[j])
		}
		out[j] = float32(s)
	}
	return out
}

func planCalls(t *testing.T, calls [][]egraph.CollectiveCall, comms []*cudacollective.Communicator, q *egraph.CollectiveQuarantine) (*egraph.CollectivePlan, error) {
	t.Helper()
	if comms == nil {
		comms = collectiveComms(t, 7, nil)
	}
	if q == nil {
		q = new(egraph.CollectiveQuarantine)
	}
	p, err := egraph.CollectiveSchedule{Communicators: comms, Calls: calls, Quarantine: q}.Plan()
	if p != nil {
		t.Cleanup(func() { p.Close() })
	}
	return p, err
}

func TestRecordedMultiRankReplay(t *testing.T) {
	for _, inPlace := range []bool{false, true} {
		calls := collectiveCalls(2, 7)
		if inPlace {
			for r := range calls {
				for i := range calls[r] {
					calls[r][i].InPlace = true
					calls[r][i].Recv = calls[r][i].Send
				}
			}
		}
		p, err := planCalls(t, calls, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		inputs := collectiveInputs(2)
		orig := cloneInputs(inputs)
		rec, err := p.Replay(context.Background(), inputs)
		if err != nil {
			t.Fatal(err)
		}
		if rec.PlanID != p.ID() || rec.Transport != cudacollective.TransportTypeControlPlaneTest || len(rec.Steps) != 2 {
			t.Fatalf("receipt = %+v", rec)
		}
		for i, step := range rec.Steps {
			want := sumOracle(orig[i])
			if step.EffectToken != uint64(100+i) {
				t.Fatalf("step %d effect = %d", i, step.EffectToken)
			}
			for r := range 2 {
				if !slices.Equal(step.Outputs[r], want) {
					t.Fatalf("in-place=%v step %d rank %d = %v, oracle %v", inPlace, i, r, step.Outputs[r], want)
				}
				// Omission control: a replay that skipped the peer would
				// return the rank's own input.
				if slices.Equal(want, orig[i][r]) {
					t.Fatal("oracle cannot detect a dropped peer")
				}
				if step.Completions[r] != calls[r][i].Completion {
					t.Fatalf("step %d rank %d completion = %d", i, r, step.Completions[r])
				}
				if got := slices.Equal(inputs[i][r], want); got != inPlace {
					t.Fatalf("in-place=%v: input overwritten = %v", inPlace, got)
				}
			}
		}
		// The plan owns a copy of the schedule: changing the caller's
		// calls after Plan does not change the replay.
		calls[0][0].Count = 2
		calls[1][0].InPlace = !calls[1][0].InPlace
		if _, err := p.Replay(context.Background(), cloneInputs(orig)); err != nil {
			t.Fatalf("replay after caller mutation: %v", err)
		}
	}
}

func TestRankParticipationAndOrder(t *testing.T) {
	if _, err := planCalls(t, collectiveCalls(2, 7), nil, nil); err != nil {
		t.Fatalf("control schedule refused: %v", err)
	}
	for _, tt := range []struct {
		name   string
		change func(*egraph.CollectiveSchedule)
	}{
		{"drop-rank", func(s *egraph.CollectiveSchedule) { s.Calls = s.Calls[:1] }},
		{"drop-communicator", func(s *egraph.CollectiveSchedule) { s.Communicators = s.Communicators[:1] }},
		{"swap-communicators", func(s *egraph.CollectiveSchedule) {
			s.Communicators[0], s.Communicators[1] = s.Communicators[1], s.Communicators[0]
		}},
		{"rank-missing-call", func(s *egraph.CollectiveSchedule) { s.Calls[1] = s.Calls[1][:1] }},
		{"rank-reordered", func(s *egraph.CollectiveSchedule) {
			c := s.Calls[1]
			c[0].EffectToken, c[1].EffectToken = c[1].EffectToken, c[0].EffectToken
		}},
		{"wrong-rank-field", func(s *egraph.CollectiveSchedule) { s.Calls[1][0].Rank = 0 }},
		{"world-size", func(s *egraph.CollectiveSchedule) { s.Calls[1][0].WorldSize = 3 }},
		{"later-world-size-all-ranks", func(s *egraph.CollectiveSchedule) {
			for r := range s.Calls {
				s.Calls[r][1].WorldSize = 3
			}
		}},
		{"communicator-name", func(s *egraph.CollectiveSchedule) { s.Calls[1][1].Communicator = "dp" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := egraph.CollectiveSchedule{Communicators: collectiveComms(t, 7, nil), Calls: cloneCalls(collectiveCalls(2, 7)), Quarantine: new(egraph.CollectiveQuarantine)}
			tt.change(&s)
			if _, err := s.Plan(); !errors.Is(err, egraph.ErrCollectiveMismatch) {
				t.Fatalf("Plan error = %v, want ErrCollectiveMismatch", err)
			}
		})
	}
}

func TestCommunicatorGenerationEpoch(t *testing.T) {
	if _, err := planCalls(t, collectiveCalls(1, 7), collectiveComms(t, 8, nil), nil); !errors.Is(err, egraph.ErrCollectiveMismatch) {
		t.Fatalf("stale call generation: %v", err)
	}
	calls := collectiveCalls(1, 7)
	calls[1][0].Generation = 8
	if _, err := planCalls(t, calls, nil, nil); !errors.Is(err, egraph.ErrCollectiveMismatch) {
		t.Fatalf("mixed generation: %v", err)
	}
	comms := collectiveComms(t, 7, nil)
	comms[1].MarkFailed(errors.New("abort"))
	if _, err := planCalls(t, collectiveCalls(1, 7), comms, nil); !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
		t.Fatalf("failed generation reused: %v", err)
	}
}

func TestCollectiveTokenOrdering(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func([][]egraph.CollectiveCall)
		want   error
	}{
		{"missing-completion", func(c [][]egraph.CollectiveCall) { c[0][1].Completion = 0 }, egraph.ErrCollectiveToken},
		{"completion-equals-predecessor", func(c [][]egraph.CollectiveCall) { c[0][1].Completion = c[0][1].Predecessor }, egraph.ErrCollectiveToken},
		{"dropped-predecessor", func(c [][]egraph.CollectiveCall) { c[1][1].Predecessor = 0 }, egraph.ErrCollectiveToken},
		{"wrong-predecessor", func(c [][]egraph.CollectiveCall) { c[1][1].Predecessor = c[0][0].Completion }, egraph.ErrCollectiveToken},
		{"stream-change", func(c [][]egraph.CollectiveCall) { c[0][1].Stream = 4 }, egraph.ErrCollectiveToken},
		{"first-waits-on-own-later-completion", func(c [][]egraph.CollectiveCall) {
			for r := range c {
				c[r][0].Predecessor = c[r][1].Completion
			}
		}, egraph.ErrCollectiveToken},
		{"first-waits-on-peer-completion", func(c [][]egraph.CollectiveCall) { c[1][0].Predecessor = c[0][0].Completion }, egraph.ErrCollectiveToken},
		{"repeated-completion", func(c [][]egraph.CollectiveCall) {
			c[0] = append(c[0], c[0][0])
			c[0][2].EffectToken, c[0][2].Predecessor = 200, c[0][1].Completion
			c[1] = append(c[1], c[1][0])
			c[1][2].EffectToken, c[1][2].Predecessor = 200, c[1][1].Completion
			c[1][2].Completion = 9999
		}, egraph.ErrCollectiveToken},
		{"repeated-effect", func(c [][]egraph.CollectiveCall) {
			for r := range c {
				c[r][1].EffectToken = c[r][0].EffectToken
			}
		}, egraph.ErrCollectiveMismatch},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := collectiveCalls(2, 7)
			tt.change(calls)
			if _, err := planCalls(t, calls, nil, nil); !errors.Is(err, tt.want) {
				t.Fatalf("Plan error = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestInPlaceAliasRules(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*egraph.CollectiveCall)
		ok     bool
	}{
		{"out-of-place-adjacent", func(c *egraph.CollectiveCall) {}, true},
		{"in-place-identical", func(c *egraph.CollectiveCall) { c.InPlace = true; c.Recv = c.Send }, true},
		{"in-place-distinct", func(c *egraph.CollectiveCall) { c.InPlace = true }, false},
		{"out-of-place-identical", func(c *egraph.CollectiveCall) { c.Recv = c.Send }, false},
		{"out-of-place-partial-overlap", func(c *egraph.CollectiveCall) { c.Recv.Offset = 12 }, false},
		{"out-of-place-recv-overlaps-start", func(c *egraph.CollectiveCall) {
			c.Recv = egraph.CollectiveBuffer{Address: c.Send.Address - 4, Bytes: 16}
		}, false},
		{"short-extent", func(c *egraph.CollectiveCall) { c.Recv.Bytes = 12 }, false},
		{"long-extent", func(c *egraph.CollectiveCall) { c.Send.Bytes = 20 }, false},
		{"nil-buffer", func(c *egraph.CollectiveCall) { c.Recv.Address = 0 }, false},
		{"range-overflow", func(c *egraph.CollectiveCall) { c.Recv.Offset = ^uint64(0) - 8 }, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := collectiveCalls(1, 7)
			tt.change(&calls[1][0])
			_, err := planCalls(t, calls, nil, nil)
			if tt.ok {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if !errors.Is(err, egraph.ErrCollectiveAlias) {
				t.Fatalf("Plan error = %v, want ErrCollectiveAlias", err)
			}
		})
	}
}

func TestDatatypeCountPartitionMismatch(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func([][]egraph.CollectiveCall)
		want   error
	}{
		{"count", func(c [][]egraph.CollectiveCall) {
			c[1][0].Count = 8
			c[1][0].Send.Bytes, c[1][0].Recv.Bytes, c[1][0].Recv.Offset = 32, 32, 32
		}, egraph.ErrCollectiveMismatch},
		{"dtype", func(c [][]egraph.CollectiveCall) { c[1][0].DType = "i32" }, egraph.ErrCollectiveMismatch},
		{"partition", func(c [][]egraph.CollectiveCall) { c[1][0].Partition = 4 }, egraph.ErrCollectiveMismatch},
		{"partition-indivisible", func(c [][]egraph.CollectiveCall) {
			for r := range c {
				c[r][0].Partition = 3
			}
		}, egraph.ErrCollectiveUnsupported},
		{"root", func(c [][]egraph.CollectiveCall) { c[1][0].Root = 1 }, egraph.ErrCollectiveMismatch},
		{"layout", func(c [][]egraph.CollectiveCall) {
			for r := range c {
				c[r][0].Layout = "strided"
			}
		}, egraph.ErrCollectiveUnsupported},
		{"reduction", func(c [][]egraph.CollectiveCall) {
			for r := range c {
				c[r][0].Reduction = "max"
			}
		}, egraph.ErrCollectiveUnsupported},
		{"operation", func(c [][]egraph.CollectiveCall) {
			for r := range c {
				c[r][0].Operation = "allgather"
			}
		}, egraph.ErrCollectiveUnsupported},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := collectiveCalls(1, 7)
			tt.change(calls)
			if _, err := planCalls(t, calls, nil, nil); !errors.Is(err, tt.want) {
				t.Fatalf("Plan error = %v, want %v", err, tt.want)
			}
		})
	}
	for _, c := range []egraph.CollectiveCall{{Reduction: ""}, {Reduction: "sum", Layout: ""}} {
		call := collectiveCalls(1, 7)[0][0]
		call.Reduction, call.Layout = c.Reduction, c.Layout
		if err := call.Validate(); err == nil || errors.Is(err, egraph.ErrCollectiveUnsupported) {
			t.Fatalf("implicit reduction or layout: %v", err)
		}
	}

	p, err := planCalls(t, collectiveCalls(1, 7), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	short := collectiveInputs(1)
	short[0][1] = short[0][1][:3]
	if _, err := p.Replay(context.Background(), short); !errors.Is(err, egraph.ErrCollectiveMismatch) {
		t.Fatalf("short replay input: %v", err)
	}
	if _, err := p.Replay(context.Background(), collectiveInputs(1)[:0]); !errors.Is(err, egraph.ErrCollectiveMismatch) {
		t.Fatalf("missing replay step: %v", err)
	}
	ints := collectiveCalls(1, 7)
	for r := range ints {
		ints[r][0].DType = "i32"
	}
	pi, err := planCalls(t, ints, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pi.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, egraph.ErrCollectiveUnsupported) {
		t.Fatalf("i32 host replay: %v", err)
	}
	desc, err := collectiveCalls(1, 7)[1][0].OperationDescriptor()
	if err != nil || desc.Kind != cudacollective.KindAllReduce || desc.Reduction != cudacollective.ReductionSum || desc.Count != 4 || desc.InPlace {
		t.Fatalf("descriptor = %+v, %v", desc, err)
	}
}

func TestFailedCollectiveQuarantinesCandidate(t *testing.T) {
	q := new(egraph.CollectiveQuarantine)
	linkDown := errors.New("link down")
	comms := collectiveComms(t, 7, &recordingTransport{err: linkDown})
	p, err := planCalls(t, collectiveCalls(1, 7), comms, q)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, linkDown) {
		t.Fatalf("replay with failing transport: %v", err)
	}
	for r, c := range comms {
		if err := c.CheckAlive(); !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
			t.Fatalf("rank %d communicator alive after failure: %v", r, err)
		}
	}
	if err := q.Err(p.ID()); !errors.Is(err, egraph.ErrCollectiveQuarantined) {
		t.Fatalf("plan not quarantined: %v", err)
	}
	if _, err := p.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, egraph.ErrCollectiveQuarantined) {
		t.Fatalf("quarantined plan replayed: %v", err)
	}
	if _, err := planCalls(t, collectiveCalls(1, 7), collectiveComms(t, 7, nil), q); !errors.Is(err, egraph.ErrCollectiveQuarantined) {
		t.Fatalf("quarantined identity re-planned on fresh communicators: %v", err)
	}
	// Controls: the same schedule plans without the quarantine record, and a
	// new generation is a new identity.
	if _, err := planCalls(t, collectiveCalls(1, 7), collectiveComms(t, 7, nil), nil); err != nil {
		t.Fatalf("control without quarantine: %v", err)
	}
	next, err := planCalls(t, collectiveCalls(1, 8), collectiveComms(t, 8, nil), q)
	if err != nil {
		t.Fatalf("new generation refused: %v", err)
	}
	if next.ID() == p.ID() {
		t.Fatal("generation not part of plan identity")
	}
	if _, err := next.Replay(context.Background(), collectiveInputs(1)); err != nil {
		t.Fatal(err)
	}
}

func TestPartialCollectiveExecutionQuarantines(t *testing.T) {
	for _, tt := range []struct {
		name string
		peer func(*cudacollective.Communicator)
	}{
		{"peer-closed", func(c *cudacollective.Communicator) { c.Close() }},
		{"peer-lost", func(c *cudacollective.Communicator) { c.ReportLostParticipant(0) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			q := new(egraph.CollectiveQuarantine)
			var comms []*cudacollective.Communicator
			tr := &recordingTransport{onExecute: func(c *cudacollective.Communicator) {
				if c.Rank() == 0 {
					tt.peer(comms[1])
				}
			}}
			comms = collectiveComms(t, 7, tr)
			p, err := planCalls(t, collectiveCalls(1, 7), comms, q)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := p.Replay(context.Background(), collectiveInputs(1)); err == nil {
				t.Fatal("replay completed without rank 1")
			}
			if len(tr.execs) != 1 {
				t.Fatalf("executions = %d, want rank 0 only", len(tr.execs))
			}
			if err := comms[0].CheckAlive(); !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
				t.Fatalf("rank 0 alive after its peer missed the collective: %v", err)
			}
			if err := q.Err(p.ID()); !errors.Is(err, egraph.ErrCollectiveQuarantined) {
				t.Fatalf("divergent plan not quarantined: %v", err)
			}
		})
	}
}

func TestTransportCapabilityFallback(t *testing.T) {
	m, err := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0, HostID: "h"}, {ID: 1, HostID: "h", DeviceID: 1}})
	if err != nil {
		t.Fatal(err)
	}
	r := cudacollective.NewResolver(&cudacollective.TCPSocketTransport{}, cudacollective.NewDeterministicTestTransport(2))
	tr, err := r.Resolve(m, cudacollective.TopologyAny)
	if err != nil {
		t.Fatal(err)
	}
	p, err := planCalls(t, collectiveCalls(1, 7), collectiveComms(t, 7, tr), nil)
	if err != nil {
		t.Fatal(err)
	}
	rec, err := p.Replay(context.Background(), collectiveInputs(1))
	if err != nil {
		t.Fatal(err)
	}
	if rec.Transport != cudacollective.TransportTypeControlPlaneTest {
		t.Fatalf("fallback transport = %v", rec.Transport)
	}
	if _, err := cudacollective.NewResolver(&cudacollective.TCPSocketTransport{}).Resolve(m, cudacollective.TopologyAny); !errors.Is(err, cudacollective.ErrExecutionUnavailable) {
		t.Fatalf("unavailable transport resolved: %v", err)
	}
	if _, err := r.Resolve(m, cudacollective.TopologyNVLink); !errors.Is(err, cudacollective.ErrUnsupportedDevice) {
		t.Fatalf("unsupported topology resolved: %v", err)
	}
	if _, err := planCalls(t, collectiveCalls(1, 7), collectiveComms(t, 7, &cudacollective.TCPSocketTransport{}), nil); !errors.Is(err, cudacollective.ErrExecutionUnavailable) {
		t.Fatalf("plan admitted unavailable transport: %v", err)
	}
	// An available transport that waits for peers cannot be driven one rank
	// at a time.
	nccl := &recordingTransport{typ: cudacollective.TransportTypeNCCLTest}
	if _, err := planCalls(t, collectiveCalls(1, 7), collectiveComms(t, 7, nccl), nil); !errors.Is(err, egraph.ErrCollectiveUnsupported) {
		t.Fatalf("plan admitted a peer-waiting transport: %v", err)
	}
}

// tickClock returns a clock that advances by d on every reading.
func tickClock(d time.Duration) func() time.Time {
	var n atomic.Int64
	base := time.Unix(0, 0)
	return func() time.Time { return base.Add(time.Duration(n.Add(1)) * d) }
}

func TestEstimateAndMeasurementRemainSeparate(t *testing.T) {
	schedule := func(estimate time.Duration) egraph.CollectiveSchedule {
		calls := collectiveCalls(2, 7)
		for r := range calls {
			for i := range calls[r] {
				calls[r][i].Estimate = estimate
			}
		}
		return egraph.CollectiveSchedule{
			Communicators: collectiveComms(t, 7, nil),
			Calls:         calls,
			Quarantine:    new(egraph.CollectiveQuarantine),
			Now:           tickClock(time.Millisecond),
		}
	}
	var recs []*egraph.CollectiveReceipt
	var ids []string
	for _, estimate := range []time.Duration{0, time.Hour} {
		p, err := schedule(estimate).Plan()
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		rec, err := p.Replay(context.Background(), collectiveInputs(2))
		if err != nil {
			t.Fatal(err)
		}
		recs, ids = append(recs, rec), append(ids, p.ID())
	}
	if ids[0] != ids[1] {
		t.Fatal("estimate changed plan identity")
	}
	if recs[0].Estimated != 0 || recs[1].Estimated != 2*time.Hour {
		t.Fatalf("estimated = %v, %v", recs[0].Estimated, recs[1].Estimated)
	}
	// Every observation is one clock tick around one runtime execution,
	// whatever the estimate.
	want := []egraph.CollectiveObservation{
		{Step: 0, Rank: 0, Elapsed: time.Millisecond}, {Step: 0, Rank: 1, Elapsed: time.Millisecond},
		{Step: 1, Rank: 0, Elapsed: time.Millisecond}, {Step: 1, Rank: 1, Elapsed: time.Millisecond},
	}
	for i, rec := range recs {
		if !slices.Equal(rec.Observations, want) {
			t.Fatalf("receipt %d observations = %+v, want %+v", i, rec.Observations, want)
		}
	}
	for i := range recs[0].Steps {
		if !slices.EqualFunc(recs[0].Steps[i].Outputs, recs[1].Steps[i].Outputs, slices.Equal) {
			t.Fatal("estimate changed replay outputs")
		}
	}
}

func TestGraphCaptureBoundary(t *testing.T) {
	for _, tt := range []struct {
		name  string
		state egraph.CollectiveStreamState
		want  error
	}{
		{"capturing", egraph.CollectiveStreamCapturing, egraph.ErrCollectiveUnsupported},
		{"cancelled", egraph.CollectiveStreamCancelled, egraph.ErrCollectiveUnsupported},
		{"unset", 0, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := collectiveCalls(1, 7)
			for r := range calls {
				calls[r][0].StreamState = tt.state
			}
			_, err := planCalls(t, calls, nil, nil)
			if err == nil {
				t.Fatal("stream state admitted")
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("Plan error = %v, want %v", err, tt.want)
			}
		})
	}
	q := new(egraph.CollectiveQuarantine)
	p, err := planCalls(t, collectiveCalls(1, 7), nil, q)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Replay(ctx, collectiveInputs(1)); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled replay: %v", err)
	}
	if err := q.Err(p.ID()); err != nil {
		t.Fatalf("cancellation before replay quarantined the plan: %v", err)
	}
	if _, err := p.Replay(context.Background(), collectiveInputs(1)); err != nil {
		t.Fatal(err)
	}
}

func TestRewriteCannotDropCollectiveEffect(t *testing.T) {
	before := collectiveCalls(3, 7)[0]
	if err := egraph.CheckCollectiveRewrite(before, slices.Clone(before)); err != nil {
		t.Fatalf("identity rewrite refused: %v", err)
	}
	estimated := slices.Clone(before)
	estimated[1].Estimate = time.Millisecond
	if err := egraph.CheckCollectiveRewrite(before, estimated); err != nil {
		t.Fatalf("estimate-only rewrite refused: %v", err)
	}
	for _, tt := range []struct {
		name  string
		after func([]egraph.CollectiveCall) []egraph.CollectiveCall
	}{
		{"drop-middle", func(c []egraph.CollectiveCall) []egraph.CollectiveCall { return slices.Delete(c, 1, 2) }},
		{"drop-last", func(c []egraph.CollectiveCall) []egraph.CollectiveCall { return c[:2] }},
		{"reorder", func(c []egraph.CollectiveCall) []egraph.CollectiveCall { c[0], c[1] = c[1], c[0]; return c }},
		{"duplicate", func(c []egraph.CollectiveCall) []egraph.CollectiveCall { return append(c, c[2]) }},
		{"completion", func(c []egraph.CollectiveCall) []egraph.CollectiveCall { c[2].Completion++; return c }},
		{"buffer", func(c []egraph.CollectiveCall) []egraph.CollectiveCall { c[0].Recv.Offset += 4; return c }},
		{"count", func(c []egraph.CollectiveCall) []egraph.CollectiveCall { c[0].Count = 2; return c }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := egraph.CheckCollectiveRewrite(before, tt.after(slices.Clone(before))); !errors.Is(err, egraph.ErrCollectiveMismatch) {
				t.Fatalf("rewrite error = %v, want ErrCollectiveMismatch", err)
			}
		})
	}
	// A drop applied consistently on every rank still forms a valid
	// schedule, so rank-wide planning alone cannot see it: the rewrite check
	// is what refuses it.
	dropped := collectiveCalls(3, 7)
	for r := range dropped {
		dropped[r] = dropped[r][:2]
	}
	if _, err := planCalls(t, dropped, nil, nil); err != nil {
		t.Fatalf("consistently dropped schedule: %v", err)
	}
}

// recordingTransport implements cudacollective.Transport for tests. It moves
// no data; it records each execution and whether its buffers were held by
// the communicator at the time.
type recordingTransport struct {
	typ       cudacollective.TransportType
	err       error                              // returned from Execute when set
	onExecute func(*cudacollective.Communicator) // called from Execute when set

	mu    sync.Mutex
	execs []recordedExec
}

type recordedExec struct {
	rank       int
	op         cudacollective.OperationDescriptor
	send, recv driver.DevicePtr
	held       bool
}

func (t *recordingTransport) Type() cudacollective.TransportType {
	if t.typ == 0 {
		return cudacollective.TransportTypeControlPlaneTest
	}
	return t.typ
}

func (t *recordingTransport) Available(cudacollective.Membership) bool { return true }

func (t *recordingTransport) Execute(ctx context.Context, comm *cudacollective.Communicator, op cudacollective.OperationDescriptor, send, recv driver.DevicePtr) error {
	bufs := []driver.DevicePtr{send}
	if !op.InPlace {
		bufs = append(bufs, recv)
	}
	held := true
	for _, b := range bufs {
		tr, err := comm.AcquireTransferBuffer(uintptr(b), op.ExpectedBytes(), 0)
		if err == nil {
			comm.ReleaseTransferBuffer(tr, nil)
		}
		held = held && errors.Is(err, cudacollective.ErrBufferInUse)
	}
	t.mu.Lock()
	t.execs = append(t.execs, recordedExec{rank: comm.Rank(), op: op, send: send, recv: recv, held: held})
	t.mu.Unlock()
	if t.onExecute != nil {
		t.onExecute(comm)
	}
	return t.err
}

func (t *recordingTransport) Close() error { return nil }

func TestCollectiveLoweringExecutesRuntimeSteps(t *testing.T) {
	calls := collectiveCalls(2, 7)
	for r := range calls {
		for i := range calls[r] {
			calls[r][i].Root = 1
		}
		calls[r][1].InPlace = true
		calls[r][1].Recv = calls[r][1].Send
	}
	tr := new(recordingTransport)
	p, err := planCalls(t, calls, collectiveComms(t, 7, tr), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.execs) != 0 {
		t.Fatalf("Plan executed %d steps", len(tr.execs))
	}
	if _, err := p.Replay(context.Background(), collectiveInputs(2)); err != nil {
		t.Fatal(err)
	}
	var want []recordedExec
	for i := range 2 {
		for r := range 2 {
			c := calls[r][i]
			want = append(want, recordedExec{
				rank: r,
				op: cudacollective.OperationDescriptor{
					Kind: cudacollective.KindAllReduce, DataType: cudair.Float32, Count: 4, Root: 1,
					Reduction: cudacollective.ReductionSum, Topology: cudacollective.TopologyAny, InPlace: c.InPlace,
				},
				send: driver.DevicePtr(c.Send.Address + c.Send.Offset),
				recv: driver.DevicePtr(c.Recv.Address + c.Recv.Offset),
				held: true,
			})
		}
	}
	if !slices.Equal(tr.execs, want) {
		t.Fatalf("runtime executions:\n got %+v\nwant %+v", tr.execs, want)
	}
}

func TestReplayRefusalsKeepCommunicators(t *testing.T) {
	alive := func(t *testing.T, q *egraph.CollectiveQuarantine, comms []*cudacollective.Communicator, ids ...string) {
		t.Helper()
		for r, c := range comms {
			if err := c.CheckAlive(); err != nil {
				t.Fatalf("rank %d communicator failed: %v", r, err)
			}
		}
		for _, id := range ids {
			if err := q.Err(id); err != nil {
				t.Fatalf("plan quarantined: %v", err)
			}
		}
	}
	t.Run("buffer-held-elsewhere", func(t *testing.T) {
		q, comms, calls := new(egraph.CollectiveQuarantine), collectiveComms(t, 7, nil), collectiveCalls(1, 7)
		p, err := planCalls(t, calls, comms, q)
		if err != nil {
			t.Fatal(err)
		}
		hold, err := comms[0].AcquireTransferBuffer(uintptr(calls[0][0].Send.Address), 16, 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, cudacollective.ErrBufferInUse) {
			t.Fatalf("replay with held buffer: %v", err)
		}
		alive(t, q, comms, p.ID())
		comms[0].ReleaseTransferBuffer(hold, nil)
		if _, err := p.Replay(context.Background(), collectiveInputs(1)); err != nil {
			t.Fatalf("replay after release: %v", err)
		}
	})
	t.Run("replay-after-close", func(t *testing.T) {
		q, comms := new(egraph.CollectiveQuarantine), collectiveComms(t, 7, nil)
		p, err := planCalls(t, collectiveCalls(1, 7), comms, q)
		if err != nil {
			t.Fatal(err)
		}
		p.Close()
		if _, err := p.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, cudacollective.ErrCommunicatorClosed) {
			t.Fatalf("replay after close: %v", err)
		}
		alive(t, q, comms, p.ID())
		again, err := planCalls(t, collectiveCalls(1, 7), comms, q)
		if err != nil {
			t.Fatalf("re-plan after close: %v", err)
		}
		if _, err := again.Replay(context.Background(), collectiveInputs(1)); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("plans-sharing-communicators", func(t *testing.T) {
		q, comms := new(egraph.CollectiveQuarantine), collectiveComms(t, 7, nil)
		a, err := planCalls(t, collectiveCalls(1, 7), comms, q)
		if err != nil {
			t.Fatal(err)
		}
		other := collectiveCalls(1, 7)
		for r := range other {
			other[r][0].Root = 1
		}
		b, err := planCalls(t, other, comms, q)
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		errs := make(chan error, 200)
		for range 100 {
			for _, p := range []*egraph.CollectivePlan{a, b} {
				wg.Go(func() {
					if _, err := p.Replay(context.Background(), collectiveInputs(1)); err != nil {
						errs <- err
					}
				})
			}
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if !errors.Is(err, cudacollective.ErrBufferInUse) {
				t.Fatalf("contended replay: %v", err)
			}
		}
		alive(t, q, comms, a.ID(), b.ID())
		for _, p := range []*egraph.CollectivePlan{a, b} {
			if _, err := p.Replay(context.Background(), collectiveInputs(1)); err != nil {
				t.Fatalf("replay after contention: %v", err)
			}
		}
	})
	t.Run("cancelled-between-collectives", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		q := new(egraph.CollectiveQuarantine)
		tr := &recordingTransport{onExecute: func(*cudacollective.Communicator) { cancel() }}
		comms := collectiveComms(t, 7, tr)
		p, err := planCalls(t, collectiveCalls(2, 7), comms, q)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.Replay(ctx, collectiveInputs(2)); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled replay: %v", err)
		}
		if len(tr.execs) != 2 {
			t.Fatalf("executions after cancel = %d, want collective 0 on both ranks", len(tr.execs))
		}
		alive(t, q, comms, p.ID())
	})
	t.Run("communicators-failed-before-replay", func(t *testing.T) {
		q, comms := new(egraph.CollectiveQuarantine), collectiveComms(t, 7, nil)
		p, err := planCalls(t, collectiveCalls(1, 7), comms, q)
		if err != nil {
			t.Fatal(err)
		}
		comms[1].ReportLostParticipant(0)
		if _, err := p.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
			t.Fatalf("replay on failed communicators: %v", err)
		}
		if err := q.Err(p.ID()); err != nil {
			t.Fatalf("bystander plan quarantined: %v", err)
		}
		again, err := planCalls(t, collectiveCalls(1, 7), collectiveComms(t, 7, nil), q)
		if err != nil {
			t.Fatalf("re-plan on fresh communicators: %v", err)
		}
		if _, err := again.Replay(context.Background(), collectiveInputs(1)); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("closed-plan-failed-communicators", func(t *testing.T) {
		q, comms := new(egraph.CollectiveQuarantine), collectiveComms(t, 7, nil)
		p, err := planCalls(t, collectiveCalls(1, 7), comms, q)
		if err != nil {
			t.Fatal(err)
		}
		p.Close()
		comms[1].ReportLostParticipant(0)
		if _, err := p.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, cudacollective.ErrCommunicatorClosed) {
			t.Fatalf("replay of closed plan: %v", err)
		}
		if err := q.Err(p.ID()); err != nil {
			t.Fatalf("closed plan quarantined: %v", err)
		}
	})
	t.Run("communicator-closed-before-replay", func(t *testing.T) {
		q, comms := new(egraph.CollectiveQuarantine), collectiveComms(t, 7, nil)
		p, err := planCalls(t, collectiveCalls(1, 7), comms, q)
		if err != nil {
			t.Fatal(err)
		}
		comms[1].Close()
		if _, err := p.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, cudacollective.ErrCommunicatorClosed) {
			t.Fatalf("replay on closed communicator: %v", err)
		}
		if err := q.Err(p.ID()); err != nil {
			t.Fatalf("plan quarantined for a closed communicator: %v", err)
		}
	})
	t.Run("communicator-closed-during-first-rank", func(t *testing.T) {
		// Rank 0's communicator closes inside its own execution and the
		// transport reports it; no rank completed the collective.
		q := new(egraph.CollectiveQuarantine)
		tr := &recordingTransport{err: cudacollective.ErrCommunicatorClosed, onExecute: func(c *cudacollective.Communicator) { c.Close() }}
		comms := collectiveComms(t, 7, tr)
		p, err := planCalls(t, collectiveCalls(1, 7), comms, q)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.Replay(context.Background(), collectiveInputs(1)); !errors.Is(err, cudacollective.ErrCommunicatorClosed) {
			t.Fatalf("replay: %v", err)
		}
		if len(tr.execs) != 1 {
			t.Fatalf("executions = %d", len(tr.execs))
		}
		if err := q.Err(p.ID()); err != nil {
			t.Fatalf("plan quarantined for a closed communicator: %v", err)
		}
		if err := comms[1].CheckAlive(); err != nil {
			t.Fatalf("rank 1 failed: %v", err)
		}
	})
	t.Run("transport-failure", func(t *testing.T) {
		q := new(egraph.CollectiveQuarantine)
		comms := collectiveComms(t, 7, &recordingTransport{err: errors.New("link down")})
		p, err := planCalls(t, collectiveCalls(1, 7), comms, q)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.Replay(context.Background(), collectiveInputs(1)); err == nil {
			t.Fatal("transport failure ignored")
		}
		for r, c := range comms {
			if err := c.CheckAlive(); !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
				t.Fatalf("rank %d alive after transport failure: %v", r, err)
			}
		}
		if err := q.Err(p.ID()); !errors.Is(err, egraph.ErrCollectiveQuarantined) {
			t.Fatalf("transport failure not quarantined: %v", err)
		}
	})
}

func TestCollectivePlanEdges(t *testing.T) {
	var zero egraph.CollectivePlan
	if _, err := zero.Replay(context.Background(), nil); err == nil {
		t.Fatal("zero plan replayed")
	}

	calls := collectiveCalls(1, 7)
	for r := range calls {
		calls[r][0].InPlace = true
		calls[r][0].Recv = calls[r][0].Send
	}
	p, err := planCalls(t, calls, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	inputs := collectiveInputs(1)
	rec, err := p.Replay(context.Background(), inputs)
	if err != nil {
		t.Fatal(err)
	}
	want := slices.Clone(inputs[0][0])
	inputs[0][0][0], inputs[0][1][0] = -1, -1
	for r, out := range rec.Steps[0].Outputs {
		if !slices.Equal(out, want) {
			t.Fatalf("rank %d receipt output aliases caller input: %v", r, out)
		}
	}

	m2, err := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0, HostID: "h"}, {ID: 1, HostID: "other", DeviceID: 1}})
	if err != nil {
		t.Fatal(err)
	}
	foreign, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Rank: 1, Membership: m2, Generation: 7, Transport: cudacollective.NewDeterministicTestTransport(2)})
	if err != nil {
		t.Fatal(err)
	}
	defer foreign.Close()
	if _, err := planCalls(t, collectiveCalls(1, 7), []*cudacollective.Communicator{collectiveComms(t, 7, nil)[0], foreign}, nil); !errors.Is(err, egraph.ErrCollectiveMismatch) {
		t.Fatalf("ranks with different memberships: %v", err)
	}
	// Distinct instances of the same transport type are allowed.
	if _, err := planCalls(t, collectiveCalls(1, 7), []*cudacollective.Communicator{collectiveComms(t, 7, nil)[0], collectiveComms(t, 7, nil)[1]}, nil); err != nil {
		t.Fatalf("distinct transport instances of one type: %v", err)
	}

	mixed := collectiveComms(t, 7, nil)
	mixed[1] = collectiveComms(t, 7, &recordingTransport{typ: cudacollective.TransportTypeNCCLTest})[1]
	if _, err := planCalls(t, collectiveCalls(1, 7), mixed, nil); !errors.Is(err, egraph.ErrCollectiveMismatch) {
		t.Fatalf("ranks on different transports: %v", err)
	}
}
