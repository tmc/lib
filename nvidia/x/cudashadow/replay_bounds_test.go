package cudashadow_test

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashadow"
)

// maxHistoryBytes is the advertised 16 MiB history bound.
const maxHistoryBytes = 16 << 20

var ptxInfo = cudashadow.PolicyInfo{
	Name:       "ptx-shadow",
	Version:    "1",
	Domain:     cudashadow.DomainPTXInstrument,
	Parameters: map[string]string{"max_entries": "8"},
	Seed:       7,
}

// A countingPolicy counts calls to the built-in PTX shadow policy. The same
// object embeds an active backend spy, so any actuator call reached through
// the policy is recorded.
type countingPolicy struct {
	cudashadow.Policy
	*cudashadow.ActiveBackendSpy
	infos, evals int
}

func newCountingPolicy(t *testing.T) *countingPolicy {
	t.Helper()
	p, err := cudashadow.NewPTXShadowPolicy(ptxInfo)
	if err != nil {
		t.Fatal(err)
	}
	c := &countingPolicy{Policy: p, ActiveBackendSpy: cudashadow.NewActiveBackendSpy()}
	t.Cleanup(func() {
		if err := c.AssertZeroMutations(); err != nil {
			t.Error(err)
		}
	})
	return c
}

func (p *countingPolicy) Info() cudashadow.PolicyInfo {
	p.infos++
	return p.Policy.Info()
}

func (p *countingPolicy) Evaluate(ctx context.Context, in cudashadow.Input) (cudashadow.Decision, error) {
	p.evals++
	return p.Policy.Evaluate(ctx, in)
}

// ptxDecision returns the decision the PTX shadow policy makes for every
// snapshot built by ptxSnapshot.
func ptxDecision(t *testing.T) cudashadow.Decision {
	t.Helper()
	p, err := cudashadow.NewPTXShadowPolicy(ptxInfo)
	if err != nil {
		t.Fatal(err)
	}
	d, err := p.Evaluate(context.Background(), ptxSnapshot(0, cudashadow.Decision{}, nil).Input)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// ptxSnapshot returns the valid snapshot i of a PTX shadow history. Its
// encoded size does not depend on i. A non-nil details records an outcome.
func ptxSnapshot(i int, d cudashadow.Decision, details map[string]string) cudashadow.Snapshot {
	seq := uint64(1_000_000 + i)
	p := cudashadow.Provenance{DriverVersion: "575.57.08", KernelModule: "nvidia.ko", Hook: cudashadow.HookPTXInstrument, Sequence: seq, TimestampNano: int64(seq)}
	s := cudashadow.Snapshot{
		ID:         fmt.Sprintf("snap-%08d", i),
		Provenance: p,
		Policy:     ptxInfo,
		Input:      cudashadow.Input{Provenance: p, Domain: cudashadow.DomainPTXInstrument, PTX: &cudashadow.PTXData{KernelName: "kernel", TargetArch: "sm_80", CodeSize: 64}},
		Decision:   d,
		Status:     cudashadow.StatusValid,
	}
	if details != nil {
		s.ObservedOutcome = &cudashadow.ObservedOutcome{TimestampNano: int64(seq), Action: "baseline", Details: details}
	}
	return s
}

// ptxHistory returns n valid snapshots sharing one decision and details.
func ptxHistory(t *testing.T, n int, details map[string]string) []cudashadow.Snapshot {
	t.Helper()
	d := ptxDecision(t)
	history := make([]cudashadow.Snapshot, n)
	for i := range history {
		history[i] = ptxSnapshot(i, d, details)
	}
	return history
}

// escaped is a 4095-byte detail value that encodes to 9360 bytes:
// encoding/json writes '<' and U+2028 as six-byte escapes and '"' as \".
var escaped = strings.Repeat("<\"\u2028é", 585)

// bulkDetails returns the largest valid outcome details: 64 entries, each
// sharing escaped. A snapshot recording them encodes to about 600 kB.
func bulkDetails() map[string]string {
	m := make(map[string]string)
	for i := range 64 {
		m[fmt.Sprintf("d%02d", i)] = escaped
	}
	return m
}

func encodedLen(t *testing.T, s cudashadow.Snapshot) int {
	t.Helper()
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	return len(data)
}

// pad sets the outcome details of s so that s encodes to exactly n bytes.
func pad(t *testing.T, s cudashadow.Snapshot, n int) cudashadow.Snapshot {
	t.Helper()
	details := map[string]string{"p00": ""}
	s.ObservedOutcome.Details = details
	rest := n - encodedLen(t, s)
	for i := 0; rest > 0; i++ {
		if i > 0 {
			rest -= len(`,"p00":""`)
		}
		// '<' encodes as six bytes and 'a' as one; a value holds 4096 bytes.
		m := rest
		if m > 6*4091 {
			m = min(6*4091, rest-10)
		}
		details[fmt.Sprintf("p%02d", i)] = strings.Repeat("<", m/6) + strings.Repeat("a", m%6)
		rest -= m
	}
	if got := encodedLen(t, s); got != n {
		t.Fatalf("padded snapshot encodes to %d bytes, want %d", got, n)
	}
	return s
}

// sizedHistory returns a valid version 1 history whose compact encoding is
// exactly total bytes. All but the last snapshot share bulkDetails.
func sizedHistory(t *testing.T, total int) []cudashadow.Snapshot {
	t.Helper()
	const frame = len(`{"version":1,"snapshots":[]}`)
	d := ptxDecision(t)
	details := bulkDetails()
	row := encodedLen(t, ptxSnapshot(0, d, details))
	k := (total - frame - 4096) / (row + 1)
	history := make([]cudashadow.Snapshot, k+1)
	for i := range k {
		history[i] = ptxSnapshot(i, d, details)
	}
	history[k] = pad(t, ptxSnapshot(k, d, map[string]string{}), total-frame-k*(row+1))
	return history
}

// overHistory returns the exact 16 MiB history with one more encoded byte.
func overHistory(t *testing.T) []cudashadow.Snapshot {
	t.Helper()
	history := sizedHistory(t, maxHistoryBytes)
	history[len(history)-1].ID += "x"
	return history
}

// TestReplayRefusesOversizedHistory is the predecessor witness: before the
// history admission, Replay evaluated every snapshot of a 4097-snapshot
// history.
func TestReplayRefusesOversizedHistory(t *testing.T) {
	history := ptxHistory(t, 4097, nil)
	p := newCountingPolicy(t)
	e, err := cudashadow.NewEvaluator(1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Replay(context.Background(), p, history)
	if err == nil || got != nil {
		t.Errorf("Replay(4097 snapshots) = %d snapshots, %v; want refusal", len(got), err)
	}
	if p.infos != 0 || p.evals != 0 {
		t.Errorf("Replay(4097 snapshots) called Info %d and Evaluate %d times; want none", p.infos, p.evals)
	}
}

func TestReplayHistoryBounds(t *testing.T) {
	tests := []struct {
		name    string
		history func(*testing.T) []cudashadow.Snapshot
		err     string // empty if admitted
	}{
		{"4095", func(t *testing.T) []cudashadow.Snapshot { return ptxHistory(t, 4095, nil) }, ""},
		{"4096", func(t *testing.T) []cudashadow.Snapshot { return ptxHistory(t, 4096, nil) }, ""},
		{"4097", func(t *testing.T) []cudashadow.Snapshot { return ptxHistory(t, 4097, nil) }, "capacity exceeded"},
		{"16MiB", func(t *testing.T) []cudashadow.Snapshot { return sizedHistory(t, maxHistoryBytes) }, ""},
		{"16MiB+1", overHistory, "exceeds 16 MiB"},
		// 4096 snapshots sharing about 600 kB of details encode to 2.3 GiB.
		{"shared", func(t *testing.T) []cudashadow.Snapshot { return ptxHistory(t, 4096, bulkDetails()) }, "exceeds 16 MiB"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			history := tt.history(t)
			p := newCountingPolicy(t)
			e, err := cudashadow.NewEvaluator(1)
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Replay(context.Background(), p, history)
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) || got != nil {
					t.Errorf("Replay = %d snapshots, %v; want error containing %q", len(got), err, tt.err)
				}
				if p.infos != 0 || p.evals != 0 {
					t.Errorf("refused Replay called Info %d and Evaluate %d times; want none", p.infos, p.evals)
				}
				return
			}
			if err != nil || len(got) != len(history) {
				t.Fatalf("Replay = %d snapshots, %v; want %d", len(got), err, len(history))
			}
			if p.infos != len(history) || p.evals != len(history) {
				t.Errorf("Replay called Info %d and Evaluate %d times; want %d", p.infos, p.evals, len(history))
			}
		})
	}
}

// TestReplayValidationPrecedence pins the documented precedence: an invalid
// snapshot is reported before cancellation, identity mismatch or
// evaluation at an earlier step, and before any policy call.
func TestReplayValidationPrecedence(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name   string
		ctx    context.Context
		change func(*cudashadow.Snapshot)
	}{
		{"evaluate", context.Background(), func(*cudashadow.Snapshot) {}},
		{"identity", context.Background(), func(s *cudashadow.Snapshot) { s.Policy.Name = "other" }},
		{"canceled", canceled, func(*cudashadow.Snapshot) {}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			history := ptxHistory(t, 3, nil)
			tt.change(&history[0])
			history[2].ID = ""
			p := newCountingPolicy(t)
			e, err := cudashadow.NewEvaluator(1)
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Replay(tt.ctx, p, history)
			if err == nil || !strings.HasPrefix(err.Error(), "cudashadow replay step 2: ") || got != nil {
				t.Errorf("Replay = %d snapshots, %v; want step 2 validation error", len(got), err)
			}
			if p.infos != 0 || p.evals != 0 {
				t.Errorf("Replay called Info %d and Evaluate %d times; want none", p.infos, p.evals)
			}
		})
	}
}

func TestReplayParity(t *testing.T) {
	d := ptxDecision(t)
	history := []cudashadow.Snapshot{
		ptxSnapshot(0, d, map[string]string{"control": "baseline"}),
		ptxSnapshot(1, d, nil),
		ptxSnapshot(2, cudashadow.Decision{Type: cudashadow.DecisionSkip, Reason: "stale input: replayed"}, nil),
		ptxSnapshot(3, d, map[string]string{"escape": "<&>\u2028\""}),
	}
	history[2].Status = cudashadow.StatusStale
	for _, tt := range []struct {
		name    string
		history []cudashadow.Snapshot
		evals   int
	}{
		{"nil", nil, 0},
		{"empty", []cudashadow.Snapshot{}, 0},
		{"mixed", history, 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := newCountingPolicy(t)
			e, err := cudashadow.NewEvaluator(1)
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Replay(context.Background(), p, tt.history)
			if err != nil || len(got) != len(tt.history) || len(got) > 0 && !reflect.DeepEqual(got, tt.history) {
				t.Fatalf("Replay = %+v, %v; want %+v", got, err, tt.history)
			}
			if p.infos != len(tt.history) || p.evals != tt.evals {
				t.Errorf("Replay called Info %d and Evaluate %d times; want %d and %d", p.infos, p.evals, len(tt.history), tt.evals)
			}
			if len(got) == 0 {
				return
			}
			got[0].ObservedOutcome.Details["control"] = "changed"
			got[0].Decision.Parameters["offset"] = 1
			got[0].Policy.Parameters["max_entries"] = "1"
			got[0].Input.PTX.KernelName = "changed"
			if tt.history[0].ObservedOutcome.Details["control"] != "baseline" || d.Parameters["offset"] != 0 || ptxInfo.Parameters["max_entries"] != "8" || tt.history[0].Input.PTX.KernelName != "kernel" {
				t.Error("replayed snapshots alias the history")
			}
		})
	}
}
