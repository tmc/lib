package cudadiageval

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// successorFixture returns a ledger with a failed cell holding two failed
// attempts and a reference, and a pending cell without attempts.
func successorFixture(t testing.TB) QualificationLedger {
	t.Helper()
	digest := strings.Repeat("a", 64)
	closure := QualificationClosure{Candidate: strings.Repeat("1", 40), Base: strings.Repeat("1", 40), SignatureMapSHA256: digest, Tree: strings.Repeat("2", 40), BinarySHA256: digest, DependenciesSHA256: digest, InputsSHA256: digest, OutputsSHA256: digest, HostOS: "linux", HostArch: "amd64", Device: "none", Architecture: "none", ProducerVersion: "test-control", Libraries: map[string]string{}, Toolchain: "go-test", Topology: "offline", Workload: "host-control", Deployment: "test", Observer: "off"}
	host := QualificationCell{ID: "host-bpf", CapabilityID: "bpf-snapshot", Capability: "supported", Gate: "host", Status: "failed", Closure: closure, SourcePaths: []string{"ebpfhost/bpfprobe/probe.go"}, Consumers: []string{"nvidia/x/cmd/gpudiageval/main.go"}, Owner: "host-owner", NextAction: "rerun after fix"}
	host.Attempts = []QualificationAttempt{
		successorAttempt(host, "attempt-0.json", "1", "failed"),
		successorAttempt(host, "attempt-1.json", "2", "failed"),
	}
	host.References = []QualificationReference{{Path: "history/old.json", SHA256: strings.Repeat("b", 64), Scope: "earlier tree", Reason: "incompatible closure"}}
	native := QualificationCell{ID: "native-bpf", CapabilityID: "bpf-snapshot", Capability: "unknown", Gate: "native", Status: "pending", Closure: QualificationClosure{Candidate: strings.Repeat("1", 40), Tree: strings.Repeat("2", 40)}, SourcePaths: []string{"ebpfhost/bpfprobe/probe.go"}, Consumers: []string{"nvidia/x/cmd/gpudiageval/main.go"}, Owner: "native-owner", NextAction: "native admission required"}
	l := QualificationLedger{Version: 1, Cells: []QualificationCell{host, native}}
	if err := l.Validate(); err != nil {
		t.Fatal(err)
	}
	return l
}

func successorAttempt(c QualificationCell, path, digit, status string) QualificationAttempt {
	return QualificationAttempt{Path: path, SHA256: strings.Repeat(digit, 64), Evidence: QualificationEvidence{CellID: c.ID, Gate: c.Gate, Status: status, Closure: c.Closure, RawPath: "raw-" + path, RawSHA256: strings.Repeat(digit, 64), Command: []string{"go", "test", "./x/cudadiageval"}, Verifier: "independent-test-control"}}
}

func cloneLedger(t testing.TB, l QualificationLedger) QualificationLedger {
	t.Helper()
	b, err := json.Marshal(l)
	if err != nil {
		t.Fatal(err)
	}
	var c QualificationLedger
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

// qualify appends a passed attempt to host-bpf and marks the cell qualified.
func qualify(l *QualificationLedger) {
	c := &l.Cells[0]
	c.Attempts = append(c.Attempts, successorAttempt(*c, "attempt-2.json", "3", "passed"))
	c.Status = "qualified"
	c.NextAction = "retain receipts"
}

func TestCheckQualificationLedgerSuccessor(t *testing.T) {
	tests := []struct {
		name   string
		change func(prev, next *QualificationLedger)
		want   string // error substring; empty means accepted
	}{
		{"unchanged", func(prev, next *QualificationLedger) {}, ""},
		{"append passed after failed and register new cell", func(prev, next *QualificationLedger) {
			qualify(next)
			c := next.Cells[0]
			c.ID = "host-bpf-v2"
			c.Closure.Tree = strings.Repeat("4", 40)
			c.Status = "pending"
			c.Attempts = nil
			c.References = nil
			next.Cells = append(next.Cells, c)
		}, ""},
		{"closure of cell without attempts changes", func(prev, next *QualificationLedger) {
			next.Cells[1].Closure.Tree = strings.Repeat("5", 40)
		}, ""},
		{"reference added", func(prev, next *QualificationLedger) {
			next.Cells[1].References = append(next.Cells[1].References, QualificationReference{Path: "history/new.json", Scope: "earlier tree", Reason: "missing artifact"})
		}, ""},
		// Each negative passes a weaker check named in its comment.
		{"failed attempt dropped", func(prev, next *QualificationLedger) {
			// Cell IDs and attempt count are unchanged.
			qualify(next)
			next.Cells[0].Attempts = next.Cells[0].Attempts[1:]
			next.Cells[0].Attempts = append(next.Cells[0].Attempts, successorAttempt(next.Cells[0], "attempt-3.json", "4", "passed"))
		}, "attempt 0"},
		{"attempt digest replaced", func(prev, next *QualificationLedger) {
			// Attempt count and paths are unchanged.
			qualify(next)
			next.Cells[0].Attempts[0].SHA256 = strings.Repeat("9", 64)
		}, "attempt 0"},
		{"attempt evidence replaced", func(prev, next *QualificationLedger) {
			// Path and envelope digest are unchanged.
			next.Cells[0].Attempts[1].Evidence.Status = "withheld"
		}, "attempt 1"},
		{"attempts reordered", func(prev, next *QualificationLedger) {
			// The set of attempts is unchanged.
			a := next.Cells[0].Attempts
			a[0], a[1] = a[1], a[0]
		}, "attempt 0"},
		{"attempts truncated", func(prev, next *QualificationLedger) {
			next.Cells[0].Attempts = next.Cells[0].Attempts[:1]
		}, "attempts"},
		{"closure edited in place", func(prev, next *QualificationLedger) {
			// Attempts are an exact prefix.
			qualify(next)
			next.Cells[0].Closure.Observer = "on"
		}, "new cell"},
		{"capability edited in place", func(prev, next *QualificationLedger) {
			next.Cells[0].CapabilityID = "bpf-ringbuf"
		}, "new cell"},
		{"closure library edited in place", func(prev, next *QualificationLedger) {
			// Top-level closure fields are unchanged.
			next.Cells[0].Closure.Libraries = map[string]string{"libcuda.so": "580.65"}
		}, "new cell"},
		{"source paths edited in place", func(prev, next *QualificationLedger) {
			next.Cells[0].SourcePaths = []string{"ebpfhost/bpfprobe/probe_v2.go"}
		}, "new cell"},
		{"consumers edited in place", func(prev, next *QualificationLedger) {
			next.Cells[0].Consumers = append(next.Cells[0].Consumers, "nvidia/x/cmd/nvtelemetry/main.go")
		}, "new cell"},
		{"dependency added in place", func(prev, next *QualificationLedger) {
			next.Cells[0].DependsOn = []string{"native-bpf"}
		}, "new cell"},
		{"source paths of cell without attempts change", func(prev, next *QualificationLedger) {
			next.Cells[1].SourcePaths = []string{"ebpfhost/bpfprobe/probe_v2.go"}
			next.Cells[1].DependsOn = []string{"host-bpf"}
		}, ""},
		{"unknown reference gains digest in place", func(prev, next *QualificationLedger) {
			prev.Cells[1].References = []QualificationReference{{Path: "history/lost.json", Scope: "earlier tree", Reason: "missing artifact"}}
			next.Cells[1].References = []QualificationReference{{Path: "history/lost.json", SHA256: strings.Repeat("d", 64), Scope: "earlier tree", Reason: "recovered"}}
		}, "reference"},
		{"unknown reference kept beside recovered digest", func(prev, next *QualificationLedger) {
			lost := QualificationReference{Path: "history/lost.json", Scope: "earlier tree", Reason: "missing artifact"}
			prev.Cells[1].References = []QualificationReference{lost}
			next.Cells[1].References = []QualificationReference{{Path: "history/lost.json", SHA256: strings.Repeat("d", 64), Scope: "earlier tree", Reason: "recovered"}, lost}
		}, ""},
		{"references reordered", func(prev, next *QualificationLedger) {
			// References are a set; order is not history.
			extra := QualificationReference{Path: "history/new.json", Scope: "earlier tree", Reason: "missing artifact"}
			next.Cells[0].References = append([]QualificationReference{extra}, next.Cells[0].References...)
		}, ""},
		{"cell deleted", func(prev, next *QualificationLedger) {
			// Every cell in next exists in prev.
			next.Cells = next.Cells[:1]
		}, "native-bpf"},
		{"reference removed", func(prev, next *QualificationLedger) {
			next.Cells[0].References = nil
		}, "reference"},
		{"reference digest replaced", func(prev, next *QualificationLedger) {
			next.Cells[0].References[0].SHA256 = strings.Repeat("c", 64)
		}, "reference"},
		{"invalid previous", func(prev, next *QualificationLedger) {
			prev.Version = 2
		}, "previous"},
		{"invalid next", func(prev, next *QualificationLedger) {
			next.Cells[0].Owner = ""
		}, "next"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prev := successorFixture(t)
			next := cloneLedger(t, prev)
			tt.change(&prev, &next)
			err := CheckQualificationLedgerSuccessor(prev, next)
			switch {
			case tt.want == "" && err != nil:
				t.Fatalf("successor refused: %v", err)
			case tt.want != "" && err == nil:
				t.Fatalf("successor accepted, want error containing %q", tt.want)
			case tt.want != "" && !strings.Contains(err.Error(), tt.want):
				t.Fatalf("error %q does not contain %q", err, tt.want)
			}
		})
	}
}

func ExampleCheckQualificationLedgerSuccessor() {
	prev := pendingLedgerExample()
	next := pendingLedgerExample()
	next.Cells = append(next.Cells, next.Cells[0])
	next.Cells[1].ID = "nvml-h100-native-v2"
	fmt.Println(CheckQualificationLedgerSuccessor(prev, next))
	fmt.Println(CheckQualificationLedgerSuccessor(next, prev))
	// Output:
	// <nil>
	// ledger successor: cell "nvml-h100-native-v2" removed
}
