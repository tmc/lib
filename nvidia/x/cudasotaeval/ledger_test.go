package cudasotaeval

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func ledgerFixture(t *testing.T) (QualificationLedger, string) {
	t.Helper()
	root := t.TempDir()
	digest := strings.Repeat("a", 64)
	closure := QualificationClosure{Candidate: strings.Repeat("1", 40), Base: strings.Repeat("1", 40), SignatureMapSHA256: digest, Tree: strings.Repeat("2", 40), BinarySHA256: digest, DependenciesSHA256: digest, InputsSHA256: digest, OutputsSHA256: digest, HostOS: "linux", HostArch: "amd64", Device: "none", Architecture: "none", ProducerVersion: "test-control", Libraries: map[string]string{}, Toolchain: "go-test", Topology: "offline", Workload: "host-control", Deployment: "test", Observer: "off"}
	raw := []byte("independent host control output\n")
	rawDigest := fmt.Sprintf("%x", sha256.Sum256(raw))
	if err := os.WriteFile(filepath.Join(root, "raw.txt"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	evidence := QualificationEvidence{CellID: "host-bpf", Gate: "host", Status: "passed", Closure: closure, RawPath: "raw.txt", RawSHA256: rawDigest, Command: []string{"go", "test", "./x/cudasotaeval"}, Verifier: "independent-test-control"}
	b, err := json.Marshal(evidence)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "receipt.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	l := QualificationLedger{Version: 1, Cells: []QualificationCell{{ID: "host-bpf", CapabilityID: "bpf-snapshot", Capability: "supported", Gate: "host", Status: "qualified", Closure: closure, SourcePaths: []string{"ebpfhost/bpfprobe/probe.go"}, Consumers: []string{"nvidia/x/cmd/gpudiageval/main.go"}, Owner: "host-owner", NextAction: "retain receipt", Attempts: []QualificationAttempt{{Path: "receipt.json", SHA256: fmt.Sprintf("%x", sha256.Sum256(b)), Evidence: evidence}}}}}
	return l, root
}
func TestQualificationLedgerScopeMutations(t *testing.T) {
	cases := []struct {
		name   string
		change func(*QualificationLedger)
	}{
		{"tree", func(l *QualificationLedger) { l.Cells[0].Closure.Tree = strings.Repeat("3", 40) }},
		{"binary", func(l *QualificationLedger) { l.Cells[0].Closure.BinarySHA256 = strings.Repeat("3", 64) }},
		{"dependency", func(l *QualificationLedger) { l.Cells[0].Closure.DependenciesSHA256 = strings.Repeat("3", 64) }},
		{"architecture", func(l *QualificationLedger) { l.Cells[0].Closure.HostArch = "arm64" }},
		{"producer version", func(l *QualificationLedger) { l.Cells[0].Closure.ProducerVersion = "new-version" }},
		{"library", func(l *QualificationLedger) {
			l.Cells[0].Closure.Libraries = map[string]string{"libcuda.so": "changed"}
		}},
		{"signature map", func(l *QualificationLedger) { l.Cells[0].Closure.SignatureMapSHA256 = strings.Repeat("3", 64) }},
		{"receipt digest", func(l *QualificationLedger) { l.Cells[0].Attempts[0].SHA256 = strings.Repeat("3", 64) }},
		{"raw output", func(l *QualificationLedger) { l.Cells[0].Attempts[0].Evidence.RawSHA256 = strings.Repeat("3", 64) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l, root := ledgerFixture(t)
			tc.change(&l)
			r, err := VerifyQualificationLedger(context.Background(), root, l)
			if err != nil {
				t.Fatal(err)
			}
			if r[0].Accepted || r[0].Status != "pending" {
				t.Fatalf("mutated closure accepted: %+v", r)
			}
		})
	}
}
func TestQualificationLedgerControls(t *testing.T) {
	l, root := ledgerFixture(t)
	r, err := VerifyQualificationLedger(context.Background(), root, l)
	if err != nil || !r[0].Accepted {
		t.Fatalf("positive: %v %+v", err, r)
	}
	native := l.Cells[0]
	native.ID = "native-bpf"
	native.Gate = "native"
	native.Status = "pending"
	native.Attempts = nil
	native.NextAction = "native admission required"
	native.Closure.BinarySHA256 = ""
	l.Cells = append(l.Cells, native)
	r, err = VerifyQualificationLedger(context.Background(), root, l)
	if err != nil {
		t.Fatal(err)
	}
	if !r[0].Accepted || r[1].Accepted || r[1].Status != "pending" {
		t.Fatalf("host/native conflated: %+v", r)
	}
	var b bytes.Buffer
	if err := WriteQualificationLedger(&b, l); err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadQualificationLedger(&b)
	if err != nil || len(decoded.Cells) != 2 {
		t.Fatalf("round trip: %v", err)
	}
	if err := os.Remove(filepath.Join(root, "raw.txt")); err != nil {
		t.Fatal(err)
	}
	r, err = VerifyQualificationLedger(context.Background(), root, l)
	if err != nil || r[0].Accepted {
		t.Fatalf("missing raw accepted: %v %+v", err, r)
	}
}
func TestQualificationLedgerStructure(t *testing.T) {
	cases := []struct {
		name   string
		change func(*QualificationLedger)
	}{
		{"duplicate", func(l *QualificationLedger) { l.Cells = append(l.Cells, l.Cells[0]) }},
		{"cycle", func(l *QualificationLedger) { l.Cells[0].DependsOn = []string{l.Cells[0].ID} }},
		{"unknown dependency", func(l *QualificationLedger) { l.Cells[0].DependsOn = []string{"absent"} }},
		{"traversal", func(l *QualificationLedger) { l.Cells[0].Attempts[0].Path = "../receipt.json" }},
		{"gate", func(l *QualificationLedger) { l.Cells[0].Gate = "native" }},
		{"unsupported pass", func(l *QualificationLedger) { l.Cells[0].Capability = "unsupported" }},
		{"missing closure", func(l *QualificationLedger) { l.Cells[0].Closure.Tree = "" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l, _ := ledgerFixture(t)
			tc.change(&l)
			if err := l.Validate(); err == nil {
				t.Fatal("invalid ledger accepted")
			}
		})
	}
}
func TestQualificationLedgerParsing(t *testing.T) {
	l, _ := ledgerFixture(t)
	b, _ := json.Marshal(l)
	for _, data := range []string{`{"version":1,"version":1,"cells":[]}`, `{"version":1,"Version":1,"cells":[]}`, `{"Version":1,"cells":[]}`, string(b) + ` {}`, `{"version":1,"cells":[],"invented":true}`, strings.Repeat(" ", ledgerLimit+1)} {
		if _, err := ReadQualificationLedger(strings.NewReader(data)); err == nil {
			t.Fatal("invalid input accepted")
		}
	}
	if _, err := ReadQualificationLedger(nil); err == nil {
		t.Fatal("nil reader")
	}
	if err := WriteQualificationLedger(ledgerShortWriter{}, l); err != io.ErrShortWrite {
		t.Fatalf("short write: %v", err)
	}
}

type ledgerShortWriter struct{}

func (ledgerShortWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }
func TestQualificationLedgerArtifactRoot(t *testing.T) {
	l, root := ledgerFixture(t)
	outside := t.TempDir()
	raw, _ := os.ReadFile(filepath.Join(root, "raw.txt"))
	if err := os.WriteFile(filepath.Join(outside, "raw.txt"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, "raw.txt")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(outside, "raw.txt"), filepath.Join(root, "raw.txt")); err != nil {
		t.Fatal(err)
	}
	r, err := VerifyQualificationLedger(context.Background(), root, l)
	if err != nil || r[0].Accepted {
		t.Fatalf("escaped root: %v %+v", err, r)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := VerifyQualificationLedger(ctx, root, l); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
}
func TestQualificationLedgerFailedAttemptRetained(t *testing.T) {
	l, root := ledgerFixture(t)
	a := l.Cells[0].Attempts[0]
	a.Path = "failed.json"
	a.Evidence.Status = "failed"
	b, _ := json.Marshal(a.Evidence)
	a.SHA256 = fmt.Sprintf("%x", sha256.Sum256(b))
	if err := os.WriteFile(filepath.Join(root, a.Path), b, 0600); err != nil {
		t.Fatal(err)
	}
	l.Cells[0].Attempts = append(l.Cells[0].Attempts, a)
	r, err := VerifyQualificationLedger(context.Background(), root, l)
	if err != nil || r[0].Accepted || r[0].Status != "pending" {
		t.Fatalf("earlier pass erased later failure: %v %+v", err, r)
	}
}
func TestQualificationLedgerDependencies(t *testing.T) {
	l, root := ledgerFixture(t)
	dep := l.Cells[0]
	dep.ID = "native-prerequisite"
	dep.Status = "pending"
	dep.Attempts = nil
	l.Cells[0].DependsOn = []string{dep.ID}
	l.Cells = append(l.Cells, dep)
	r, err := VerifyQualificationLedger(context.Background(), root, l)
	if err != nil || r[0].Accepted || r[0].Reason != "dependency evidence not accepted" {
		t.Fatalf("dependency: %v %+v", err, r)
	}
}

func TestQualificationLedgerHistoricalReference(t *testing.T) {
	l, root := ledgerFixture(t)
	cell := l.Cells[0]
	cell.ID = "historical-native"
	cell.Gate = "native"
	cell.Status = "pending"
	cell.Capability = "unknown"
	cell.Attempts = nil
	cell.References = []QualificationReference{{Path: "raw.txt", SHA256: l.Cells[0].Attempts[0].Evidence.RawSHA256, Scope: "original host control", Reason: "native source and target closure absent"}}
	l.Cells = append(l.Cells, cell)
	r, err := VerifyQualificationLedger(context.Background(), root, l)
	if err != nil {
		t.Fatal(err)
	}
	if r[0].Accepted || r[0].ReferencesVerified != 1 || r[0].Status != "pending" {
		t.Fatalf("historical bytes promoted qualification: %+v", r[0])
	}
}
