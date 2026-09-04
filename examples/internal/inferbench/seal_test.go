package inferbench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPathManifestRejectsModelSubstitution(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	manifest, err := ManifestPath(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyPathManifest(dir, manifest); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("substitute"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := VerifyPathManifest(dir, manifest); err == nil {
		t.Fatal("VerifyPathManifest accepted substituted model")
	}
}

func TestBuildReceiptRejectsBinarySubstitution(t *testing.T) {
	h := strings.Repeat("a", 64)
	receipt := BuildReceipt{BinarySHA256: h, SourceSHA256: h, DependenciesSHA256: h, Toolchain: "go1.27", Command: []string{"go", "build"}}
	if err := receipt.Validate(); err != nil {
		t.Fatal(err)
	}
	receipt.BinarySHA256 = "stale"
	if err := receipt.Validate(); err == nil {
		t.Fatal("BuildReceipt accepted substituted binary")
	}
}

func TestSealManifestRejectsMissingMutationInput(t *testing.T) {
	h := strings.Repeat("a", 64)
	manifest := SealManifest{
		Build:         BuildReceipt{BinarySHA256: h, SourceSHA256: h, DependenciesSHA256: h, Toolchain: "go1.27", Command: []string{"go", "build"}},
		Model:         PathManifest{SHA256: h, Files: []FileArtifact{{Name: "model.safetensors", Mode: "-rw-------", Size: 1, SHA256: h}}},
		WeightsSHA256: h, Mutation: MutationProbe{ComparatorSHA256: h, ExpectedSHA256: h}, SentinelSHA256: h,
	}
	if err := manifest.Validate(); err == nil {
		t.Fatal("SealManifest accepted missing mutation input")
	}
}

func TestSentinelChainRejectsPosthocInsertion(t *testing.T) {
	dir := t.TempDir()
	base := time.Unix(100, 0).UTC()
	for i := 0; i < 20; i++ {
		raw := IntervalReceipt{Index: i, Sample: Sample{StartedAt: base.Add(time.Duration(3*i) * time.Second), EndedAt: base.Add(time.Duration(3*i+1) * time.Second)}}
		if err := WriteJSONAtomic(filepath.Join(dir, formatInterval(i)), raw); err != nil {
			t.Fatal(err)
		}
	}
	log := filepath.Join(dir, "sentinels.jsonl")
	previous := ""
	for sequence, index := range []int{9, 19} {
		raw, err := HashFile(filepath.Join(dir, formatInterval(index)))
		if err != nil {
			t.Fatal(err)
		}
		record := SentinelRecord{Sequence: sequence, AfterIndex: index, CapturedAt: base.Add(time.Duration(3*index+2) * time.Second), DeviceID: "gpu0", PredecessorSHA256: raw, PreviousSHA256: previous, Duration: time.Millisecond}
		if err := AppendSentinel(log, record); err != nil {
			t.Fatal(err)
		}
		records, err := ReadSentinelChain(log, dir, "gpu0", 10, index+1)
		if err != nil {
			t.Fatal(err)
		}
		previous = records[len(records)-1].SHA256
	}
	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if err := os.WriteFile(log, []byte(lines[0]+"\n"+lines[0]+"\n"+lines[1]+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadSentinelChain(log, dir, "gpu0", 10, 20); err == nil {
		t.Fatal("ReadSentinelChain accepted inserted sentinel")
	}
}

func TestSentinelChainRejectsPosthocTimestamp(t *testing.T) {
	dir := t.TempDir()
	base := time.Unix(100, 0).UTC()
	for i := 0; i < 20; i++ {
		raw := IntervalReceipt{Index: i, Sample: Sample{StartedAt: base.Add(time.Duration(3*i) * time.Second), EndedAt: base.Add(time.Duration(3*i+1) * time.Second)}}
		if err := WriteJSONAtomic(filepath.Join(dir, formatInterval(i)), raw); err != nil {
			t.Fatal(err)
		}
	}
	log := filepath.Join(dir, "sentinels.jsonl")
	raw, err := HashFile(filepath.Join(dir, formatInterval(9)))
	if err != nil {
		t.Fatal(err)
	}
	// Recompute a self-consistent record whose timestamp reveals that it was
	// created after the successor interval had already started.
	record := SentinelRecord{Sequence: 0, AfterIndex: 9, CapturedAt: base.Add(32 * time.Second), DeviceID: "gpu0", PredecessorSHA256: raw, Duration: time.Millisecond}
	if err := AppendSentinel(log, record); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadSentinelChain(log, dir, "gpu0", 10, 20); err == nil {
		t.Fatal("ReadSentinelChain accepted posthoc timestamp")
	}
}

func formatInterval(i int) string {
	const digits = "0123456789"
	b := []byte("000000.json")
	for p := 5; p >= 0; p-- {
		b[p] = digits[i%10]
		i /= 10
	}
	return string(b)
}
