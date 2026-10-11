package cudatracehost

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func TestDiscoveredThreads(t *testing.T) {
	ids := []int{19, 7}
	threads, err := discoveredThreads(ids, func(id int) (uint64, error) { return uint64(id + 100), nil }, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []ThreadSnapshot{{7, 107}, {19, 119}}
	if !reflect.DeepEqual(threads, want) || !reflect.DeepEqual(ids, []int{19, 7}) {
		t.Fatalf("threads=%v ids=%v", threads, ids)
	}
	for _, tt := range []struct {
		name       string
		ids        []int
		limit      int
		generation uint64
		err        error
	}{
		{"empty", nil, 2, 1, nil}, {"bound", []int{1, 2, 3}, 2, 1, nil}, {"duplicate", []int{1, 1}, 2, 1, nil}, {"invalid tid", []int{0}, 2, 1, nil}, {"unknown birth", []int{1}, 2, 0, nil}, {"exited", []int{1}, 2, 1, errors.New("task exited")},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := discoveredThreads(tt.ids, func(int) (uint64, error) { return tt.generation, tt.err }, tt.limit)
			if err == nil {
				t.Fatal("invalid snapshot accepted")
			}
		})
	}
}

func TestVerifyThreads(t *testing.T) {
	threads := []ThreadSnapshot{{7, 107}, {19, 119}}
	for _, tt := range []struct {
		name    string
		changed bool
		exited  bool
		wantErr bool
	}{
		{"same", false, false, false}, {"recycled tid", true, false, true}, {"exited during attachment", false, true, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := verifyThreads(threads, func(id int) (uint64, error) {
				if id == 19 && tt.exited {
					return 0, errors.New("exited")
				}
				if id == 19 && tt.changed {
					return 120, nil
				}
				return uint64(id + 100), nil
			})
			if (err != nil) != tt.wantErr {
				t.Fatalf("verify=%v", err)
			}
		})
	}
}

func TestManifestSnapshotCopyAndSerialization(t *testing.T) {
	c := &Collector{manifest: Manifest{ThreadCoverage: SnapshotCoverage{
		Status: "generation_verified_attachment_snapshot", Clock: "elapsed_since_discovery",
		DiscoveryEndNanos: 10, AttachmentStartNanos: 11, AttachmentEndNanos: 20, VerificationEndNanos: 22,
		Threads: []ThreadSnapshot{{7, 107}}, FutureThreads: "unobserved", BlindInterval: "discovery_through_attachment",
	}}}
	m := c.Manifest()
	m.ThreadCoverage.Threads[0].Generation = 999
	if got := c.Manifest().ThreadCoverage.Threads[0].Generation; got != 107 {
		t.Fatalf("manifest aliases snapshot: %d", got)
	}
	data, err := json.Marshal(c.Manifest())
	if err != nil {
		t.Fatal(err)
	}
	var decoded Manifest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, c.Manifest()) {
		t.Fatalf("snapshot lost on public manifest round-trip: %s", data)
	}
	if decoded.Complete || decoded.ThreadCoverage.FutureThreads != "unobserved" {
		t.Fatal("snapshot promoted to complete coverage")
	}
}

func TestSnapshotBoundRefusesBeforeGenerationRead(t *testing.T) {
	calls := 0
	_, err := discoveredThreads([]int{1, 2, 3}, func(int) (uint64, error) { calls++; return 1, nil }, 2)
	if err == nil || calls != 0 {
		t.Fatalf("err=%v generation reads=%d", err, calls)
	}
}

func TestLegacyManifestCoverageUnknown(t *testing.T) {
	var m Manifest
	if err := json.Unmarshal([]byte(`{"ThreadScope":"attachment_snapshot","Complete":false}`), &m); err != nil {
		t.Fatal(err)
	}
	if m.ThreadCoverage.Status != "" || len(m.ThreadCoverage.Threads) != 0 || m.Complete {
		t.Fatal("legacy snapshot gained coverage")
	}
}
