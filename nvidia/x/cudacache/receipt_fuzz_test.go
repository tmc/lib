package cudacache_test

import (
	"bytes"
	"context"
	"encoding/json"
	"maps"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func FuzzReceiptEnvelope(f *testing.F) {
	const receiptLimit = 4 << 10
	const inputLimit = 8 << 10
	const workload = "receipt-fuzz-workload"
	policy := cudacache.SearchPolicyID{
		Strategy: "beam", Seed: 7, RuleSet: "receipt-fuzz", ScoringEpoch: "epoch-1",
	}
	key := cudacache.SearchReceiptKey(policy, workload)
	read := func(t testing.TB, data []byte) (*cudacache.SearchReceipt, error) {
		t.Helper()
		root := t.TempDir()
		dir := filepath.Join(root, "receipts")
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, key+".receipt"), data, 0o600); err != nil {
			t.Fatal(err)
		}
		// A new Cache must read this iteration's disk bytes, not a prior hit.
		c, err := cudacache.New(cudacache.Config{RootDir: root, MaxArtifactBytes: receiptLimit})
		if err != nil {
			t.Fatal(err)
		}
		return c.GetReceipt(context.Background(), policy, workload)
	}

	// Obtain the valid envelope through the public writer. The timing/rank
	// values are synthetic persistence data, not a measurement or qualification.
	root := f.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: root, MaxArtifactBytes: receiptLimit})
	if err != nil {
		f.Fatal(err)
	}
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	seedReceipt := &cudacache.SearchReceipt{
		PolicyID: policy, WorkloadKey: workload,
		CandidateID: "candidate-a", ArtifactKey: "artifact-a", Complete: true,
		Started: start, Finished: start.Add(time.Second),
		TimingKind: cudatrial.TimingWallElapsed,
		Timings:    map[string]int64{string(cudatrial.TimingWallElapsed): 123},
		RankScore:  4.25, Epoch: "epoch-1",
	}
	if err := c.PutReceipt(context.Background(), seedReceipt); err != nil {
		f.Fatal(err)
	}
	valid, err := os.ReadFile(filepath.Join(root, "receipts", key+".receipt"))
	if err != nil {
		f.Fatal(err)
	}
	if len(valid) >= receiptLimit {
		f.Fatal("valid seed leaves no space for the byte-limit control")
	}
	if got, err := read(f, valid); err != nil || got == nil || got.CandidateID != seedReceipt.CandidateID {
		f.Fatalf("public writer/fresh-reader seed failed: %v, %v", got, err)
	}
	changed := func(field string, value json.RawMessage) []byte {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(valid, &fields); err != nil {
			f.Fatal(err)
		}
		if value == nil {
			delete(fields, field)
		} else {
			fields[field] = value
		}
		data, err := json.Marshal(fields)
		if err != nil {
			f.Fatal(err)
		}
		return data
	}
	wrongWorkload := changed("workload_key", json.RawMessage(`"other-workload"`))
	if _, err := read(f, wrongWorkload); err == nil {
		f.Fatal("receipt replayed under another workload was admitted")
	}
	for _, seed := range [][]byte{
		valid,
		wrongWorkload,
		{},
		[]byte("null"),
		valid[:len(valid)/2],
		changed("schema_version", nil),
		changed("schema_version", json.RawMessage("2")),
		changed("candidate_id", json.RawMessage(`""`)),
		changed("timings", json.RawMessage(`{"wall_elapsed":-1}`)),
		changed("finished", json.RawMessage(`"2025-12-31T23:59:59Z"`)),
		// Empty and nil option maps have the same policy identity.
		changed("policy_id", json.RawMessage(`{"strategy":"beam","options":{},"seed":7,"rule_set":"receipt-fuzz","scoring_epoch":"epoch-1"}`)),
		append(bytes.Clone(valid), bytes.Repeat([]byte(" "), receiptLimit+1-len(valid))...),
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > inputLimit {
			return
		}
		got, err := read(t, data)
		if err != nil {
			return // Refusal may return partial data; no stronger contract is imposed.
		}
		if got == nil {
			t.Fatal("successful receipt read returned nil")
		}
		if len(data) > receiptLimit {
			t.Fatal("successful read exceeded the configured byte limit")
		}
		// Decode independently with standard JSON. Do not call the SUT's
		// Validate or use its hashed key as the semantic admission oracle.
		var wire struct {
			Version *int `json:"schema_version"`
			Policy  struct {
				Strategy string            `json:"strategy"`
				Options  map[string]string `json:"options"`
				Seed     int64             `json:"seed"`
				RuleSet  string            `json:"rule_set"`
				Epoch    string            `json:"scoring_epoch"`
			} `json:"policy_id"`
			Workload  string           `json:"workload_key"`
			Candidate string           `json:"candidate_id"`
			Artifact  string           `json:"artifact_key"`
			Complete  bool             `json:"complete"`
			Started   time.Time        `json:"started"`
			Finished  time.Time        `json:"finished"`
			Timing    string           `json:"timing_kind"`
			Timings   map[string]int64 `json:"timings"`
			Rank      float64          `json:"rank_score"`
			Epoch     string           `json:"epoch"`
			Error     string           `json:"error_message"`
		}
		if err := json.Unmarshal(data, &wire); err != nil {
			t.Fatalf("invalid JSON admitted: %v", err)
		}
		if wire.Version == nil || *wire.Version != 1 {
			t.Fatal("absent or unsupported receipt schema admitted")
		}
		if got.WorkloadKey != workload || got.PolicyID.Strategy != policy.Strategy ||
			got.PolicyID.Seed != policy.Seed || got.PolicyID.RuleSet != policy.RuleSet ||
			got.PolicyID.ScoringEpoch != policy.ScoringEpoch || !maps.Equal(got.PolicyID.Options, policy.Options) {
			t.Fatal("receipt for a different workload or policy admitted")
		}
		if got.CandidateID == "" || got.ArtifactKey == "" || math.IsNaN(got.RankScore) || math.IsInf(got.RankScore, 0) {
			t.Fatal("receipt admitted empty identity or nonfinite rank")
		}
		for _, ns := range got.Timings {
			if ns < 0 {
				t.Fatal("negative timing admitted")
			}
		}
		if !got.Started.IsZero() && !got.Finished.IsZero() && got.Finished.Before(got.Started) {
			t.Fatal("reversed nonzero timestamps admitted")
		}
		if got.WorkloadKey != wire.Workload || got.PolicyID.Strategy != wire.Policy.Strategy ||
			got.PolicyID.Seed != wire.Policy.Seed || got.PolicyID.RuleSet != wire.Policy.RuleSet ||
			got.PolicyID.ScoringEpoch != wire.Policy.Epoch || !maps.Equal(got.PolicyID.Options, wire.Policy.Options) ||
			got.CandidateID != wire.Candidate || got.ArtifactKey != wire.Artifact ||
			got.Complete != wire.Complete || !got.Started.Equal(wire.Started) || !got.Finished.Equal(wire.Finished) ||
			string(got.TimingKind) != wire.Timing || !maps.Equal(got.Timings, wire.Timings) ||
			got.RankScore != wire.Rank || got.Epoch != wire.Epoch || got.ErrorMessage != wire.Error {
			t.Fatal("successful receipt does not retain the standard-JSON payload")
		}
	})
}
