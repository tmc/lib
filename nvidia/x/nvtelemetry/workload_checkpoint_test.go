package nvtelemetry_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// These are synthetic application assertions, not framework or storage receipts.
func checkpointFixture() (nvtelemetry.TrainingContext, nvtelemetry.TrainingCheckpointContext) {
	c := nvtelemetry.TrainingContext{Version: 1, RunID: "synthetic", Updates: []nvtelemetry.TrainingUpdate{}}
	for _, step := range []struct {
		n       uint64
		outcome string
	}{{11, "updated"}, {12, "skipped"}, {13, "failed"}} {
		for rank := range uint32(2) {
			u := nvtelemetry.TrainingUpdate{Attempt: "a", Step: step.n, Rank: rank, WorldSize: 2, Outcome: step.outcome, Quality: "unknown", Evidence: "synthetic:update"}
			if step.outcome == "updated" {
				u.Quality, u.QualityPolicy = "accepted", "synthetic:oracle"
			}
			c.Updates = append(c.Updates, u)
		}
	}
	p := nvtelemetry.TrainingCheckpointContext{Version: 1, RunID: c.RunID, Checkpoints: []nvtelemetry.TrainingCheckpoint{}}
	for _, state := range []string{"requested", "writing", "published", "restored"} {
		for rank := range uint32(2) {
			r := nvtelemetry.TrainingCheckpoint{CheckpointID: "h", State: state, SourceAttempt: "a", SavedStep: 10, Rank: rank, WorldSize: 2, Evidence: "synthetic:checkpoint"}
			if state == "published" || state == "restored" {
				r.Digest, r.Storage = strings.Repeat("a", 64), "synthetic:storage"
			}
			if state == "restored" {
				generation := uint64(1)
				r.RestoredAttempt, r.ResumeGeneration = "b", &generation
			}
			p.Checkpoints = append(p.Checkpoints, r)
		}
	}
	return c, p
}

func checkpointSource() nvtelemetry.WorkloadSource {
	return nvtelemetry.WorkloadSource{Provider: "application", Version: "synthetic-v1", RunID: "synthetic"}
}

func checkpointExports(t *testing.T, c nvtelemetry.TrainingContext, p nvtelemetry.TrainingCheckpointContext) (nvtelemetry.WorkloadExport, nvtelemetry.WorkloadExport) {
	t.Helper()
	var train, checkpoints bytes.Buffer
	if err := nvtelemetry.WriteTrainingContext(&train, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if err := nvtelemetry.WriteTrainingCheckpointContext(&checkpoints, p, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	te, _, err := nvtelemetry.ReadTrainingContext(&train, checkpointSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	ce, _, err := nvtelemetry.ReadTrainingCheckpointContext(&checkpoints, checkpointSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	return te, ce
}

func checkpointSummary(t *testing.T, c nvtelemetry.TrainingContext, p nvtelemetry.TrainingCheckpointContext) nvtelemetry.TrainingRecoverySummary {
	t.Helper()
	te, ce := checkpointExports(t, c, p)
	s, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Recoveries) != 1 {
		t.Fatalf("recoveries=%d, want 1", len(s.Recoveries))
	}
	return s
}

func TestTrainingCheckpointPublic(t *testing.T) {
	if driver.Loaded() {
		t.Fatal("driver loaded before host fixture")
	}
	t.Cleanup(func() {
		if driver.Loaded() {
			t.Error("host fixture loaded driver")
		}
	})
	c, p := checkpointFixture()
	te, ce := checkpointExports(t, c, p)
	for _, e := range []nvtelemetry.WorkloadExport{te, ce} {
		h := sha256.Sum256(e.Raw)
		if e.RawSHA256 != hex.EncodeToString(h[:]) {
			t.Fatal("reader digest is not raw-byte SHA256")
		}
	}
	s, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if s.RunID != c.RunID || s.TrainingRawSHA256 != te.RawSHA256 || s.CheckpointRawSHA256 != ce.RawSHA256 || len(s.Recoveries) != 1 || !reflect.DeepEqual(s.Checkpoints, p.Checkpoints) {
		t.Fatalf("raw or lifecycle binding: %+v", s)
	}
	want := nvtelemetry.TrainingContextSummary{Accepted: 1, Skipped: 1, Failed: 1}
	r := s.Recoveries[0]
	if !r.RecoveryAvailable || r.ObservedAcceptedLoss == nil || *r.ObservedAcceptedLoss != 1 || r.UpdatePopulationComplete || r.After != want || s.Updates != want || r.SourceAttempt != "a" || r.RestoredAttempt != "b" || r.ResumeGeneration != 1 || r.SavedStep != 10 || r.Digest != strings.Repeat("a", 64) {
		t.Fatalf("hand oracle checkpoint10/accepted11/skipped12/failed13: %+v", s)
	}
	// Decoded objects and caller-owned metadata cannot override retained Raw.
	_, decoded, err := nvtelemetry.ReadTrainingCheckpointContext(bytes.NewReader(ce.Raw), checkpointSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	decoded.RunID, decoded.Checkpoints[0].SavedStep = "changed", 999
	*decoded.Checkpoints[6].ResumeGeneration = 99
	ce.Source.Units = map[string]string{"unrelated": "changed"}
	s2, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{})
	if err != nil || !reflect.DeepEqual(s, s2) {
		t.Fatalf("mutable metadata replaced bytes: %+v %v", s2, err)
	}
	*s.Recoveries[0].ObservedAcceptedLoss = 42
	*s.Checkpoints[6].ResumeGeneration = 42
	s3, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{})
	if err != nil || *s3.Recoveries[0].ObservedAcceptedLoss != 1 || s3.Recoveries[0].ResumeGeneration != 1 {
		t.Fatal("summary retained caller mutable state", err)
	}
}

func TestTrainingCheckpointRecoveryRefusals(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*nvtelemetry.TrainingContext, *nvtelemetry.TrainingCheckpointContext)
		point  bool
	}{
		{"missing publication rank", func(_ *nvtelemetry.TrainingContext, p *nvtelemetry.TrainingCheckpointContext) {
			p.Checkpoints = append(p.Checkpoints[:5], p.Checkpoints[6:]...)
		}, false},
		{"no publication", func(_ *nvtelemetry.TrainingContext, p *nvtelemetry.TrainingCheckpointContext) {
			p.Checkpoints = append(p.Checkpoints[:4], p.Checkpoints[6:]...)
		}, false},
		{"missing restore rank", func(_ *nvtelemetry.TrainingContext, p *nvtelemetry.TrainingCheckpointContext) {
			p.Checkpoints = p.Checkpoints[:7]
		}, false},
		{"storage mismatch", func(_ *nvtelemetry.TrainingContext, p *nvtelemetry.TrainingCheckpointContext) {
			p.Checkpoints[6].Storage = "other:storage"
		}, false},
		{"missing update rank", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			c.Updates = c.Updates[1:]
		}, true},
		{"changed update world", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			c.Updates = c.Updates[1:]
			c.Updates[0].Rank, c.Updates[0].WorldSize = 0, 1
		}, true},
		{"changed update attempt", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			c.Updates[0].Attempt = "other"
		}, true},
		{"unknown source attempt", func(_ *nvtelemetry.TrainingContext, p *nvtelemetry.TrainingCheckpointContext) {
			for i := range p.Checkpoints {
				p.Checkpoints[i].SourceAttempt = "other"
			}
		}, true},
		{"no updates", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			c.Updates = []nvtelemetry.TrainingUpdate{}
		}, true},
		{"contradictory saved outcome", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			for _, u := range append([]nvtelemetry.TrainingUpdate(nil), c.Updates[4:]...) {
				u.Step = 10
				c.Updates = append(c.Updates, u)
			}
		}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, p := checkpointFixture()
			tt.change(&c, &p)
			s := checkpointSummary(t, c, p)
			r := s.Recoveries[0]
			if r.RecoveryAvailable != tt.point || r.ObservedAcceptedLoss != nil || r.UpdatePopulationComplete || r.UnavailableReason == "" {
				t.Fatalf("unqualified recovery/loss: %+v", r)
			}
		})
	}
	for _, field := range []string{"attempt", "generation"} {
		t.Run("split restored "+field, func(t *testing.T) {
			c, p := checkpointFixture()
			if field == "attempt" {
				p.Checkpoints[7].RestoredAttempt = "c"
			} else {
				n := uint64(2)
				p.Checkpoints[7].ResumeGeneration = &n
			}
			te, ce := checkpointExports(t, c, p)
			s, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{})
			if err != nil || len(s.Recoveries) != 2 {
				t.Fatalf("split identity: %+v %v", s, err)
			}
			for _, r := range s.Recoveries {
				if r.RecoveryAvailable || r.ObservedAcceptedLoss != nil {
					t.Fatalf("merged distinct restart identities: %+v", r)
				}
			}
		})
	}
}

func TestTrainingCheckpointObservedPopulation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*nvtelemetry.TrainingContext, *nvtelemetry.TrainingCheckpointContext)
		want   uint64
	}{
		{"inclusive saved update", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			for _, u := range append([]nvtelemetry.TrainingUpdate(nil), c.Updates[:2]...) {
				u.Step = 10
				c.Updates = append(c.Updates, u)
			}
		}, 1},
		{"huge observed gap", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			for _, u := range append([]nvtelemetry.TrainingUpdate(nil), c.Updates[:2]...) {
				u.Step = ^uint64(0)
				c.Updates = append(c.Updates, u)
			}
		}, 2},
		{"other attempt", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			for _, u := range append([]nvtelemetry.TrainingUpdate(nil), c.Updates[:2]...) {
				u.Attempt = "b"
				c.Updates = append(c.Updates, u)
			}
		}, 1},
		{"entire accepted update omitted", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			c.Updates = c.Updates[2:]
		}, 0},
		{"all observations at saved step", func(c *nvtelemetry.TrainingContext, _ *nvtelemetry.TrainingCheckpointContext) {
			c.Updates = c.Updates[:2]
			for i := range c.Updates {
				c.Updates[i].Step = 10
			}
		}, 0},
		{"zero step and generation", func(c *nvtelemetry.TrainingContext, p *nvtelemetry.TrainingCheckpointContext) {
			for i := range p.Checkpoints {
				p.Checkpoints[i].SavedStep = 0
				if p.Checkpoints[i].ResumeGeneration != nil {
					n := uint64(0)
					p.Checkpoints[i].ResumeGeneration = &n
				}
			}
			for i := range c.Updates {
				c.Updates[i].Step -= 11
			}
		}, 0},
		{"rank-specific storage", func(_ *nvtelemetry.TrainingContext, p *nvtelemetry.TrainingCheckpointContext) {
			p.Checkpoints[5].Storage, p.Checkpoints[7].Storage = "rank1:storage", "rank1:storage"
		}, 1},
		{"unknown record chronology", func(_ *nvtelemetry.TrainingContext, p *nvtelemetry.TrainingCheckpointContext) {
			for i, j := 0, len(p.Checkpoints)-1; i < j; i, j = i+1, j-1 {
				p.Checkpoints[i], p.Checkpoints[j] = p.Checkpoints[j], p.Checkpoints[i]
			}
		}, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, p := checkpointFixture()
			tt.change(&c, &p)
			s := checkpointSummary(t, c, p)
			r := s.Recoveries[0]
			if !r.RecoveryAvailable || r.ObservedAcceptedLoss == nil || *r.ObservedAcceptedLoss != tt.want || r.UpdatePopulationComplete {
				t.Fatalf("conditional observed loss=%+v want %d", r, tt.want)
			}
			if !reflect.DeepEqual(s.Checkpoints, p.Checkpoints) {
				t.Fatal("rewrote source order")
			}
		})
	}
	c, p := checkpointFixture()
	p.Checkpoints = p.Checkpoints[:4]
	te, ce := checkpointExports(t, c, p)
	s, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{})
	if err != nil || len(s.Recoveries) != 0 || len(s.Checkpoints) != 4 || s.Updates.Failed != 1 {
		t.Fatalf("invented recovery from requested/writing: %+v %v", s, err)
	}
}

func TestTrainingCheckpointQuality(t *testing.T) {
	for _, quality := range []string{"rejected", "unknown"} {
		t.Run(quality, func(t *testing.T) {
			c, p := checkpointFixture()
			c.Updates[0].Quality = quality
			if quality == "unknown" {
				c.Updates[0].QualityPolicy = ""
			}
			r := checkpointSummary(t, c, p).Recoveries[0]
			if !r.RecoveryAvailable || r.ObservedAcceptedLoss == nil || *r.ObservedAcceptedLoss != 0 || r.UpdatePopulationComplete {
				t.Fatalf("quality became accepted useful work: %+v", r)
			}
			if quality == "rejected" && r.After.Rejected != 1 || quality == "unknown" && r.After.Unknown != 1 {
				t.Fatalf("lost unaccepted outcome: %+v", r.After)
			}
		})
	}
}

func TestTrainingCheckpointRawBinding(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*nvtelemetry.WorkloadExport, *nvtelemetry.WorkloadExport)
	}{
		{"training digest", func(t, c *nvtelemetry.WorkloadExport) { t.RawSHA256 = strings.Repeat("0", 64) }},
		{"checkpoint digest", func(t, c *nvtelemetry.WorkloadExport) { c.RawSHA256 = "" }},
		{"training bytes", func(t, c *nvtelemetry.WorkloadExport) { t.Raw = append(t.Raw, ' ') }},
		{"checkpoint bytes", func(t, c *nvtelemetry.WorkloadExport) { c.Raw = append(c.Raw, ' ') }},
		{"training run metadata", func(t, c *nvtelemetry.WorkloadExport) { t.Source.RunID = "other" }},
		{"checkpoint run metadata", func(t, c *nvtelemetry.WorkloadExport) { c.Source.RunID = "other" }},
		{"provider", func(t, c *nvtelemetry.WorkloadExport) { c.Source.Provider = "wandb" }},
		{"paginated", func(t, c *nvtelemetry.WorkloadExport) { c.NextPageToken = "next" }},
		{"run bytes", func(t, c *nvtelemetry.WorkloadExport) {
			c.Raw = bytes.ReplaceAll(c.Raw, []byte("synthetic\""), []byte("other\""))
			c.Source.RunID = "other"
			h := sha256.Sum256(c.Raw)
			c.RawSHA256 = hex.EncodeToString(h[:])
		}},
		{"mutated unknown field", func(t, c *nvtelemetry.WorkloadExport) {
			c.Raw = bytes.Replace(c.Raw, []byte(`"version":1`), []byte(`"version":1,"injected":0`), 1)
			h := sha256.Sum256(c.Raw)
			c.RawSHA256 = hex.EncodeToString(h[:])
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, p := checkpointFixture()
			te, ce := checkpointExports(t, c, p)
			tt.change(&te, &ce)
			if _, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("accepted unbound bytes/metadata")
			}
		})
	}
}

func TestTrainingCheckpointStrictReader(t *testing.T) {
	_, p := checkpointFixture()
	p.Checkpoints = p.Checkpoints[6:7]
	p.Checkpoints[0].SavedStep = 0
	n := uint64(0)
	p.Checkpoints[0].ResumeGeneration = &n
	var b bytes.Buffer
	if err := nvtelemetry.WriteTrainingCheckpointContext(&b, p, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	raw := b.String()
	for _, tt := range []struct{ name, raw string }{
		{"duplicate key", strings.Replace(raw, `"version":1`, `"version":1,"version":1`, 1)},
		{"extra field", strings.Replace(raw, `"version":1`, `"version":1,"durable":true`, 1)},
		{"version", strings.Replace(raw, `"version":1`, `"version":2`, 1)},
		{"wrong run", strings.Replace(raw, `"run_id":"synthetic"`, `"run_id":"other"`, 1)},
		{"state", strings.Replace(raw, `"state":"restored"`, `"state":"durable"`, 1)},
		{"digest", strings.Replace(raw, strings.Repeat("a", 64), "bad", 1)},
		{"storage", strings.Replace(raw, `"storage":"synthetic:storage"`, `"storage":""`, 1)},
		{"evidence", strings.Replace(raw, `"evidence":"synthetic:checkpoint"`, `"evidence":""`, 1)},
		{"restored attempt", strings.Replace(raw, `"restored_attempt":"b"`, `"restored_attempt":"a"`, 1)},
		{"absent step", strings.Replace(raw, `"saved_step":0,`, "", 1)},
		{"null step", strings.Replace(raw, `"saved_step":0`, `"saved_step":null`, 1)},
		{"fractional step", strings.Replace(raw, `"saved_step":0`, `"saved_step":0.5`, 1)},
		{"negative step", strings.Replace(raw, `"saved_step":0`, `"saved_step":-1`, 1)},
		{"absent rank", strings.Replace(raw, `"rank":0,`, "", 1)},
		{"null rank", strings.Replace(raw, `"rank":0`, `"rank":null`, 1)},
		{"absent world", strings.Replace(raw, `"world_size":2,`, "", 1)},
		{"null world", strings.Replace(raw, `"world_size":2`, `"world_size":null`, 1)},
		{"absent generation", strings.Replace(raw, `"resume_generation":0,`, "", 1)},
		{"null generation", strings.Replace(raw, `"resume_generation":0`, `"resume_generation":null`, 1)},
		{"fractional generation", strings.Replace(raw, `"resume_generation":0`, `"resume_generation":0.5`, 1)},
		{"null records", strings.Replace(raw, raw[strings.Index(raw, `"checkpoints":`):], `"checkpoints":null}`, 1)},
		{"truncated", raw[:len(raw)-1]}, {"trailing", raw + ` {}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.raw == raw {
				t.Fatal("ineffective negative fixture")
			}
			if _, _, err := nvtelemetry.ReadTrainingCheckpointContext(strings.NewReader(tt.raw), checkpointSource(), nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("accepted invalid checkpoint JSON")
			}
		})
	}
	if _, _, err := nvtelemetry.ReadTrainingContext(strings.NewReader(raw), checkpointSource(), nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("old training reader accepted checkpoint schema")
	}
	var train bytes.Buffer
	c, _ := checkpointFixture()
	if err := nvtelemetry.WriteTrainingContext(&train, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := nvtelemetry.ReadTrainingCheckpointContext(&train, checkpointSource(), nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("checkpoint reader accepted old updates schema")
	}
}

func TestTrainingCheckpointWriterRefusals(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*nvtelemetry.TrainingCheckpointContext)
	}{
		{"nil records", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints = nil }},
		{"duplicate", func(p *nvtelemetry.TrainingCheckpointContext) {
			p.Checkpoints = append(p.Checkpoints, p.Checkpoints[4])
		}},
		{"digest contradiction", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints[7].Digest = strings.Repeat("b", 64) }},
		{"source attempt contradiction", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints[7].SourceAttempt = "other" }},
		{"step contradiction", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints[7].SavedStep = 11 }},
		{"world contradiction", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints[7].WorldSize = 3 }},
		{"invalid rank", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints[7].Rank = 2 }},
		{"invalid world", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints[7].WorldSize = 0 }},
		{"missing restart", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints[7].ResumeGeneration = nil }},
		{"restart on publication", func(p *nvtelemetry.TrainingCheckpointContext) { n := uint64(1); p.Checkpoints[4].ResumeGeneration = &n }},
		{"empty attempt", func(p *nvtelemetry.TrainingCheckpointContext) { p.Checkpoints[4].SourceAttempt = "" }},
		{"conflicting restore", func(p *nvtelemetry.TrainingCheckpointContext) {
			q := p.Checkpoints[6]
			q.CheckpointID = "different"
			p.Checkpoints = append(p.Checkpoints, q)
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, p := checkpointFixture()
			tt.change(&p)
			var b bytes.Buffer
			if err := nvtelemetry.WriteTrainingCheckpointContext(&b, p, nvtelemetry.WorkloadLimits{}); err == nil || b.Len() != 0 {
				t.Fatal("emitted invalid checkpoint context", err)
			}
			raw, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := nvtelemetry.ReadTrainingCheckpointContext(bytes.NewReader(raw), checkpointSource(), nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("reader accepted writer refusal")
			}
		})
	}
}

type checkpointErrorReader struct{ err error }

func (r checkpointErrorReader) Read([]byte) (int, error) { return 0, r.err }

type checkpointShortWriter struct{}

func (checkpointShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

type checkpointErrorWriter struct{ err error }

func (w checkpointErrorWriter) Write([]byte) (int, error) { return 0, w.err }

func TestTrainingCheckpointLimitsAndCancel(t *testing.T) {
	c, p := checkpointFixture()
	te, ce := checkpointExports(t, c, p)
	for _, limits := range []nvtelemetry.WorkloadLimits{{MaxBytes: 10}, {MaxMetrics: 1}, {MaxStringBytes: 2}, {MaxBytes: -1}, {MaxMetrics: 65537}, {MaxBytes: 1<<63 - 1}} {
		var b bytes.Buffer
		if err := nvtelemetry.WriteTrainingCheckpointContext(&b, p, limits); err == nil || b.Len() != 0 {
			t.Fatal("writer failed bounded pre-emission refusal", limits, err)
		}
		if _, _, err := nvtelemetry.ReadTrainingCheckpointContext(bytes.NewReader(ce.Raw), checkpointSource(), limits); err == nil {
			t.Fatal("reader admitted excessive input", limits)
		}
		if _, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, limits); err == nil {
			t.Fatal("summary ignored limits", limits)
		}
	}
	if err := nvtelemetry.WriteTrainingCheckpointContext(checkpointShortWriter{}, p, nvtelemetry.WorkloadLimits{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write: %v", err)
	}
	if err := nvtelemetry.WriteTrainingCheckpointContext(checkpointErrorWriter{context.Canceled}, p, nvtelemetry.WorkloadLimits{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("writer cancellation: %v", err)
	}
	if _, _, err := nvtelemetry.ReadTrainingCheckpointContext(checkpointErrorReader{context.Canceled}, checkpointSource(), nvtelemetry.WorkloadLimits{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("reader cancellation: %v", err)
	}
	if err := nvtelemetry.WriteTrainingCheckpointContext(nil, p, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("nil writer admitted")
	}
	if _, _, err := nvtelemetry.ReadTrainingCheckpointContext(nil, checkpointSource(), nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("nil reader admitted")
	}
	// Exact encoded-byte and record/string boundaries are admitted.
	limits := nvtelemetry.WorkloadLimits{MaxBytes: int64(len(ce.Raw)), MaxMetrics: len(p.Checkpoints), MaxStringBytes: 64}
	var b bytes.Buffer
	if err := nvtelemetry.WriteTrainingCheckpointContext(&b, p, limits); err != nil || !bytes.Equal(b.Bytes(), ce.Raw) {
		t.Fatal("exact writer boundary", err)
	}
	if _, _, err := nvtelemetry.ReadTrainingCheckpointContext(bytes.NewReader(b.Bytes()), checkpointSource(), limits); err != nil {
		t.Fatal("exact reader boundary", err)
	}
}

func TestTrainingCheckpointConcurrent(t *testing.T) {
	c, p := checkpointFixture()
	te, ce := checkpointExports(t, c, p)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 16 {
				s, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{})
				if err != nil || len(s.Recoveries) != 1 || s.Recoveries[0].ObservedAcceptedLoss == nil || *s.Recoveries[0].ObservedAcceptedLoss != 1 {
					t.Errorf("concurrent summary: %+v %v", s, err)
				}
			}
		})
	}
	wg.Wait()
}
