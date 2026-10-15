package nvtelemetry

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

// TrainingCheckpoint is an application-supplied checkpoint observation. SavedStep
// includes completed optimizer update N, so recovery counts updates strictly after
// N. Digest is a lowercase SHA-256 identifying the global checkpoint. Storage may
// differ by rank, but publication and restore must name the same storage per rank.
// Evidence is an application artifact reference, not verified durability proof.
// RestoredAttempt and ResumeGeneration are required only for restored records;
// generation zero is valid and distinct from an absent generation.
type TrainingCheckpoint struct {
	CheckpointID     string  `json:"checkpoint_id"`
	State            string  `json:"state"` // requested, writing, published, restored
	Digest           string  `json:"digest,omitempty"`
	Storage          string  `json:"storage,omitempty"`
	SourceAttempt    string  `json:"source_attempt"`
	SavedStep        uint64  `json:"saved_step"`
	Rank             uint32  `json:"rank"`
	WorldSize        uint32  `json:"world_size"`
	RestoredAttempt  string  `json:"restored_attempt,omitempty"`
	ResumeGeneration *uint64 `json:"resume_generation,omitempty"`
	Evidence         string  `json:"evidence"`
}

// TrainingCheckpointContext is a distinct version 1 application interchange.
// Records retain source order, which is not a measured or declared event clock.
// Missing lifecycle records remain missing; state names do not prove chronology,
// storage durability, or a framework's actual checkpoint behavior.
type TrainingCheckpointContext struct {
	Version     int                  `json:"version"`
	RunID       string               `json:"run_id"`
	Checkpoints []TrainingCheckpoint `json:"checkpoints"`
}

// TrainingRecovery reports one explicitly supplied restored attempt/generation.
// RecoveryAvailable requires matching publication and restore for every rank.
// After counts observed source-attempt outcomes strictly after SavedStep.
// ObservedAcceptedLoss is nil unless recovery and those observed rank groups are
// complete and consistent. A nonnil count is conditional on the supplied records:
// entire omitted updates are undetectable, so UpdatePopulationComplete is always
// false. Even an observed zero is not a total-loss claim.
type TrainingRecovery struct {
	CheckpointID             string
	Digest                   string
	SourceAttempt            string
	SavedStep                uint64
	RestoredAttempt          string
	ResumeGeneration         uint64
	RecoveryAvailable        bool
	UnavailableReason        string
	After                    TrainingContextSummary
	ObservedAcceptedLoss     *uint64
	UpdatePopulationComplete bool
}

// TrainingRecoverySummary binds all identities and observations to the original
// training and checkpoint export bytes. It does not authenticate application
// assertions or infer phase-to-kernel, time, quality, or durable-storage joins.
type TrainingRecoverySummary struct {
	RunID               string
	TrainingRawSHA256   string
	CheckpointRawSHA256 string
	Updates             TrainingContextSummary
	Checkpoints         []TrainingCheckpoint
	Recoveries          []TrainingRecovery
}

// ReadTrainingCheckpointContext reads bounded version 1 supplied observations.
// It rejects duplicate/unknown fields, missing integer presence, contradictory
// checkpoint identities, and duplicate rank/state/recovery identities. A partial
// publication or restore is retained, but cannot qualify a global recovery point.
func ReadTrainingCheckpointContext(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, TrainingCheckpointContext, error) {
	if s.Provider != "application" {
		return WorkloadExport{}, TrainingCheckpointContext{}, errors.New("training checkpoint context requires application source")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, TrainingCheckpointContext{}, err
	}
	var c TrainingCheckpointContext
	if err := decodeWorkloadJSON(e.Raw, &c); err != nil {
		return e, c, err
	}
	d := json.NewDecoder(bytes.NewReader(e.Raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return e, c, fmt.Errorf("decode training checkpoint context: %w", err)
	}
	if err := requireContextIntegers(e.Raw, "checkpoints", "saved_step", "rank", "world_size"); err != nil {
		return e, c, err
	}
	if err := validateTrainingCheckpoints(c, s.RunID, l); err != nil {
		return e, c, err
	}
	return e, c, nil
}

// WriteTrainingCheckpointContext validates and writes the bounded interchange.
// Validation and encoded byte-budget refusal precede any write. This writer does
// not publish or restore a checkpoint or generate application evidence.
func WriteTrainingCheckpointContext(w io.Writer, c TrainingCheckpointContext, l WorkloadLimits) error {
	l, err := workloadLimits(l)
	if err != nil {
		return err
	}
	if w == nil {
		return errors.New("missing training checkpoint context writer")
	}
	if err := validateTrainingCheckpoints(c, c.RunID, l); err != nil {
		return err
	}
	raw, err := marshalContext(c.RunID, "checkpoints", c.Checkpoints, l)
	if err != nil {
		return err
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		return io.ErrShortWrite
	}
	return err
}

func validateTrainingCheckpoints(c TrainingCheckpointContext, run string, l WorkloadLimits) error {
	if c.Version != 1 || c.RunID != run || !checkpointText(c.RunID) || len(c.RunID) > l.MaxStringBytes || c.Checkpoints == nil || len(c.Checkpoints) > l.MaxMetrics {
		return errors.New("invalid training checkpoint version, run or record count")
	}
	type identity struct {
		attempt string
		step    uint64
		world   uint32
		digest  string
	}
	type recordKey struct {
		id, state, restored string
		rank                uint32
		generation          uint64
	}
	ids := make(map[string]identity)
	seen := make(map[recordKey]bool)
	restores := make(map[checkpointRestoreKey]string)
	for _, p := range c.Checkpoints {
		for _, s := range []string{p.CheckpointID, p.SourceAttempt, p.Evidence, p.Digest, p.Storage, p.RestoredAttempt} {
			if len(s) > l.MaxStringBytes || s != "" && !checkpointText(s) {
				return errors.New("invalid training checkpoint string or string limit")
			}
		}
		if p.CheckpointID == "" || p.SourceAttempt == "" || p.Evidence == "" || p.WorldSize == 0 || uint64(p.WorldSize) > uint64(l.MaxMetrics) || p.Rank >= p.WorldSize {
			return errors.New("invalid training checkpoint identity")
		}
		if p.Digest != "" {
			if len(p.Digest) != 64 || strings.ToLower(p.Digest) != p.Digest {
				return errors.New("invalid checkpoint sha256")
			}
			if _, err := hex.DecodeString(p.Digest); err != nil {
				return errors.New("invalid checkpoint sha256")
			}
		}
		switch p.State {
		case "requested", "writing":
		case "published", "restored":
			if p.Digest == "" || p.Storage == "" {
				return errors.New("missing published checkpoint digest or storage")
			}
		default:
			return errors.New("unsupported training checkpoint state")
		}
		var generation uint64
		if p.State == "restored" {
			if p.RestoredAttempt == "" || p.RestoredAttempt == p.SourceAttempt || p.ResumeGeneration == nil {
				return errors.New("missing or invalid checkpoint restart identity")
			}
			generation = *p.ResumeGeneration
			rk := checkpointRestoreKey{p.RestoredAttempt, generation}
			if prior := restores[rk]; prior != "" && prior != p.CheckpointID {
				return errors.New("contradictory checkpoint for restored attempt and generation")
			}
			restores[rk] = p.CheckpointID
		} else if p.RestoredAttempt != "" || p.ResumeGeneration != nil {
			return errors.New("restart identity on nonrestored checkpoint")
		}
		id, ok := ids[p.CheckpointID]
		if ok && (id.attempt != p.SourceAttempt || id.step != p.SavedStep || id.world != p.WorldSize || id.digest != "" && p.Digest != "" && id.digest != p.Digest) {
			return errors.New("contradictory training checkpoint identity")
		}
		if !ok {
			id = identity{p.SourceAttempt, p.SavedStep, p.WorldSize, p.Digest}
		} else if p.Digest != "" {
			id.digest = p.Digest
		}
		ids[p.CheckpointID] = id
		k := recordKey{p.CheckpointID, p.State, p.RestoredAttempt, p.Rank, generation}
		if seen[k] {
			return errors.New("duplicate checkpoint rank state or recovery")
		}
		seen[k] = true
	}
	return nil
}

func checkpointText(s string) bool {
	return s != "" && strings.TrimSpace(s) == s && !strings.ContainsAny(s, "\x00\n\r\t")
}

type checkpointRestoreKey struct {
	attempt    string
	generation uint64
}

// SummarizeTrainingRecovery verifies both RawSHA256 values and rereads the actual
// bounded Raw bytes through the context readers. Mutable decoded objects and
// export metadata cannot replace the run, attempt or step identities in Raw.
// Counts reuse SummarizeTrainingContext's all-rank outcome classification.
func SummarizeTrainingRecovery(training, checkpoints WorkloadExport, l WorkloadLimits) (TrainingRecoverySummary, error) {
	l, err := workloadLimits(l)
	if err != nil {
		return TrainingRecoverySummary{}, err
	}
	for _, e := range []WorkloadExport{training, checkpoints} {
		if len(e.Raw) == 0 || int64(len(e.Raw)) > l.MaxBytes || e.NextPageToken != "" {
			return TrainingRecoverySummary{}, errors.New("missing, excessive or paginated context bytes")
		}
		h := sha256.Sum256(e.Raw)
		if e.RawSHA256 != hex.EncodeToString(h[:]) {
			return TrainingRecoverySummary{}, errors.New("context raw sha256 mismatch")
		}
	}
	te, tc, err := ReadTrainingContext(bytes.NewReader(training.Raw), training.Source, l)
	if err != nil {
		return TrainingRecoverySummary{}, err
	}
	ce, cc, err := ReadTrainingCheckpointContext(bytes.NewReader(checkpoints.Raw), checkpoints.Source, l)
	if err != nil {
		return TrainingRecoverySummary{}, err
	}
	if tc.RunID != cc.RunID {
		return TrainingRecoverySummary{}, errors.New("training and checkpoint runs differ")
	}
	updates, err := SummarizeTrainingContext(tc, l)
	if err != nil {
		return TrainingRecoverySummary{}, err
	}
	attempts, err := checkpointAttempts(tc, l)
	if err != nil {
		return TrainingRecoverySummary{}, err
	}
	out := TrainingRecoverySummary{RunID: tc.RunID, TrainingRawSHA256: te.RawSHA256, CheckpointRawSHA256: ce.RawSHA256, Updates: updates, Checkpoints: cc.Checkpoints}
	published := make(map[string]map[uint32]TrainingCheckpoint)
	restored := make(map[checkpointRestoreKey][]TrainingCheckpoint)
	var order []checkpointRestoreKey
	for _, p := range cc.Checkpoints {
		if p.State == "published" {
			if published[p.CheckpointID] == nil {
				published[p.CheckpointID] = make(map[uint32]TrainingCheckpoint)
			}
			published[p.CheckpointID][p.Rank] = p
		}
		if p.State == "restored" {
			k := checkpointRestoreKey{p.RestoredAttempt, *p.ResumeGeneration}
			if restored[k] == nil {
				order = append(order, k)
			}
			restored[k] = append(restored[k], p)
		}
	}
	for _, k := range order {
		group := restored[k]
		p := group[0]
		r := TrainingRecovery{CheckpointID: p.CheckpointID, Digest: p.Digest, SourceAttempt: p.SourceAttempt, SavedStep: p.SavedStep, RestoredAttempt: k.attempt, ResumeGeneration: k.generation}
		a := attempts[p.SourceAttempt]
		i := sort.Search(len(a.steps), func(i int) bool { return a.steps[i].step > p.SavedStep })
		if i < len(a.steps) {
			r.After = a.suffix[i]
		}
		switch {
		case len(published[p.CheckpointID]) != int(p.WorldSize):
			r.UnavailableReason = "incomplete checkpoint publication ranks"
		case len(group) != int(p.WorldSize):
			r.UnavailableReason = "incomplete checkpoint restore ranks"
		default:
			r.RecoveryAvailable = true
			for _, restored := range group {
				publication := published[p.CheckpointID][restored.Rank]
				if publication.Digest != restored.Digest || publication.Storage != restored.Storage {
					r.RecoveryAvailable = false
					r.UnavailableReason = "checkpoint publication and restore differ"
				}
			}
		}
		if r.RecoveryAvailable {
			switch {
			case i > 0 && a.steps[i-1].step == p.SavedStep && (!a.steps[i-1].completed || a.steps[i-1].world != p.WorldSize):
				r.RecoveryAvailable = false
				r.UnavailableReason = "saved step contradicts completed update"
			case len(a.steps) == 0:
				r.UnavailableReason = "source attempt has no observed updates"
			case r.After.Incomplete != 0 || i < len(a.steps) && a.worlds[i] != p.WorldSize:
				r.UnavailableReason = "incomplete or changed observed update ranks"
			default:
				count := r.After.Accepted
				r.ObservedAcceptedLoss = &count
			}
		}
		out.Recoveries = append(out.Recoveries, r)
	}
	return out, nil
}

type checkpointStep struct {
	step      uint64
	world     uint32
	completed bool
	counts    TrainingContextSummary
}

type checkpointAttempt struct {
	steps  []checkpointStep
	suffix []TrainingContextSummary
	worlds []uint32
}

// Sort once and retain suffix summaries so many restore observations do not
// repeatedly scan every update. No step-number gap becomes a work count.
func checkpointAttempts(c TrainingContext, l WorkloadLimits) (map[string]checkpointAttempt, error) {
	type key struct {
		attempt string
		step    uint64
	}
	groups := make(map[key][]TrainingUpdate)
	for _, u := range c.Updates {
		k := key{u.Attempt, u.Step}
		groups[k] = append(groups[k], u)
	}
	out := make(map[string]checkpointAttempt)
	for k, group := range groups {
		counts, err := SummarizeTrainingContext(TrainingContext{Version: 1, RunID: c.RunID, Updates: group}, l)
		if err != nil {
			return nil, err
		}
		completed := len(group) == int(group[0].WorldSize)
		for _, u := range group {
			completed = completed && u.Outcome == "updated"
		}
		a := out[k.attempt]
		a.steps = append(a.steps, checkpointStep{k.step, group[0].WorldSize, completed, counts})
		out[k.attempt] = a
	}
	for attempt, a := range out {
		sort.Slice(a.steps, func(i, j int) bool { return a.steps[i].step < a.steps[j].step })
		a.suffix = make([]TrainingContextSummary, len(a.steps)+1)
		a.worlds = make([]uint32, len(a.steps))
		for i := len(a.steps) - 1; i >= 0; i-- {
			x, y := a.steps[i].counts, a.suffix[i+1]
			a.suffix[i] = TrainingContextSummary{Accepted: x.Accepted + y.Accepted, Rejected: x.Rejected + y.Rejected, Skipped: x.Skipped + y.Skipped, Failed: x.Failed + y.Failed, Unknown: x.Unknown + y.Unknown, Incomplete: x.Incomplete + y.Incomplete}
			a.worlds[i] = a.steps[i].world
			if i+1 < len(a.steps) && a.worlds[i+1] != a.worlds[i] {
				a.worlds[i] = 0
			}
		}
		out[attempt] = a
	}
	return out, nil
}
