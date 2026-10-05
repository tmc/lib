package cudacache

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

const strategyBytes = 2 << 20
const strategyRows = 4096

// Strategy describes an exploration technique for an exact target. Confidence
// and PredictedSpeedup are priors, never measured latency. HistoricalSpeedups
// are producer-declared qualified observations; integrity is not attestation.
// Observations counts declared trajectory observations, Failures counts failed
// correctness checks, and TrajectoryIDs keeps replay identities for deduplication.
type Strategy struct {
	Family             cudatrial.BottleneckFamily `json:"family"`
	TargetArch         string                     `json:"target_arch"`
	Technique          string                     `json:"technique"`
	SourceGroup        string                     `json:"source_group"`
	Confidence         float64                    `json:"confidence"`
	PredictedSpeedup   float64                    `json:"predicted_speedup"`
	HistoricalSpeedups []float64                  `json:"historical_speedups,omitempty"`
	Observations       uint64                     `json:"observations,omitempty"`
	Failures           uint64                     `json:"failures,omitempty"`
	TrajectoryIDs      []string                   `json:"trajectory_ids,omitempty"`
}

// StrategyKnowledgeBase is a bounded, concurrent catalog. Its zero value is
// empty and usable. Load replaces the catalog atomically after verification.
type StrategyKnowledgeBase struct {
	mu   sync.RWMutex
	rows []Strategy
}

func validateStrategy(s Strategy) error {
	if !s.Family.Valid() || cudakernel.ValidateTargetArch(s.TargetArch) != nil || s.Technique == "" || len(s.Technique) > 128 || len(s.SourceGroup) > 128 || len(s.HistoricalSpeedups) > 256 || s.Observations > 4096 || s.Failures > s.Observations || len(s.TrajectoryIDs) > 4096 {
		return fmt.Errorf("cudacache: invalid strategy identity or bounds")
	}
	if len(s.TrajectoryIDs) > int(s.Observations) || len(s.HistoricalSpeedups) > int(s.Observations-s.Failures) {
		return fmt.Errorf("cudacache: inconsistent strategy history counts")
	}
	for _, name := range []string{s.Technique, s.SourceGroup} {
		for _, c := range name {
			if c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') {
				return fmt.Errorf("cudacache: invalid strategy name")
			}
		}
	}
	if math.IsNaN(s.Confidence) || s.Confidence < 0 || s.Confidence > 1 || math.IsNaN(s.PredictedSpeedup) || math.IsInf(s.PredictedSpeedup, 0) || s.PredictedSpeedup <= 0 {
		return fmt.Errorf("cudacache: invalid strategy prior")
	}
	for _, speed := range s.HistoricalSpeedups {
		if math.IsNaN(speed) || math.IsInf(speed, 0) || speed <= 0 {
			return fmt.Errorf("cudacache: invalid historical speedup")
		}
	}
	seen := make(map[string]bool)
	for _, id := range s.TrajectoryIDs {
		if len(id) != 64 {
			return fmt.Errorf("cudacache: invalid strategy trajectory identity")
		}
		b, err := hex.DecodeString(id)
		if err != nil || len(b) != 32 || hex.EncodeToString(b) != id || seen[id] {
			return fmt.Errorf("cudacache: invalid strategy trajectory identity")
		}
		seen[id] = true
	}
	return nil
}

func cloneStrategy(s Strategy) Strategy {
	s.HistoricalSpeedups = append([]float64(nil), s.HistoricalSpeedups...)
	s.TrajectoryIDs = append([]string(nil), s.TrajectoryIDs...)
	return s
}

// Update replaces a technique with the same family, target and source group.
// Caller-provided historical observations remain declarations, not device proof.
func (k *StrategyKnowledgeBase) Update(s Strategy) error {
	if k == nil {
		return fmt.Errorf("cudacache: nil knowledge base")
	}
	if err := validateStrategy(s); err != nil {
		return err
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	rows := append([]Strategy(nil), k.rows...)
	index := -1
	for i, old := range rows {
		if old.Family == s.Family && old.TargetArch == s.TargetArch && old.Technique == s.Technique && old.SourceGroup == s.SourceGroup {
			index = i
			break
		}
	}
	if index < 0 {
		rows = append(rows, cloneStrategy(s))
	} else {
		rows[index] = cloneStrategy(s)
	}
	if len(rows) > strategyRows {
		return fmt.Errorf("cudacache: strategy row limit")
	}
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.Family != b.Family {
			return a.Family < b.Family
		}
		if a.TargetArch != b.TargetArch {
			return a.TargetArch < b.TargetArch
		}
		if a.Technique != b.Technique {
			return a.Technique < b.Technique
		}
		return a.SourceGroup < b.SourceGroup
	})
	b, err := json.Marshal(rows)
	if err != nil || len(b) > strategyBytes-1024 {
		return fmt.Errorf("cudacache: strategy byte limit")
	}
	k.rows = rows
	return nil
}

// Query returns detached strategies, ordered by confidence and then technique.
// Exact target matching prevents transferring an architecture-specific prior.
func (k *StrategyKnowledgeBase) Query(f cudatrial.BottleneckFamily, target string) []Strategy {
	if k == nil {
		return nil
	}
	k.mu.RLock()
	var rows []Strategy
	for _, s := range k.rows {
		if s.Family == f && s.TargetArch == target {
			rows = append(rows, cloneStrategy(s))
		}
	}
	k.mu.RUnlock()
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Confidence != rows[j].Confidence {
			return rows[i].Confidence > rows[j].Confidence
		}
		if rows[i].Technique != rows[j].Technique {
			return rows[i].Technique < rows[j].Technique
		}
		return rows[i].SourceGroup < rows[j].SourceGroup
	})
	return rows
}

type strategyEnvelope struct {
	Version int        `json:"version"`
	SHA256  string     `json:"sha256"`
	Rows    []Strategy `json:"rows"`
}

// Write writes a versioned checksummed snapshot without holding locks over I/O.
func (k *StrategyKnowledgeBase) Write(w io.Writer) error {
	if k == nil || w == nil {
		return fmt.Errorf("cudacache: nil snapshot owner or writer")
	}
	k.mu.RLock()
	rows := make([]Strategy, len(k.rows))
	for i, s := range k.rows {
		rows[i] = cloneStrategy(s)
	}
	k.mu.RUnlock()
	b, err := json.Marshal(rows)
	if err != nil {
		return err
	}
	encoded, err := json.Marshal(strategyEnvelope{1, fmt.Sprintf("%x", sha256.Sum256(b)), rows})
	if err != nil {
		return err
	}
	if len(encoded)+1 > strategyBytes {
		return fmt.Errorf("cudacache: strategy byte limit")
	}
	encoded = append(encoded, '\n')
	_, err = io.Copy(w, bytes.NewReader(encoded))
	return err
}

// Load reads at most 2 MiB, checks integrity and rejects duplicate strategy
// identities. Failure leaves the receiver unchanged. The checksum detects
// corruption; it cannot authenticate the producer or its hardware claims.
func (k *StrategyKnowledgeBase) Load(r io.Reader) error {
	if k == nil || r == nil {
		return fmt.Errorf("cudacache: nil snapshot owner or reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, strategyBytes+1))
	if err != nil {
		return err
	}
	if len(b) > strategyBytes {
		return fmt.Errorf("cudacache: strategy byte limit")
	}
	var e strategyEnvelope
	if err := memoryJSON(b); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&e); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("cudacache: trailing strategy JSON")
	}
	if e.Version != 1 || len(e.Rows) > strategyRows {
		return fmt.Errorf("cudacache: unsupported catalog or row limit")
	}
	body, err := json.Marshal(e.Rows)
	if err != nil || fmt.Sprintf("%x", sha256.Sum256(body)) != e.SHA256 {
		return fmt.Errorf("%w: strategy checksum", ErrCorruptReceipt)
	}
	seen := make(map[string]bool)
	for _, s := range e.Rows {
		if err := validateStrategy(s); err != nil {
			return err
		}
		key := string(s.Family) + "\x00" + s.TargetArch + "\x00" + s.Technique + "\x00" + s.SourceGroup
		if seen[key] {
			return fmt.Errorf("cudacache: duplicate strategy")
		}
		seen[key] = true
	}
	k.mu.Lock()
	k.rows = e.Rows
	k.mu.Unlock()
	return nil
}
