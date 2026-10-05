package cudatrial

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// ErrUnqualifiedOutcome reports a rollout without admitted hardware evidence.
var ErrUnqualifiedOutcome = errors.New("cudatrial: unqualified outcome")

// TrajectoryReceipt records a producer-declared execution observation. Domain
// is an exact caller-defined cross-task compatibility digest. WorkloadKey binds
// the individual task. Measured latency is an explicit GPU event channel, not
// host time. SHA256 detects corruption, not dishonest producers.
type TrajectoryReceipt struct {
	Version                int              `json:"version"`
	WorkloadKey            string           `json:"workload_key"`
	Domain                 string           `json:"domain"`
	TargetArch             string           `json:"target_arch"`
	ArtifactDigest         string           `json:"artifact_digest"`
	CompilerKey            string           `json:"compiler_key"`
	BaselineArtifactDigest string           `json:"baseline_artifact_digest"`
	BaselineCompilerKey    string           `json:"baseline_compiler_key"`
	InputDigest            string           `json:"input_digest"`
	PolicyID               string           `json:"policy_id"`
	AppliedRules           []string         `json:"applied_rules"`
	BottleneckState        BottleneckFamily `json:"bottleneck_state"`
	MeasuredLatencyNs      int64            `json:"measured_latency_ns"`
	BaselineLatencyNs      int64            `json:"baseline_latency_ns"`
	SpeedupRatio           float64          `json:"speedup_ratio"`
	VerifiedCorrect        bool             `json:"verified_correct"`
	PendingHardware        bool             `json:"pending_hardware"`
	DeviceUUID             string           `json:"device_uuid"`
	LockPID                int              `json:"lock_pid"`
	SHA256                 string           `json:"sha256"`
	liveSeal               string
}

func trajectoryDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32 && s == hex.EncodeToString(b)
}

func (r TrajectoryReceipt) checksum() string {
	r.SHA256 = ""
	b, err := json.Marshal(r)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%x", sha256.Sum256(b))
}

// Validate checks a persisted receipt's declared identity and integrity. It does
// not re-execute hardware or authenticate the producer. Unqualified records
// cannot enter replay memory or contribute speedups.
func (r TrajectoryReceipt) Validate() error {
	if r.PendingHardware || r.Version != 1 || r.DeviceUUID == "" || len(r.DeviceUUID) > 128 || strings.TrimSpace(r.DeviceUUID) != r.DeviceUUID || r.LockPID <= 0 {
		return ErrUnqualifiedOutcome
	}
	if err := r.validateIdentity(); err != nil {
		return err
	}
	if r.MeasuredLatencyNs <= 0 || r.BaselineLatencyNs <= 0 || math.IsNaN(r.SpeedupRatio) || math.IsInf(r.SpeedupRatio, 0) || r.SpeedupRatio != float64(r.BaselineLatencyNs)/float64(r.MeasuredLatencyNs) {
		return fmt.Errorf("cudatrial: inconsistent trajectory timing")
	}
	if r.SHA256 == "" || r.checksum() != r.SHA256 {
		return fmt.Errorf("cudatrial: trajectory checksum mismatch")
	}
	return nil
}

func (r TrajectoryReceipt) validateIdentity() error {
	if !r.BottleneckState.Valid() || cudakernel.ValidateTargetArch(r.TargetArch) != nil || len(r.AppliedRules) == 0 || len(r.AppliedRules) > 128 {
		return fmt.Errorf("cudatrial: invalid trajectory category, target or rules")
	}
	for _, s := range []string{r.WorkloadKey, r.Domain, r.ArtifactDigest, r.CompilerKey, r.BaselineArtifactDigest, r.BaselineCompilerKey, r.InputDigest, r.PolicyID} {
		if !trajectoryDigest(s) {
			return fmt.Errorf("cudatrial: invalid trajectory digest")
		}
	}
	for _, rule := range r.AppliedRules {
		if rule == "" || len(rule) > 256 {
			return fmt.Errorf("cudatrial: invalid trajectory rule")
		}
	}
	return nil
}

// ObserveTrajectory derives event timings and device identity from a live
// device under an inherited GPU lock. Native producers must supply actual
// completed baseline/candidate receipts and an independent correctness result.
// This API checks consistency; it cannot authenticate arbitrary receipt claims.
// A nil device always returns PendingHardware and ErrUnqualifiedOutcome.
func ObserveTrajectory(device *cuda.Device, r TrajectoryReceipt, baseline, candidate Receipt) (TrajectoryReceipt, error) {
	r.Version = 1
	r.PendingHardware = true
	r.SpeedupRatio = 0
	r.MeasuredLatencyNs = 0
	r.BaselineLatencyNs = 0
	r.SHA256 = ""
	r.DeviceUUID = ""
	r.LockPID = 0
	r.BaselineArtifactDigest = ""
	r.BaselineCompilerKey = ""
	if device == nil {
		return r, errors.Join(ErrUnqualifiedOutcome, cudaerrors.ErrHardwarePending)
	}
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		return r, fmt.Errorf("%w: %w", ErrUnqualifiedOutcome, err)
	}
	a, b, err := admitTrajectoryReceipts(&r, baseline, candidate)
	if err != nil {
		return r, err
	}
	err = device.WithContext(func() error {
		major, minor, err := device.ComputeCapability()
		if err != nil {
			return err
		}
		if r.TargetArch != fmt.Sprintf("sm_%d%d", major, minor) {
			return fmt.Errorf("%w: device target mismatch", ErrUnqualifiedOutcome)
		}
		uuid, err := device.Raw().Uuid()
		if err != nil {
			return err
		}
		r.DeviceUUID = fmt.Sprintf("%x", uuid.Bytes)
		if r.DeviceUUID == strings.Repeat("0", 32) {
			return fmt.Errorf("%w: unavailable device UUID", ErrUnqualifiedOutcome)
		}
		if baseline.Provenance["device_uuid"] != r.DeviceUUID || candidate.Provenance["device_uuid"] != r.DeviceUUID {
			return fmt.Errorf("%w: event device mismatch", ErrUnqualifiedOutcome)
		}
		return nil
	})
	if err != nil {
		return r, fmt.Errorf("%w: %w", ErrUnqualifiedOutcome, err)
	}
	r.LockPID = owner.PID
	r.PendingHardware = false
	r.BaselineLatencyNs = int64(a)
	r.MeasuredLatencyNs = int64(b)
	r.SpeedupRatio = float64(a) / float64(b)
	r.AppliedRules = append([]string(nil), r.AppliedRules...)
	r.SHA256 = r.checksum()
	if err := r.Validate(); err != nil {
		return r, err
	}
	r.liveSeal = r.SHA256
	return r, nil
}

// HardwareObserved reports whether this exact value was admitted by the live
// constructor in this process. JSON decoding and caller mutation lose this
// proof. Persisted integrity alone never grants it.
func (r TrajectoryReceipt) HardwareObserved() bool {
	return r.liveSeal != "" && r.liveSeal == r.SHA256 && r.Validate() == nil
}

// TrajectoryMemory retains at most Capacity qualified observations. Its zero
// value refuses Record; construct it with NewTrajectoryMemory.
type TrajectoryMemory struct {
	mu       sync.RWMutex
	capacity int
	rows     []TrajectoryReceipt
}

// NewTrajectoryMemory creates a memory bounded to 1..4096 rows.
func NewTrajectoryMemory(capacity int) (*TrajectoryMemory, error) {
	if capacity <= 0 || capacity > 4096 {
		return nil, fmt.Errorf("cudatrial: invalid trajectory capacity")
	}
	return &TrajectoryMemory{capacity: capacity}, nil
}

// Record admits a validated receipt, replacing the oldest when capacity is full.
func (m *TrajectoryMemory) Record(r TrajectoryReceipt) error {
	if !r.HardwareObserved() {
		return ErrUnqualifiedOutcome
	}
	return m.record(r)
}

// ImportTrusted replays a persisted producer declaration after the caller has
// independently decided to trust its provenance. It does not establish live
// hardware observation. Do not use it to import synthetic timing as evidence.
func (m *TrajectoryMemory) ImportTrusted(r TrajectoryReceipt) error {
	r.liveSeal = ""
	return m.record(r)
}

func (m *TrajectoryMemory) record(r TrajectoryReceipt) error {
	if m == nil {
		return fmt.Errorf("cudatrial: nil trajectory memory")
	}
	if err := r.Validate(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.capacity <= 0 {
		return fmt.Errorf("cudatrial: uninitialized trajectory memory")
	}
	r.AppliedRules = append([]string(nil), r.AppliedRules...)
	for _, old := range m.rows {
		if old.SHA256 == r.SHA256 {
			return nil
		}
	}
	if len(m.rows) == m.capacity {
		rows := append(append([]TrajectoryReceipt(nil), m.rows[1:]...), r)
		b, err := json.Marshal(rows)
		if err != nil || len(b) > 2<<20 {
			return fmt.Errorf("cudatrial: trajectory memory byte limit")
		}
		m.rows = rows
		return nil
	}
	rows := append(append([]TrajectoryReceipt(nil), m.rows...), r)
	b, err := json.Marshal(rows)
	if err != nil || len(b) > 2<<20 {
		return fmt.Errorf("cudatrial: trajectory memory byte limit")
	}
	m.rows = append(m.rows, r)
	return nil
}

// RuleReward is a dimensionless exploration prior. Failures score -8; measured
// regressions have negative log2 speedup. Values are clamped to [-8,8].
type RuleReward struct {
	Rule         string
	Reward       float64
	Observations int
}

// Rank returns detached mean rewards from exact domain/category/target matches.
// It never changes an evaluator's latency, correctness or objective score.
func (m *TrajectoryMemory) Rank(domain string, family BottleneckFamily, target string) []RuleReward {
	return m.Replay(domain, family, target).Rewards
}

// TrajectoryReplay is an immutable-by-copy snapshot of consulted receipt IDs
// and their mean rewards. IDs refer to integrity checksums, not machine-code
// identity or authenticated device execution.
type TrajectoryReplay struct {
	ReceiptIDs []string
	Rewards    []RuleReward
}

// Replay freezes matching history under one lock, so its rewards and receipt
// lineage cannot disagree because of a concurrent Record.
func (m *TrajectoryMemory) Replay(domain string, family BottleneckFamily, target string) TrajectoryReplay {
	var snapshot TrajectoryReplay
	if m == nil {
		return snapshot
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	by := make(map[string]RuleReward)
	for _, r := range m.rows {
		if r.Domain != domain || r.BottleneckState != family || r.TargetArch != target {
			continue
		}
		snapshot.ReceiptIDs = append(snapshot.ReceiptIDs, r.SHA256)
		v := -8.0
		if r.VerifiedCorrect {
			v = math.Max(-8, math.Min(8, math.Log2(r.SpeedupRatio)))
		}
		seen := make(map[string]bool)
		for _, rule := range r.AppliedRules {
			if seen[rule] {
				continue
			}
			seen[rule] = true
			x := by[rule]
			x.Rule = rule
			x.Reward += v
			x.Observations++
			by[rule] = x
		}
	}
	var out []RuleReward
	for _, x := range by {
		x.Reward /= float64(x.Observations)
		out = append(out, x)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Reward != out[j].Reward {
			return out[i].Reward > out[j].Reward
		}
		return out[i].Rule < out[j].Rule
	})
	snapshot.Rewards = out
	return snapshot
}
