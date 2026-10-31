package autotune

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
)

var (
	// ErrDistPGOQuorumIncomplete reports that a distributed PGO candidate does
	// not carry votes from all ranks in the target topology.
	ErrDistPGOQuorumIncomplete = errors.New("autotune: distributed pgo quorum vote count does not match topology world size")

	// ErrDistPGOHoldoutUnverified reports that at least one rank did not pass
	// the held-out ABBA statistical promotion gate.
	ErrDistPGOHoldoutUnverified = errors.New("autotune: distributed pgo rank vote failed holdout verification")

	// ErrDistPGOCoherenceFailed reports that at least one rank detected a
	// memory or switch-fabric coherence violation.
	ErrDistPGOCoherenceFailed = errors.New("autotune: distributed pgo rank vote failed coherence check")

	// ErrDistPGOTrajectoryMismatch reports that a rank's greedy token-ID or
	// numerical trajectory digest is empty or diverges from peer ranks.
	ErrDistPGOTrajectoryMismatch = errors.New("autotune: distributed pgo trajectory digest empty or divergent across ranks")

	// ErrDistPGOStragglerRegression reports that the slowest rank's latency
	// (MaxRankLatencyUs) does not improve over BaselineLatencyUs.
	ErrDistPGOStragglerRegression = errors.New("autotune: distributed pgo straggler max rank latency does not improve over baseline")
)

// DistTopologyKey uniquely identifies a multi-GPU fabric topology for
// distributed e-graph and PGO schedule caching.
type DistTopologyKey struct {
	WorldSize        int     `json:"world_size"`
	Arch             string  `json:"arch,omitempty"`
	SwitchBWGBps     float64 `json:"switch_bw_gbps,omitempty"`
	SwitchLatencyUs  float64 `json:"switch_latency_us,omitempty"`
	TensorCoreTFLOPS float64 `json:"tensor_core_tflops,omitempty"`
	Interconnect     string  `json:"interconnect,omitempty"`
}

// Validate checks that the topology key has a valid WorldSize and non-negative
// fabric parameters.
func (t DistTopologyKey) Validate() error {
	if t.WorldSize <= 0 {
		return fmt.Errorf("dist topology key: invalid world size %d", t.WorldSize)
	}
	if t.SwitchBWGBps < 0 || math.IsNaN(t.SwitchBWGBps) || math.IsInf(t.SwitchBWGBps, 0) {
		return fmt.Errorf("dist topology key: invalid switch bandwidth %g", t.SwitchBWGBps)
	}
	if t.SwitchLatencyUs < 0 || math.IsNaN(t.SwitchLatencyUs) || math.IsInf(t.SwitchLatencyUs, 0) {
		return fmt.Errorf("dist topology key: invalid switch latency %g", t.SwitchLatencyUs)
	}
	return nil
}

// DistPGOCacheKey uniquely identifies a topology-aware multi-rank PGO schedule
// cache slot. Any change to WorldSize, fabric bandwidth/latency, GPU
// architecture, driver/CUDA version, model weights, plan key, or batch shape
// bucket produces a distinct [DistPGOCacheKey.CacheID].
type DistPGOCacheKey struct {
	Topology       DistTopologyKey `json:"topology"`
	Arch           string          `json:"arch,omitempty"`
	DriverVersion  string          `json:"driver_version,omitempty"`
	CUDAVersion    string          `json:"cuda_version,omitempty"`
	WeightDigest   string          `json:"weight_digest,omitempty"`
	PlanKey        string          `json:"plan_key"`
	ShapeBucket    int             `json:"shape_bucket"`
	CollectiveRule string          `json:"collective_rule,omitempty"`
}

// Validate checks that the distributed cache key is well-formed.
func (k DistPGOCacheKey) Validate() error {
	if err := k.Topology.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(k.PlanKey) == "" {
		return errors.New("dist pgo cache key: empty plan key")
	}
	if k.ShapeBucket <= 0 {
		return fmt.Errorf("dist pgo cache key: invalid shape bucket %d", k.ShapeBucket)
	}
	return nil
}

// CacheID returns the deterministic hex SHA-256 digest of the canonical key.
func (k DistPGOCacheKey) CacheID() string {
	b, err := json.Marshal(k)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func compareDistPGOCacheKeys(a, b DistPGOCacheKey) int {
	if c := cmp.Compare(a.Topology.WorldSize, b.Topology.WorldSize); c != 0 {
		return c
	}
	if c := cmp.Compare(a.Topology.Arch, b.Topology.Arch); c != 0 {
		return c
	}
	if c := cmp.Compare(a.Topology.SwitchBWGBps, b.Topology.SwitchBWGBps); c != 0 {
		return c
	}
	if c := cmp.Compare(a.Topology.SwitchLatencyUs, b.Topology.SwitchLatencyUs); c != 0 {
		return c
	}
	if c := cmp.Compare(a.Arch, b.Arch); c != 0 {
		return c
	}
	if c := cmp.Compare(a.PlanKey, b.PlanKey); c != 0 {
		return c
	}
	if c := cmp.Compare(a.ShapeBucket, b.ShapeBucket); c != 0 {
		return c
	}
	return cmp.Compare(a.CacheID(), b.CacheID())
}

// RankVerificationVote records a single rank's post-qualification verification
// vote for a distributed schedule candidate.
type RankVerificationVote struct {
	Rank              int     `json:"rank"`
	HoldoutVerified   bool    `json:"holdout_verified"`
	CoherenceClean    bool    `json:"coherence_clean"`
	TrajectoryDigest  string  `json:"trajectory_digest"`
	LatencyUs         float64 `json:"latency_us"`
	BaselineLatencyUs float64 `json:"baseline_latency_us,omitempty"`
	HoldoutRatio      float64 `json:"holdout_ratio,omitempty"`
	OneSidedPValue    float64 `json:"one_sided_p_value,omitempty"`
}

// DistPGOCacheEntry stores a quorum-verified distributed schedule, its
// per-rank verification votes, latency bounds across ranks, and optional
// compiled cubin/PTX artifacts.
type DistPGOCacheEntry struct {
	Key               DistPGOCacheKey        `json:"key"`
	CandidateName     string                 `json:"candidate_name"`
	CollectiveRule    string                 `json:"collective_rule,omitempty"`
	PlanSignature     string                 `json:"plan_signature,omitempty"`
	ScheduleJSON      []byte                 `json:"schedule_json,omitempty"`
	PTX               string                 `json:"ptx,omitempty"`
	Cubin             []byte                 `json:"cubin,omitempty"`
	CubinSHA256       [32]byte               `json:"cubin_sha256,omitempty"`
	SourceSHA256      [32]byte               `json:"source_sha256,omitempty"`
	BaselineLatencyUs float64                `json:"baseline_latency_us"`
	MaxRankLatencyUs  float64                `json:"max_rank_latency_us"`
	MinRankLatencyUs  float64                `json:"min_rank_latency_us,omitempty"`
	MeanRankLatencyUs float64                `json:"mean_rank_latency_us,omitempty"`
	SpeedupRatio      float64                `json:"speedup_ratio,omitempty"`
	TrajectoryDigest  string                 `json:"trajectory_digest"`
	RankVotes         []RankVerificationVote `json:"rank_votes"`
	PromotedAt        time.Time              `json:"promoted_at"`
}

func validateAndNormalizeDistEntry(entry *DistPGOCacheEntry) error {
	if entry == nil {
		return errors.New("dist pgo cache: nil entry")
	}
	if err := entry.Key.Validate(); err != nil {
		return err
	}
	worldSize := entry.Key.Topology.WorldSize
	if len(entry.RankVotes) != worldSize {
		return fmt.Errorf("%w: got %d rank votes, want %d", ErrDistPGOQuorumIncomplete, len(entry.RankVotes), worldSize)
	}

	var (
		refDigest   string
		maxVoteLat  float64
		minVoteLat  float64
		sumVoteLat  float64
		haveVoteLat bool
	)
	for i, vote := range entry.RankVotes {
		if !vote.HoldoutVerified {
			return fmt.Errorf("%w: rank %d (index %d) HoldoutVerified=false", ErrDistPGOHoldoutUnverified, vote.Rank, i)
		}
		if !vote.CoherenceClean {
			return fmt.Errorf("%w: rank %d (index %d) CoherenceClean=false", ErrDistPGOCoherenceFailed, vote.Rank, i)
		}
		digest := strings.TrimSpace(vote.TrajectoryDigest)
		if digest == "" {
			return fmt.Errorf("%w: rank %d (index %d) has empty TrajectoryDigest", ErrDistPGOTrajectoryMismatch, vote.Rank, i)
		}
		if i == 0 {
			refDigest = digest
		} else if digest != refDigest {
			return fmt.Errorf("%w: rank %d digest %q != rank 0 digest %q", ErrDistPGOTrajectoryMismatch, vote.Rank, digest, refDigest)
		}
		if vote.LatencyUs > 0 && !math.IsNaN(vote.LatencyUs) && !math.IsInf(vote.LatencyUs, 0) {
			if !haveVoteLat || vote.LatencyUs > maxVoteLat {
				maxVoteLat = vote.LatencyUs
			}
			if !haveVoteLat || vote.LatencyUs < minVoteLat {
				minVoteLat = vote.LatencyUs
			}
			sumVoteLat += vote.LatencyUs
			haveVoteLat = true
		}
	}

	if entry.TrajectoryDigest != "" && strings.TrimSpace(entry.TrajectoryDigest) != refDigest {
		return fmt.Errorf("%w: entry digest %q != rank quorum digest %q", ErrDistPGOTrajectoryMismatch, entry.TrajectoryDigest, refDigest)
	}
	entry.TrajectoryDigest = refDigest

	if haveVoteLat {
		if entry.MaxRankLatencyUs <= 0 || maxVoteLat > entry.MaxRankLatencyUs {
			entry.MaxRankLatencyUs = maxVoteLat
		}
		if entry.MinRankLatencyUs <= 0 {
			entry.MinRankLatencyUs = minVoteLat
		}
		if entry.MeanRankLatencyUs <= 0 {
			entry.MeanRankLatencyUs = sumVoteLat / float64(worldSize)
		}
	}

	if math.IsNaN(entry.BaselineLatencyUs) || math.IsInf(entry.BaselineLatencyUs, 0) || entry.BaselineLatencyUs <= 0 {
		return fmt.Errorf("%w: invalid BaselineLatencyUs %g", ErrDistPGOStragglerRegression, entry.BaselineLatencyUs)
	}
	if math.IsNaN(entry.MaxRankLatencyUs) || math.IsInf(entry.MaxRankLatencyUs, 0) || entry.MaxRankLatencyUs <= 0 {
		return fmt.Errorf("%w: invalid MaxRankLatencyUs %g", ErrDistPGOStragglerRegression, entry.MaxRankLatencyUs)
	}
	if !(entry.MaxRankLatencyUs < entry.BaselineLatencyUs) {
		return fmt.Errorf("%w: max rank latency %.3f us >= baseline %.3f us", ErrDistPGOStragglerRegression, entry.MaxRankLatencyUs, entry.BaselineLatencyUs)
	}
	entry.SpeedupRatio = entry.BaselineLatencyUs / entry.MaxRankLatencyUs

	if len(entry.Cubin) > 0 {
		sum := sha256.Sum256(entry.Cubin)
		if entry.CubinSHA256 == ([32]byte{}) {
			entry.CubinSHA256 = sum
		} else if entry.CubinSHA256 != sum {
			return fmt.Errorf("%w: cubin sha256 got %x, entry wants %x", ErrPGODigestMismatch, sum, entry.CubinSHA256)
		}
	} else if entry.CubinSHA256 != ([32]byte{}) {
		return fmt.Errorf("%w: non-zero cubin sha256 with empty cubin payload", ErrPGODigestMismatch)
	}

	if entry.PromotedAt.IsZero() {
		entry.PromotedAt = time.Now().UTC()
	}
	return nil
}

// DistVerifiedPGOCache manages a topology-keyed multi-rank PGO schedule cache
// backed by atomic on-disk persistence and in-memory indexing.
type DistVerifiedPGOCache struct {
	Dir     string
	mu      sync.RWMutex
	entries map[string]DistPGOCacheEntry
}

// NewDistVerifiedPGOCache returns a [DistVerifiedPGOCache] rooted at dir.
func NewDistVerifiedPGOCache(dir string) *DistVerifiedPGOCache {
	return &DistVerifiedPGOCache{
		Dir:     dir,
		entries: make(map[string]DistPGOCacheEntry),
	}
}

// RecordQuorumWinner enforces all-rank consensus before persisting entry:
//  1. len(entry.RankVotes) == entry.Key.Topology.WorldSize
//  2. Every rank vote has HoldoutVerified == true and CoherenceClean == true
//  3. Every rank vote's TrajectoryDigest is non-empty and identical across all ranks
//  4. entry.MaxRankLatencyUs < entry.BaselineLatencyUs (even the slowest straggler rank improves over baseline).
func (c *DistVerifiedPGOCache) RecordQuorumWinner(entry DistPGOCacheEntry) error {
	if c == nil {
		return errors.New("dist pgo cache record: nil cache")
	}
	norm := cloneDistPGOCacheEntry(entry)
	if err := validateAndNormalizeDistEntry(&norm); err != nil {
		return fmt.Errorf("dist pgo cache record: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	id := norm.Key.CacheID()
	if c.Dir != "" {
		if err := os.MkdirAll(c.Dir, 0o755); err != nil {
			return fmt.Errorf("dist pgo cache record: mkdir: %w", err)
		}
		cubinPath := filepath.Join(c.Dir, id+".dist.cubin")
		jsonPath := filepath.Join(c.Dir, id+".dist.json")

		diskEntry := norm
		diskEntry.Cubin = nil
		metaBytes, err := json.MarshalIndent(diskEntry, "", "  ")
		if err != nil {
			return fmt.Errorf("dist pgo cache record: marshal entry: %w", err)
		}
		if len(norm.Cubin) > 0 {
			if err := writeFileAtomic(c.Dir, id+".dist.cubin.tmp.*", cubinPath, norm.Cubin); err != nil {
				return fmt.Errorf("dist pgo cache record: write cubin: %w", err)
			}
		}
		if err := writeFileAtomic(c.Dir, id+".dist.json.tmp.*", jsonPath, metaBytes); err != nil {
			if len(norm.Cubin) > 0 {
				_ = os.Remove(cubinPath)
			}
			return fmt.Errorf("dist pgo cache record: write metadata: %w", err)
		}
	}

	if c.entries == nil {
		c.entries = make(map[string]DistPGOCacheEntry)
	}
	c.entries[id] = norm
	return nil
}

// LookupVerified loads and re-verifies the quorum-promoted distributed PGO
// entry for key. When key is not present in the cache, LookupVerified returns
// (DistPGOCacheEntry{}, false, nil). If a persisted entry was tampered with or
// violates any quorum invariant, LookupVerified returns a non-nil error.
func (c *DistVerifiedPGOCache) LookupVerified(key DistPGOCacheKey) (DistPGOCacheEntry, bool, error) {
	if c == nil {
		return DistPGOCacheEntry{}, false, errors.New("dist pgo cache lookup: nil cache")
	}
	if err := key.Validate(); err != nil {
		return DistPGOCacheEntry{}, false, fmt.Errorf("dist pgo cache lookup: %w", err)
	}
	id := key.CacheID()

	if c.Dir != "" {
		jsonPath := filepath.Join(c.Dir, id+".dist.json")
		cubinPath := filepath.Join(c.Dir, id+".dist.cubin")

		c.mu.RLock()
		metaBytes, metaErr := os.ReadFile(jsonPath)
		var cubinBytes []byte
		var cubinErr error
		if metaErr == nil {
			cubinBytes, cubinErr = os.ReadFile(cubinPath)
		}
		c.mu.RUnlock()

		if errors.Is(metaErr, os.ErrNotExist) {
			return DistPGOCacheEntry{}, false, nil
		}
		if metaErr != nil {
			return DistPGOCacheEntry{}, false, fmt.Errorf("dist pgo cache lookup %s: read metadata: %w", id, metaErr)
		}

		var entry DistPGOCacheEntry
		if err := json.Unmarshal(metaBytes, &entry); err != nil {
			return DistPGOCacheEntry{}, false, fmt.Errorf("dist pgo cache lookup %s: unmarshal metadata: %w", id, err)
		}
		if entry.Key != key {
			return DistPGOCacheEntry{}, false, fmt.Errorf("dist pgo cache lookup %s: %w: got %+v, want %+v", id, ErrPGOKeyMismatch, entry.Key, key)
		}
		if cubinErr == nil {
			entry.Cubin = cubinBytes
		} else if !errors.Is(cubinErr, os.ErrNotExist) {
			return DistPGOCacheEntry{}, false, fmt.Errorf("dist pgo cache lookup %s: read cubin: %w", id, cubinErr)
		}
		if err := validateAndNormalizeDistEntry(&entry); err != nil {
			return DistPGOCacheEntry{}, false, fmt.Errorf("dist pgo cache lookup %s: %w", id, err)
		}
		return entry, true, nil
	}

	c.mu.RLock()
	entry, ok := c.entries[id]
	c.mu.RUnlock()
	if !ok {
		return DistPGOCacheEntry{}, false, nil
	}
	cp := cloneDistPGOCacheEntry(entry)
	if cp.Key != key {
		return DistPGOCacheEntry{}, false, fmt.Errorf("dist pgo cache lookup %s: %w", id, ErrPGOKeyMismatch)
	}
	if err := validateAndNormalizeDistEntry(&cp); err != nil {
		return DistPGOCacheEntry{}, false, fmt.Errorf("dist pgo cache lookup %s: %w", id, err)
	}
	return cp, true, nil
}

// Invalidate removes the cached distributed schedule entry for key.
func (c *DistVerifiedPGOCache) Invalidate(key DistPGOCacheKey) error {
	if c == nil {
		return errors.New("dist pgo cache invalidate: nil cache")
	}
	id := key.CacheID()
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.entries, id)
	if c.Dir == "" {
		return nil
	}
	var errs []error
	for _, ext := range []string{".dist.json", ".dist.cubin"} {
		if err := os.Remove(filepath.Join(c.Dir, id+ext)); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// List returns all valid distributed cache keys currently stored in c in
// deterministic order.
func (c *DistVerifiedPGOCache) List() ([]DistPGOCacheKey, error) {
	if c == nil {
		return nil, errors.New("dist pgo cache list: nil cache")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.Dir == "" {
		keys := make([]DistPGOCacheKey, 0, len(c.entries))
		for _, e := range c.entries {
			keys = append(keys, e.Key)
		}
		slices.SortFunc(keys, compareDistPGOCacheKeys)
		return keys, nil
	}

	entries, err := os.ReadDir(c.Dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("dist pgo cache list: %w", err)
	}
	var keys []DistPGOCacheKey
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".dist.json") || strings.Contains(name, ".tmp.") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(c.Dir, name))
		if err != nil {
			return nil, fmt.Errorf("dist pgo cache list %s: %w", name, err)
		}
		var e DistPGOCacheEntry
		if err := json.Unmarshal(b, &e); err != nil {
			return nil, fmt.Errorf("dist pgo cache list %s: %w", name, err)
		}
		keys = append(keys, e.Key)
	}
	slices.SortFunc(keys, compareDistPGOCacheKeys)
	return keys, nil
}

func cloneDistPGOCacheEntry(e DistPGOCacheEntry) DistPGOCacheEntry {
	out := e
	out.ScheduleJSON = slices.Clone(e.ScheduleJSON)
	out.Cubin = slices.Clone(e.Cubin)
	out.RankVotes = slices.Clone(e.RankVotes)
	return out
}
