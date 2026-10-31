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
	// ErrPGODigestMismatch reports that a cached cubin binary is empty or its
	// SHA-256 digest does not match the promotion proof's CubinSHA256.
	ErrPGODigestMismatch = errors.New("autotune: pgo cubin sha256 digest mismatch")

	// ErrPGOTokenMismatch reports that a promotion proof does not carry exact
	// greedy token-ID trajectory agreement.
	ErrPGOTokenMismatch = errors.New("autotune: pgo promotion proof requires matched tokens")

	// ErrPGOHoldoutGate reports that a promotion proof failed the held-out ABBA
	// statistical promotion gate.
	ErrPGOHoldoutGate = errors.New("autotune: pgo promotion proof failed holdout statistical gate")

	// ErrPGOKeyMismatch reports that the key stored inside a cached promotion
	// proof does not match the lookup or storage PGOCacheKey.
	ErrPGOKeyMismatch = errors.New("autotune: pgo cache key mismatch")
)

const (
	minPGOHoldoutBlocks = 6
	maxPGOOneSidedP     = 0.05
)

// PGOCacheKey uniquely identifies a hardware-and-model-specific compiled PGO
// schedule cache slot. Any change to the GPU target architecture, driver,
// CUDA version, model weight/quantization digest, execution plan, or shape
// bucket maps to a distinct [PGOCacheKey.CacheID].
type PGOCacheKey struct {
	Arch          string `json:"arch"`
	DriverVersion string `json:"driver_version"`
	CUDAVersion   string `json:"cuda_version"`
	WeightDigest  string `json:"weight_digest"`
	PlanKey       string `json:"plan_key"`
	ShapeBucket   int    `json:"shape_bucket"`
}

// Validate checks that all components of k are populated.
func (k PGOCacheKey) Validate() error {
	if k.Arch == "" {
		return errors.New("pgo cache key: empty arch")
	}
	if k.DriverVersion == "" {
		return errors.New("pgo cache key: empty driver version")
	}
	if k.CUDAVersion == "" {
		return errors.New("pgo cache key: empty cuda version")
	}
	if k.WeightDigest == "" {
		return errors.New("pgo cache key: empty weight digest")
	}
	if k.PlanKey == "" {
		return errors.New("pgo cache key: empty plan key")
	}
	if k.ShapeBucket <= 0 {
		return fmt.Errorf("pgo cache key: invalid shape bucket %d", k.ShapeBucket)
	}
	return nil
}

// CacheID returns the deterministic hex SHA-256 of the canonical key tuple.
func (k PGOCacheKey) CacheID() string {
	b, err := json.Marshal(k)
	if err != nil {
		panic(err) // PGOCacheKey contains only plain string and int fields
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func comparePGOCacheKeys(a, b PGOCacheKey) int {
	if c := cmp.Compare(a.Arch, b.Arch); c != 0 {
		return c
	}
	if c := cmp.Compare(a.DriverVersion, b.DriverVersion); c != 0 {
		return c
	}
	if c := cmp.Compare(a.CUDAVersion, b.CUDAVersion); c != 0 {
		return c
	}
	if c := cmp.Compare(a.WeightDigest, b.WeightDigest); c != 0 {
		return c
	}
	if c := cmp.Compare(a.PlanKey, b.PlanKey); c != 0 {
		return c
	}
	return cmp.Compare(a.ShapeBucket, b.ShapeBucket)
}

// PGOPromotionProof records the cryptographic digests and statistical
// qualification evidence required to admit a compiled PGO schedule into
// [VerifiedPGOCache].
type PGOPromotionProof struct {
	Key            PGOCacheKey `json:"key"`
	CandidateName  string      `json:"candidate_name"`
	ScheduleJSON   []byte      `json:"schedule_json,omitempty"`
	CubinSHA256    [32]byte    `json:"cubin_sha256"`
	SourceSHA256   [32]byte    `json:"source_sha256"`
	ScreenBlocks   int         `json:"screen_blocks"`
	HoldoutBlocks  int         `json:"holdout_blocks"`
	ScreenRatio    float64     `json:"screen_ratio"`
	HoldoutRatio   float64     `json:"holdout_ratio"`
	AANullRatio    float64     `json:"aa_null_ratio"`
	OneSidedPValue float64     `json:"one_sided_p_value"`
	TokensMatched  bool        `json:"tokens_matched"`
	PromotedAt     time.Time   `json:"promoted_at"`
}

func verifyPGOProofAndCubin(key PGOCacheKey, proof PGOPromotionProof, cubin []byte) error {
	if err := key.Validate(); err != nil {
		return err
	}
	if proof.Key != (PGOCacheKey{}) && proof.Key != key {
		return fmt.Errorf("%w: proof key %+v does not match %+v", ErrPGOKeyMismatch, proof.Key, key)
	}
	if !proof.TokensMatched {
		return ErrPGOTokenMismatch
	}
	if proof.HoldoutBlocks < minPGOHoldoutBlocks {
		return fmt.Errorf("%w: holdout blocks %d < %d", ErrPGOHoldoutGate, proof.HoldoutBlocks, minPGOHoldoutBlocks)
	}
	if math.IsNaN(proof.OneSidedPValue) || math.IsInf(proof.OneSidedPValue, 0) || proof.OneSidedPValue < 0 || proof.OneSidedPValue > maxPGOOneSidedP {
		return fmt.Errorf("%w: one-sided p-value %g > %g", ErrPGOHoldoutGate, proof.OneSidedPValue, maxPGOOneSidedP)
	}
	if math.IsNaN(proof.HoldoutRatio) || math.IsInf(proof.HoldoutRatio, 0) || proof.HoldoutRatio <= 1.0 {
		return fmt.Errorf("%w: holdout ratio %g <= 1.0", ErrPGOHoldoutGate, proof.HoldoutRatio)
	}
	if len(cubin) == 0 {
		return fmt.Errorf("%w: empty cubin", ErrPGODigestMismatch)
	}
	gotSum := sha256.Sum256(cubin)
	if gotSum != proof.CubinSHA256 {
		return fmt.Errorf("%w: cubin sha256 got %x, proof wants %x", ErrPGODigestMismatch, gotSum, proof.CubinSHA256)
	}
	return nil
}

// VerifiedPGOCache manages an on-disk directory storing verified PGO promotion
// proofs (<cache_id>.json) and compiled cubins (<cache_id>.cubin).
type VerifiedPGOCache struct {
	Dir string
	mu  sync.RWMutex
}

// NewVerifiedPGOCache returns a [VerifiedPGOCache] rooted at dir.
func NewVerifiedPGOCache(dir string) *VerifiedPGOCache {
	return &VerifiedPGOCache{Dir: dir}
}

// Store verifies the promotion proof and cubin digest and atomically writes
// <cache_id>.cubin and <cache_id>.json into c.Dir.
func (c *VerifiedPGOCache) Store(key PGOCacheKey, proof PGOPromotionProof, cubin []byte) error {
	if c == nil || c.Dir == "" {
		return errors.New("pgo cache store: empty cache directory")
	}
	if err := verifyPGOProofAndCubin(key, proof, cubin); err != nil {
		return fmt.Errorf("pgo cache store: %w", err)
	}
	proof.Key = key
	if proof.PromotedAt.IsZero() {
		proof.PromotedAt = time.Now().UTC()
	}
	metaBytes, err := json.MarshalIndent(proof, "", "  ")
	if err != nil {
		return fmt.Errorf("pgo cache store: marshal proof: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if err := os.MkdirAll(c.Dir, 0o755); err != nil {
		return fmt.Errorf("pgo cache store: mkdir: %w", err)
	}
	id := key.CacheID()
	cubinPath := filepath.Join(c.Dir, id+".cubin")
	jsonPath := filepath.Join(c.Dir, id+".json")

	if err := writeFileAtomic(c.Dir, id+".cubin.tmp.*", cubinPath, cubin); err != nil {
		return fmt.Errorf("pgo cache store: write cubin: %w", err)
	}
	if err := writeFileAtomic(c.Dir, id+".json.tmp.*", jsonPath, metaBytes); err != nil {
		_ = os.Remove(cubinPath)
		return fmt.Errorf("pgo cache store: write metadata: %w", err)
	}
	return nil
}

// LoadVerified loads the promotion proof and cubin for key, recomputes the
// cubin's SHA-256 digest, and verifies key binding and holdout promotion
// invariants before returning.
func (c *VerifiedPGOCache) LoadVerified(key PGOCacheKey) (PGOPromotionProof, []byte, error) {
	if c == nil || c.Dir == "" {
		return PGOPromotionProof{}, nil, errors.New("pgo cache load: empty cache directory")
	}
	if err := key.Validate(); err != nil {
		return PGOPromotionProof{}, nil, fmt.Errorf("pgo cache load: %w", err)
	}
	id := key.CacheID()
	jsonPath := filepath.Join(c.Dir, id+".json")
	cubinPath := filepath.Join(c.Dir, id+".cubin")

	c.mu.RLock()
	metaBytes, metaErr := os.ReadFile(jsonPath)
	var cubin []byte
	var cubinErr error
	if metaErr == nil {
		cubin, cubinErr = os.ReadFile(cubinPath)
	}
	c.mu.RUnlock()

	if metaErr != nil {
		return PGOPromotionProof{}, nil, fmt.Errorf("pgo cache load %s: read proof: %w", id, metaErr)
	}
	if cubinErr != nil {
		return PGOPromotionProof{}, nil, fmt.Errorf("pgo cache load %s: read cubin: %w", id, cubinErr)
	}

	var proof PGOPromotionProof
	if err := json.Unmarshal(metaBytes, &proof); err != nil {
		return PGOPromotionProof{}, nil, fmt.Errorf("pgo cache load %s: unmarshal proof: %w", id, err)
	}
	if proof.Key != key {
		return PGOPromotionProof{}, nil, fmt.Errorf("pgo cache load %s: %w: got %+v, want %+v", id, ErrPGOKeyMismatch, proof.Key, key)
	}
	if err := verifyPGOProofAndCubin(key, proof, cubin); err != nil {
		return PGOPromotionProof{}, nil, fmt.Errorf("pgo cache load %s: %w", id, err)
	}
	return proof, cubin, nil
}

// Invalidate removes the cached metadata and cubin files for key if present.
func (c *VerifiedPGOCache) Invalidate(key PGOCacheKey) error {
	if c == nil || c.Dir == "" {
		return errors.New("pgo cache invalidate: empty cache directory")
	}
	id := key.CacheID()
	c.mu.Lock()
	defer c.mu.Unlock()

	var errs []error
	for _, ext := range []string{".json", ".cubin"} {
		if err := os.Remove(filepath.Join(c.Dir, id+ext)); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// List returns all valid keys currently stored in c.Dir in deterministic order.
func (c *VerifiedPGOCache) List() ([]PGOCacheKey, error) {
	if c == nil || c.Dir == "" {
		return nil, errors.New("pgo cache list: empty cache directory")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	entries, err := os.ReadDir(c.Dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("pgo cache list: %w", err)
	}
	var keys []PGOCacheKey
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".json") || strings.Contains(name, ".tmp.") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(c.Dir, name))
		if err != nil {
			return nil, fmt.Errorf("pgo cache list %s: %w", name, err)
		}
		var proof PGOPromotionProof
		if err := json.Unmarshal(b, &proof); err != nil {
			return nil, fmt.Errorf("pgo cache list %s: %w", name, err)
		}
		keys = append(keys, proof.Key)
	}
	slices.SortFunc(keys, comparePGOCacheKeys)
	return keys, nil
}

func writeFileAtomic(dir, pattern, dst string, data []byte) error {
	f, err := os.CreateTemp(dir, pattern)
	if err != nil {
		return err
	}
	tmpName := f.Name()
	cleanup := func() {
		_ = f.Close()
		_ = os.Remove(tmpName)
	}
	if _, err := f.Write(data); err != nil {
		cleanup()
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, dst); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	return nil
}
