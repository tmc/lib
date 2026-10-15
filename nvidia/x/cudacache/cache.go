package cudacache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

var (
	// ErrNotFound is returned when an artifact or receipt is not found in the cache.
	ErrNotFound = errors.New("cudacache: entry not found")

	// ErrCorruptArtifact is returned when an artifact's payload fails SHA-256 digest verification.
	ErrCorruptArtifact = errors.New("cudacache: corrupt artifact")

	// ErrIncompatibleTarget is returned when an artifact cannot run on the requested target architecture.
	ErrIncompatibleTarget = errors.New("cudacache: incompatible target architecture")

	// ErrUnqualifiedOutcome is returned when an artifact or receipt lacks valid qualification.
	ErrUnqualifiedOutcome = errors.New("cudacache: unqualified outcome")

	// ErrCorruptReceipt is returned when a receipt cannot be deserialized or validated.
	ErrCorruptReceipt = errors.New("cudacache: corrupt receipt")
)

const (
	// DefaultMaxArtifactBytes is the default maximum size of an artifact file (128 MB).
	DefaultMaxArtifactBytes int64 = 128 << 20

	// DefaultMaxRemoteDownloadBytes is the default maximum size for remote downloads (64 MB).
	DefaultMaxRemoteDownloadBytes int64 = 64 << 20
)

// Config configures the local and optional remote cache.
type Config struct {
	// RootDir is the local filesystem directory for storing cached artifacts and receipts.
	RootDir string

	// MaxArtifactBytes limits the maximum size in bytes of an artifact stored or read locally.
	MaxArtifactBytes int64

	// MaxRemoteDownloadBytes limits the maximum size in bytes of a download from RemoteSource.
	MaxRemoteDownloadBytes int64

	// Remote is an optional remote artifact and receipt provider.
	Remote RemoteSource

	// Trust specifies the trust policy applied to remote artifacts.
	Trust TrustPolicy
}

// Cache provides content-addressed storage and retrieval for compiled artifacts and receipts.
type Cache struct {
	rootDir          string
	artifactsDir     string
	receiptsDir      string
	quarantineDir    string
	maxArtifactBytes int64
	maxRemoteBytes   int64
	remote           RemoteSource
	trust            TrustPolicy
	trajectoryMu     sync.Mutex // Serializes trajectory publication and quarantine.
	publicationMu    sync.Mutex // Keeps artifact and receipt disk and memory publication together.

	// flightMu guards concurrent writes/downloads for identical keys.
	flightMu sync.Mutex
	flight   map[string]*flightCall

	// memMu caches validated artifacts in memory for fast lookup.
	memMu        sync.RWMutex
	memArtifacts map[string]*cudakernel.Artifact
	memReceipts  map[string]*SearchReceipt
}

type flightCall struct {
	done chan struct{}
	val  any
	err  error
}

// New creates a new Cache rooted at cfg.RootDir.
func New(cfg Config) (*Cache, error) {
	if cfg.MaxArtifactBytes < 0 || cfg.MaxRemoteDownloadBytes < 0 || cfg.MaxArtifactBytes == math.MaxInt64 || cfg.MaxRemoteDownloadBytes == math.MaxInt64 {
		return nil, errors.New("cudacache: invalid byte limit")
	}
	trust, err := copyTrust(cfg.Trust)
	if err != nil {
		return nil, err
	}
	if cfg.Remote != nil && trust.authenticated() {
		if len(trust.Signers) == 0 {
			return nil, fmt.Errorf("%w: pinned signer keys required", ErrUntrustedSource)
		}
		if _, ok := cfg.Remote.(RemoteProofSource); !ok {
			return nil, fmt.Errorf("%w: proof transport unavailable", ErrUntrustedSource)
		}
	}
	if cfg.RootDir == "" {
		return nil, errors.New("cudacache: root directory cannot be empty")
	}

	maxArt := cfg.MaxArtifactBytes
	if maxArt <= 0 {
		maxArt = DefaultMaxArtifactBytes
	}
	maxRemote := cfg.MaxRemoteDownloadBytes
	if maxRemote <= 0 {
		maxRemote = DefaultMaxRemoteDownloadBytes
	}

	c := &Cache{
		rootDir:          cfg.RootDir,
		artifactsDir:     filepath.Join(cfg.RootDir, "artifacts"),
		receiptsDir:      filepath.Join(cfg.RootDir, "receipts"),
		quarantineDir:    filepath.Join(cfg.RootDir, "quarantine"),
		maxArtifactBytes: maxArt,
		maxRemoteBytes:   maxRemote,
		remote:           cfg.Remote,
		trust:            trust,
		flight:           make(map[string]*flightCall),
		memArtifacts:     make(map[string]*cudakernel.Artifact),
		memReceipts:      make(map[string]*SearchReceipt),
	}
	if remote, ok := cfg.Remote.(*HTTPRemoteSource); ok && remote != nil {
		copy := *remote
		copy.maxDownloadBytes = maxRemote
		c.remote = &copy
	}

	for _, dir := range []string{c.artifactsDir, c.receiptsDir, c.quarantineDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("cudacache: create directory %q: %w", dir, err)
		}
	}

	return c, nil
}

// PutArtifact stores an artifact in the local content-addressed cache under key.
// The write is atomic and verified by SHA-256 digest before final rename.
func (c *Cache) PutArtifact(ctx context.Context, key string, art *cudakernel.Artifact) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if art == nil {
		return errors.New("cudacache: nil artifact")
	}
	if err := validateKey(key); err != nil {
		return err
	}
	if err := art.Validate(); err != nil {
		return fmt.Errorf("cudacache: invalid artifact: %w", err)
	}

	data, err := json.Marshal(art)
	if err != nil {
		return fmt.Errorf("cudacache: marshal artifact: %w", err)
	}

	if int64(len(data)) > c.maxArtifactBytes {
		return fmt.Errorf("cudacache: artifact size (%d bytes) exceeds limit (%d bytes)", len(data), c.maxArtifactBytes)
	}
	art, err = c.validateAndDecodeArtifact(key, data, "")
	if err != nil {
		return err
	}

	destPath := c.artifactPath(key)
	c.publicationMu.Lock()
	defer c.publicationMu.Unlock()
	if err := c.atomicWrite(destPath, data); err != nil {
		return fmt.Errorf("cudacache: write artifact: %w", err)
	}

	c.memMu.Lock()
	c.memArtifacts[key] = art
	c.memMu.Unlock()

	return nil
}

// GetArtifact retrieves an artifact by key. It checks in-memory cache, local disk CAS,
// and finally the remote source if configured.
// It verifies the SHA-256 digest and validates the artifact structure. If corruption
// is detected, the corrupted payload is moved to quarantine and ErrCorruptArtifact is returned.
func (c *Cache) GetArtifact(ctx context.Context, key string) (*cudakernel.Artifact, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateKey(key); err != nil {
		return nil, err
	}

	// 1. Fast memory check
	c.memMu.RLock()
	art, ok := c.memArtifacts[key]
	c.memMu.RUnlock()
	if ok {
		return cloneArtifact(art)
	}

	// Single flight deduplication for disk/remote load
	c.flightMu.Lock()
	if fc, inFlight := c.flight[key]; inFlight {
		c.flightMu.Unlock()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-fc.done:
		}
		if fc.err != nil {
			return nil, fc.err
		}
		if a, ok := fc.val.(*cudakernel.Artifact); ok {
			return cloneArtifact(a)
		}
		return nil, ErrNotFound
	}
	fc := &flightCall{done: make(chan struct{})}
	c.flight[key] = fc
	c.flightMu.Unlock()

	art, err := c.loadArtifact(ctx, key)
	fc.val = art
	fc.err = err
	close(fc.done)

	c.flightMu.Lock()
	delete(c.flight, key)
	c.flightMu.Unlock()

	if err != nil {
		return nil, err
	}
	return cloneArtifact(art)
}

func (c *Cache) loadArtifact(ctx context.Context, key string) (*cudakernel.Artifact, error) {
	destPath := c.artifactPath(key)

	// 2. Local disk check
	c.publicationMu.Lock()
	art, err := c.loadLocalArtifact(key, destPath)
	c.publicationMu.Unlock()
	if !errors.Is(err, ErrNotFound) {
		return art, err
	}

	// 3. Remote source lookup. Callbacks run outside the publication lock.
	if c.remote == nil {
		return nil, ErrNotFound
	}

	rc, size, err := c.remote.FetchArtifact(ctx, key)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if errors.Is(err, ErrRemoteNotFound) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("%w: %v", ErrRemoteUnavailable, err)
	}
	if rc == nil {
		return nil, fmt.Errorf("%w: nil artifact stream", ErrRemoteUnavailable)
	}
	defer rc.Close()

	if size > c.maxRemoteBytes {
		return nil, fmt.Errorf("%w: remote size %d exceeds limit %d", ErrDownloadLimitExceeded, size, c.maxRemoteBytes)
	}

	// Bounded reading to guard against malicious/unbounded streams
	limitReader := io.LimitReader(rc, c.maxRemoteBytes+1)
	remoteData, err := io.ReadAll(limitReader)
	if err != nil {
		return nil, fmt.Errorf("cudacache: read remote stream: %w", err)
	}
	if int64(len(remoteData)) > c.maxRemoteBytes {
		return nil, fmt.Errorf("%w: downloaded %d bytes, exceeds limit %d", ErrDownloadLimitExceeded, len(remoteData), c.maxRemoteBytes)
	}

	if err := c.verifyRemote(ctx, RemoteArtifact, key, remoteData); err != nil {
		return nil, err
	}
	art, err = c.validateAndDecodeArtifact(key, remoteData, "")
	if err != nil {
		return nil, err
	}

	c.publicationMu.Lock()
	defer c.publicationMu.Unlock()
	// A local publication completed while the callback ran takes precedence.
	c.memMu.RLock()
	current, ok := c.memArtifacts[key]
	c.memMu.RUnlock()
	if ok {
		return current, nil
	}

	// Non-fatal write error: still return the validated artifact.
	_ = c.atomicWrite(destPath, remoteData)
	c.memMu.Lock()
	c.memArtifacts[key] = art
	c.memMu.Unlock()
	return art, nil
}

// loadLocalArtifact runs under publicationMu, including validation and quarantine.
func (c *Cache) loadLocalArtifact(key, destPath string) (*cudakernel.Artifact, error) {
	c.memMu.RLock()
	art, ok := c.memArtifacts[key]
	c.memMu.RUnlock()
	if ok {
		return art, nil
	}
	data, err := readFileBounded(destPath, c.maxArtifactBytes)
	if err == nil {
		art, err := c.validateAndDecodeArtifact(key, data, destPath)
		if err != nil {
			return nil, err
		}
		c.memMu.Lock()
		c.memArtifacts[key] = art
		c.memMu.Unlock()
		return art, nil
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("cudacache: read artifact file: %w", err)
	}

	return nil, ErrNotFound
}

func (c *Cache) validateAndDecodeArtifact(key string, data []byte, localFilePath string) (*cudakernel.Artifact, error) {
	if int64(len(data)) > c.maxArtifactBytes {
		c.quarantine(key, data, localFilePath)
		return nil, fmt.Errorf("%w: artifact bytes %d exceed limit", ErrCorruptArtifact, len(data))
	}

	var art cudakernel.Artifact
	if err := json.Unmarshal(data, &art); err != nil {
		c.quarantine(key, data, localFilePath)
		return nil, fmt.Errorf("%w: unmarshal failed: %v", ErrCorruptArtifact, err)
	}

	// Verify SHA-256 of Raw bytes against art.Digest
	computed := sha256.Sum256(art.Raw)
	computedHex := hex.EncodeToString(computed[:])
	if art.Digest != computedHex {
		c.quarantine(key, data, localFilePath)
		return nil, fmt.Errorf("%w: digest mismatch (want %s, got %s)", ErrCorruptArtifact, art.Digest, computedHex)
	}

	if err := art.Validate(); err != nil {
		c.quarantine(key, data, localFilePath)
		return nil, fmt.Errorf("%w: validation failed: %v", ErrCorruptArtifact, err)
	}

	return &art, nil
}

func (c *Cache) quarantine(key string, data []byte, localFilePath string) {
	if localFilePath != "" {
		_ = os.Remove(localFilePath)
	}
	if len(data) == 0 {
		return
	}
	qPath := filepath.Join(c.quarantineDir, fmt.Sprintf("%s.%d.corrupt", key, time.Now().UnixNano()))
	_ = os.WriteFile(qPath, data, 0o644)
}

// PutReceipt writes a SearchReceipt under policyID and workloadKey.
func (c *Cache) PutReceipt(ctx context.Context, receipt *SearchReceipt) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if receipt == nil {
		return errors.New("cudacache: nil receipt")
	}
	if err := receipt.Validate(); err != nil {
		return fmt.Errorf("cudacache: invalid receipt: %w", err)
	}

	key := SearchReceiptKey(receipt.PolicyID, receipt.WorkloadKey)
	data, err := json.Marshal(receipt)
	if err != nil {
		return fmt.Errorf("cudacache: marshal receipt: %w", err)
	}
	if int64(len(data)) > c.maxArtifactBytes {
		return ErrDownloadLimitExceeded
	}
	var snapshot SearchReceipt
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return err
	}

	destPath := c.receiptPath(key)
	c.publicationMu.Lock()
	defer c.publicationMu.Unlock()
	if err := c.atomicWrite(destPath, data); err != nil {
		return fmt.Errorf("cudacache: write receipt: %w", err)
	}

	c.memMu.Lock()
	c.memReceipts[key] = &snapshot
	c.memMu.Unlock()

	return nil
}

// GetReceipt retrieves a SearchReceipt for a policy and workload.
func (c *Cache) GetReceipt(ctx context.Context, policyID SearchPolicyID, workloadKey string) (*SearchReceipt, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	key := SearchReceiptKey(policyID, workloadKey)

	// Memory check
	c.memMu.RLock()
	r, ok := c.memReceipts[key]
	c.memMu.RUnlock()
	if ok {
		return cloneReceipt(r)
	}

	// Disk check
	destPath := c.receiptPath(key)
	c.publicationMu.Lock()
	r, err := c.loadLocalReceipt(key, destPath)
	c.publicationMu.Unlock()
	if !errors.Is(err, ErrNotFound) {
		if err != nil {
			return nil, err
		}
		return cloneReceipt(r)
	}

	// Remote check. Callbacks run outside the publication lock.
	if c.remote != nil {
		rData, err := c.remote.FetchReceipt(ctx, key)
		if err == nil {
			if int64(len(rData)) > c.maxRemoteBytes || int64(len(rData)) > c.maxArtifactBytes {
				return nil, ErrDownloadLimitExceeded
			}
			if err := c.verifyRemote(ctx, RemoteReceipt, key, rData); err != nil {
				return nil, err
			}
			var receipt SearchReceipt
			if err := json.Unmarshal(rData, &receipt); err != nil {
				return nil, fmt.Errorf("%w: %v", ErrCorruptReceipt, err)
			}
			if err := receipt.Validate(); err != nil {
				return nil, fmt.Errorf("%w: %v", ErrCorruptReceipt, err)
			}
			if SearchReceiptKey(receipt.PolicyID, receipt.WorkloadKey) != key {
				return nil, fmt.Errorf("%w: identity mismatch", ErrCorruptReceipt)
			}

			c.publicationMu.Lock()
			defer c.publicationMu.Unlock()
			// Preserve any publication completed while the callback ran.
			c.memMu.RLock()
			current, ok := c.memReceipts[key]
			c.memMu.RUnlock()
			if ok {
				return cloneReceipt(current)
			}
			_ = c.atomicWrite(destPath, rData)
			c.memMu.Lock()
			c.memReceipts[key] = &receipt
			c.memMu.Unlock()
			return cloneReceipt(&receipt)
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if !errors.Is(err, ErrRemoteNotFound) {
			return nil, fmt.Errorf("cudacache: fetch receipt: %w", err)
		}
	}

	return nil, ErrNotFound
}

// loadLocalReceipt runs under publicationMu so it cannot cache an older disk read.
func (c *Cache) loadLocalReceipt(key, destPath string) (*SearchReceipt, error) {
	c.memMu.RLock()
	r, ok := c.memReceipts[key]
	c.memMu.RUnlock()
	if ok {
		return r, nil
	}
	data, err := readFileBounded(destPath, c.maxArtifactBytes)
	if err == nil {
		var receipt SearchReceipt
		if err := json.Unmarshal(data, &receipt); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrCorruptReceipt, err)
		}
		if err := receipt.Validate(); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrCorruptReceipt, err)
		}
		if SearchReceiptKey(receipt.PolicyID, receipt.WorkloadKey) != key {
			return nil, fmt.Errorf("%w: identity mismatch", ErrCorruptReceipt)
		}
		c.memMu.Lock()
		c.memReceipts[key] = &receipt
		c.memMu.Unlock()
		return &receipt, nil
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("cudacache: read receipt file: %w", err)
	}

	return nil, ErrNotFound
}

func validateKey(key string) error {
	if key == "" || len(key) > 200 || key == "." || key == ".." {
		return errors.New("cudacache: invalid cache key")
	}
	for _, r := range key {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == '.') {
			return errors.New("cudacache: invalid cache key")
		}
	}
	return nil
}

func readFileBounded(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, ErrDownloadLimitExceeded
	}
	return b, nil
}

func cloneArtifact(a *cudakernel.Artifact) (*cudakernel.Artifact, error) {
	b, err := json.Marshal(a)
	if err != nil {
		return nil, err
	}
	var copy cudakernel.Artifact
	err = json.Unmarshal(b, &copy)
	return &copy, err
}

func cloneReceipt(r *SearchReceipt) (*SearchReceipt, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	var copy SearchReceipt
	err = json.Unmarshal(b, &copy)
	return &copy, err
}

func (c *Cache) artifactPath(key string) string {
	return filepath.Join(c.artifactsDir, key+".artifact")
}

func (c *Cache) receiptPath(key string) string {
	return filepath.Join(c.receiptsDir, key+".receipt")
}

func (c *Cache) atomicWrite(targetPath string, data []byte) error {
	dir := filepath.Dir(targetPath)
	f, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	tmpName := f.Name()
	defer os.Remove(tmpName)

	if _, err := f.Write(data); err != nil {
		f.Close()
		return fmt.Errorf("write data: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("sync temp file: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close temp file: %w", err)
	}

	if err := os.Rename(tmpName, targetPath); err != nil {
		return fmt.Errorf("rename to target: %w", err)
	}
	return nil
}
