package cudacache

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// PutTrajectory atomically persists a receipt admitted by the live observer.
// The content key is the receipt checksum, independent of compiled artifacts.
func (c *Cache) PutTrajectory(ctx context.Context, r cudatrial.TrajectoryReceipt) error {
	if !r.HardwareObserved() {
		return ErrUnqualifiedOutcome
	}
	return c.putTrajectory(ctx, r)
}

// PutTrustedTrajectory persists a declaration whose producer provenance the
// caller independently trusts. It does not establish live hardware evidence.
func (c *Cache) PutTrustedTrajectory(ctx context.Context, r cudatrial.TrajectoryReceipt) error {
	return c.putTrajectory(ctx, r)
}

func (c *Cache) putTrajectory(ctx context.Context, r cudatrial.TrajectoryReceipt) error {
	if ctx == nil || c == nil {
		return fmt.Errorf("cudacache: nil trajectory context or cache")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.receiptsDir == "" || c.quarantineDir == "" || c.maxArtifactBytes <= 0 {
		return fmt.Errorf("cudacache: uninitialized trajectory cache")
	}
	c.trajectoryMu.Lock()
	defer c.trajectoryMu.Unlock()
	if err := r.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrUnqualifiedOutcome, err)
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if len(b) > 64<<10 || int64(len(b)) > c.maxArtifactBytes {
		return fmt.Errorf("cudacache: trajectory byte limit")
	}
	dir := filepath.Join(c.receiptsDir, "trajectories")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	return c.atomicWrite(filepath.Join(dir, r.SHA256+".json"), b)
}

// GetTrajectory reads a bounded receipt and checks key, identity and checksum.
// Corrupt local bytes are quarantined and never returned as search guidance.
func (c *Cache) GetTrajectory(ctx context.Context, key string) (cudatrial.TrajectoryReceipt, error) {
	var r cudatrial.TrajectoryReceipt
	if ctx == nil || c == nil {
		return r, fmt.Errorf("cudacache: nil trajectory context or cache")
	}
	if err := ctx.Err(); err != nil {
		return r, err
	}
	if c.receiptsDir == "" || c.quarantineDir == "" || c.maxArtifactBytes <= 0 {
		return r, fmt.Errorf("cudacache: uninitialized trajectory cache")
	}
	c.trajectoryMu.Lock()
	defer c.trajectoryMu.Unlock()
	if len(key) != 64 {
		return r, fmt.Errorf("cudacache: invalid trajectory key")
	}
	decoded, err := hex.DecodeString(key)
	if err != nil || len(decoded) != 32 || hex.EncodeToString(decoded) != key {
		return r, fmt.Errorf("cudacache: invalid trajectory key")
	}
	path := filepath.Join(c.receiptsDir, "trajectories", key+".json")
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return r, ErrNotFound
	}
	if err != nil {
		return r, err
	}
	if !info.Mode().IsRegular() {
		return r, fmt.Errorf("%w: trajectory file type", ErrCorruptReceipt)
	}
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return r, ErrNotFound
	}
	if err != nil {
		return r, err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil {
		return r, err
	}
	if !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		return r, fmt.Errorf("%w: trajectory file changed", ErrCorruptReceipt)
	}
	limit := int64(64 << 10)
	if c.maxArtifactBytes < limit {
		limit = c.maxArtifactBytes
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return r, err
	}
	bad := func(cause error) (cudatrial.TrajectoryReceipt, error) {
		q, err := os.CreateTemp(c.quarantineDir, key+"-*.trajectory-corrupt")
		if err != nil {
			return cudatrial.TrajectoryReceipt{}, fmt.Errorf("%w: %v; quarantine: %w", ErrCorruptReceipt, cause, err)
		}
		qpath := q.Name()
		if err := q.Close(); err != nil {
			os.Remove(qpath)
			return cudatrial.TrajectoryReceipt{}, fmt.Errorf("%w: %v; quarantine close: %w", ErrCorruptReceipt, cause, err)
		}
		// Rename retains the entire original, including over-limit bytes, without
		// reading them into memory or deleting the source before retention.
		if err := os.Rename(path, qpath); err != nil {
			os.Remove(qpath)
			return cudatrial.TrajectoryReceipt{}, fmt.Errorf("%w: %v; quarantine rename: %w", ErrCorruptReceipt, cause, err)
		}
		return cudatrial.TrajectoryReceipt{}, fmt.Errorf("%w: %v", ErrCorruptReceipt, cause)
	}
	if int64(len(b)) > limit {
		return bad(fmt.Errorf("trajectory byte limit"))
	}
	if err := memoryJSON(b); err != nil {
		return bad(err)
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&r); err != nil {
		return bad(err)
	}
	if d.Decode(new(any)) != io.EOF {
		return bad(fmt.Errorf("trailing trajectory JSON"))
	}
	if r.SHA256 != key {
		return bad(fmt.Errorf("trajectory key mismatch"))
	}
	if err := r.Validate(); err != nil {
		return bad(err)
	}
	return r, nil
}
