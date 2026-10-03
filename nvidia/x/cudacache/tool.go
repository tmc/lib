package cudacache

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
)

// ErrToolIdentityChanged reports that a tool file observed earlier is no
// longer the same file, or no longer has the same contents.
var ErrToolIdentityChanged = errors.New("cudacache: tool identity changed")

// ToolFileIdentity is what stat reports about a tool file. It identifies a
// file, not its contents: two identities that differ may name identical
// bytes, and only a digest settles that.
//
// Device, Inode and ChangeTimeNanos are set only when Unix is true.
type ToolFileIdentity struct {
	Path            string `json:"path"`
	Size            int64  `json:"size"`
	ModTimeNanos    int64  `json:"mtime_ns"`
	Device          uint64 `json:"dev,omitempty"`
	Inode           uint64 `json:"ino,omitempty"`
	ChangeTimeNanos int64  `json:"ctime_ns,omitempty"`
	Unix            bool   `json:"unix"`
}

// StatTool returns the identity of the regular file at path.
func StatTool(path string) (ToolFileIdentity, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return ToolFileIdentity{}, fmt.Errorf("stat tool: %w", err)
	}
	return toolFileIdentity(path, fi)
}

func toolFileIdentity(path string, fi os.FileInfo) (ToolFileIdentity, error) {
	if !fi.Mode().IsRegular() {
		return ToolFileIdentity{}, fmt.Errorf("stat tool %s: not a regular file", path)
	}
	id := ToolFileIdentity{
		Path:         path,
		Size:         fi.Size(),
		ModTimeNanos: fi.ModTime().UnixNano(),
	}
	id.Device, id.Inode, id.ChangeTimeNanos, id.Unix = unixFileIdentity(fi)
	return id, nil
}

// ToolDigest binds the SHA-256 of a tool file to the identity of the
// descriptor that was hashed.
type ToolDigest struct {
	Name   string           `json:"name"`   // "nvrtc", "driver", or a caller's name
	SHA256 string           `json:"sha256"` // lowercase hex
	File   ToolFileIdentity `json:"file"`
}

// DigestTool hashes the file at path through one open descriptor. It stats
// the descriptor before and after reading, and the path after reading, and
// reports ErrToolIdentityChanged if any of the three disagree: the file was
// written to, or replaced, while it was being hashed.
func DigestTool(name, path string) (ToolDigest, error) {
	f, err := os.Open(path)
	if err != nil {
		return ToolDigest{}, fmt.Errorf("digest tool %s: %w", name, err)
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return ToolDigest{}, fmt.Errorf("digest tool %s: %w", name, err)
	}
	before, err := toolFileIdentity(path, fi)
	if err != nil {
		return ToolDigest{}, fmt.Errorf("digest tool %s: %w", name, err)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ToolDigest{}, fmt.Errorf("digest tool %s: %w", name, err)
	}
	fi, err = f.Stat()
	if err != nil {
		return ToolDigest{}, fmt.Errorf("digest tool %s: %w", name, err)
	}
	after, err := toolFileIdentity(path, fi)
	if err != nil {
		return ToolDigest{}, fmt.Errorf("digest tool %s: %w", name, err)
	}
	named, err := StatTool(path)
	if err != nil {
		return ToolDigest{}, fmt.Errorf("digest tool %s: %w: %w", name, ErrToolIdentityChanged, err)
	}
	if after != before || named != before {
		return ToolDigest{}, fmt.Errorf("digest tool %s: %s changed while it was hashed: %w", name, path, ErrToolIdentityChanged)
	}
	d := ToolDigest{Name: name, SHA256: hex.EncodeToString(h.Sum(nil)), File: before}
	toolMemoMu.Lock()
	toolMemo[name+"\x00"+path] = d
	toolMemoMu.Unlock()
	return d, nil
}

// VerifyTool reports whether d.File.Path still holds the contents d recorded.
// A missing file, a different digest, or a change during hashing reports
// ErrToolIdentityChanged.
//
// When d.File.Unix is true, the current identity equals d.File, and this
// process previously hashed that identity to d.SHA256, the recorded digest
// is reused without reading the file. A caller's stat fields alone cannot
// establish its claimed digest. The reuse assumes a write
// changes the modification or change time; on a filesystem whose timestamps
// are too coarse to record a same-size write, such a write goes undetected.
// Without Unix identity the file is always rehashed.
func VerifyTool(d ToolDigest) error {
	if d.File.Path == "" || d.SHA256 == "" {
		return fmt.Errorf("verify tool %s: no recorded path or digest", d.Name)
	}
	cur, err := StatTool(d.File.Path)
	if err != nil {
		return fmt.Errorf("verify tool %s: %w: %w", d.Name, ErrToolIdentityChanged, err)
	}
	if d.File.Unix && cur == d.File {
		toolMemoMu.Lock()
		known, ok := toolMemo[d.Name+"\x00"+d.File.Path]
		toolMemoMu.Unlock()
		if ok && known.File == d.File && known.SHA256 == d.SHA256 {
			return nil
		}
	}
	now, err := DigestTool(d.Name, d.File.Path)
	if err != nil {
		if errors.Is(err, ErrToolIdentityChanged) {
			return fmt.Errorf("verify tool: %w", err)
		}
		return fmt.Errorf("verify tool %s: %w: %w", d.Name, ErrToolIdentityChanged, err)
	}
	if now.SHA256 != d.SHA256 {
		return fmt.Errorf("verify tool %s: %s has sha256 %s, recorded %s: %w", d.Name, d.File.Path, now.SHA256, d.SHA256, ErrToolIdentityChanged)
	}
	return nil
}

// WithRevalidatedTools runs op between two VerifyTool passes over tools.
// If the first pass fails, op does not run. The second pass runs even when
// op fails, and if it fails its error replaces op's result: an output
// produced while a tool changed is not attributable to either version.
func WithRevalidatedTools[T any](tools []ToolDigest, op func() (T, error)) (T, error) {
	var zero T
	for _, d := range tools {
		if err := VerifyTool(d); err != nil {
			return zero, err
		}
	}
	v, opErr := op()
	for _, d := range tools {
		if err := VerifyTool(d); err != nil {
			return zero, err
		}
	}
	return v, opErr
}
