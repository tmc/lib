package isolation

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

// ManifestAttachment names the late-evidence binding manifest.
const ManifestAttachment = "gpu-isolation-binding-v1.json"

// MaxSources bounds retained evidence and successive binding operations.
const MaxSources = 16

// Source binds immutable receipt bytes to an attachment.
type Source struct {
	Attachment string `json:"attachment"`
	SHA256     string `json:"sha256"`
}

// Hop records the input capture identity before adding late evidence.
type Hop struct {
	CaptureSHA256          string `json:"input_capture_sha256"`
	ReceiptSHA256          string `json:"added_receipt_sha256"`
	PreviousManifestSHA256 string `json:"previous_manifest_sha256,omitempty"`
}

// Manifest preserves independent capture binding and every prior receipt.
type Manifest struct {
	Version  int      `json:"version"`
	Role     string   `json:"role"`
	Binding  Binding  `json:"binding"`
	Sources  []Source `json:"sources"`
	Hops     []Hop    `json:"hops"`
	Consumer string   `json:"consumer"`
}

// ReadManifest reads a bounded binding manifest and hashes its original bytes.
func ReadManifest(r io.Reader) (*Manifest, string, error) {
	b, err := io.ReadAll(io.LimitReader(r, MaxBytes+1))
	if err != nil {
		return nil, "", fmt.Errorf("read isolation binding manifest: %w", err)
	}
	if len(b) > MaxBytes {
		return nil, "", fmt.Errorf("isolation binding manifest exceeds size limit")
	}
	hash := sha256.Sum256(b)
	digest := hex.EncodeToString(hash[:])
	if err := uniqueJSON(b); err != nil {
		return nil, digest, err
	}
	var m Manifest
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&m); err != nil {
		return nil, digest, fmt.Errorf("decode isolation binding manifest: %w", err)
	}
	if m.Version != 1 || m.Role != "gpu-isolation-binding" || len(m.Sources) == 0 || len(m.Sources) > MaxSources || len(m.Hops) == 0 || len(m.Hops) > MaxSources || m.Consumer == "" {
		return nil, digest, fmt.Errorf("invalid isolation binding manifest identity or bounds")
	}
	seen := map[string]bool{}
	for _, s := range m.Sources {
		if s.Attachment == "" || filepath.Base(s.Attachment) != s.Attachment || strings.ContainsAny(s.Attachment, "/\\") || s.Attachment == "." || s.Attachment == ".." || s.Attachment == ManifestAttachment || seen[s.Attachment] || !validHash(s.SHA256) {
			return nil, digest, fmt.Errorf("invalid isolation receipt source")
		}
		seen[s.Attachment] = true
	}
	for _, h := range m.Hops {
		found := false
		for _, source := range m.Sources {
			if source.SHA256 == h.ReceiptSHA256 {
				found = true
			}
		}
		if !found {
			return nil, digest, fmt.Errorf("isolation binding hop receipt source missing")
		}
		if !validHash(h.CaptureSHA256) || !validHash(h.ReceiptSHA256) || (h.PreviousManifestSHA256 != "" && !validHash(h.PreviousManifestSHA256)) {
			return nil, digest, fmt.Errorf("invalid isolation binding hop hashes")
		}
	}
	return &m, digest, nil
}
func validHash(s string) bool {
	if len(s) != 64 || strings.ToLower(s) != s {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
