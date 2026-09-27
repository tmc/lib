package cudacache

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const (
	manifestVersion = 1
	manifestDomain  = "cudacache-manifest-v1\n"
)

// Manifest is the JSON record NVRTCAdapter publishes for each entry. It
// binds a compiler cache key to the identity it was computed from, the
// toolchain that produced the payload, and the payload's digest.
//
// ManifestSHA256 detects corruption and partial writes. It is not keyed and
// does not authenticate the manifest's author.
type Manifest struct {
	Version        int              `json:"version"`
	Key            string           `json:"key"`             // CompilerCacheKey(InputKey, Identity)
	InputKey       string           `json:"input_key"`       // caller's input key
	Identity       CompilerIdentity `json:"identity"`        // identity the key was computed from
	Recorded       ToolProvenance   `json:"recorded"`        // toolchain that produced the payload
	ArtifactKey    string           `json:"artifact_key"`    // Key + "-" + PayloadSHA256
	PayloadSHA256  string           `json:"payload_sha256"`  // equals the artifact's Digest
	Nonce          string           `json:"nonce"`           // random per write; tells a rebuild from the manifest it replaced
	ManifestSHA256 string           `json:"manifest_sha256"` // over every other field
}

// digest returns the SHA-256 of m's JSON encoding with ManifestSHA256 empty.
func (m Manifest) digest() (string, error) {
	m.ManifestSHA256 = ""
	data, err := json.Marshal(m)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	h.Write([]byte(manifestDomain))
	h.Write(data)
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (c *Cache) manifestPath(key string) string {
	return filepath.Join(c.rootDir, "manifests", key+".manifest")
}

func (c *Cache) markerPath(key string) string {
	return filepath.Join(c.artifactsDir, key+".toolchain-mismatch-marker")
}

// writeManifest gives m a fresh nonce, seals it and publishes it atomically.
func (c *Cache) writeManifest(m Manifest) error {
	m.Nonce = rand.Text()
	sum, err := m.digest()
	if err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	m.ManifestSHA256 = sum
	data, err := json.Marshal(m)
	if err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	path := c.manifestPath(m.Key)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	if err := c.atomicWrite(path, data); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
}

// checkManifest decodes data and verifies it is a sealed manifest for key.
// The returned error does not wrap ErrCorruptArtifact; the caller adds it.
func checkManifest(key string, data []byte) (Manifest, error) {
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return Manifest{}, fmt.Errorf("decode manifest: %v", err)
	}
	if m.Version != manifestVersion {
		return m, fmt.Errorf("manifest version %d, want %d", m.Version, manifestVersion)
	}
	sum, err := m.digest()
	if err != nil {
		return m, fmt.Errorf("digest manifest: %v", err)
	}
	if m.ManifestSHA256 != sum {
		return m, fmt.Errorf("manifest sha256 %s, computed %s", m.ManifestSHA256, sum)
	}
	if m.Key != key {
		return m, fmt.Errorf("manifest key %s, want %s", m.Key, key)
	}
	if k := CompilerCacheKey(m.InputKey, m.Identity); k != key {
		return m, fmt.Errorf("manifest identity has key %s, want %s", k, key)
	}
	if m.ArtifactKey != key+"-"+m.PayloadSHA256 {
		return m, fmt.Errorf("manifest artifact key %s does not name payload %s", m.ArtifactKey, m.PayloadSHA256)
	}
	return m, nil
}

// quarantineEntry moves the manifest for key, and the artifact it names if
// artifactKey is valid, to quarantine and drops the artifact from memory.
func (c *Cache) quarantineEntry(key string, manifest []byte, artifactKey string) {
	c.quarantine(key, manifest, c.manifestPath(key))
	if validateKey(artifactKey) != nil {
		return
	}
	c.evictMemory(artifactKey)
	path := c.artifactPath(artifactKey)
	data, _ := readFileBounded(path, c.maxArtifactBytes)
	c.quarantine(artifactKey, data, path)
}

// evictEntry removes the manifest for key and the artifact it names.
func (c *Cache) evictEntry(key, artifactKey string) error {
	c.evictMemory(artifactKey)
	if err := os.Remove(c.manifestPath(key)); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("evict manifest: %w", err)
	}
	if err := os.Remove(c.artifactPath(artifactKey)); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("evict artifact: %w", err)
	}
	return nil
}

func (c *Cache) evictMemory(key string) {
	c.memMu.Lock()
	delete(c.memArtifacts, key)
	c.memMu.Unlock()
}
