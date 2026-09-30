package tracecontract

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path"
	"sort"
	"strings"
)

// Member binds the bytes of one archive member. Kind describes the member;
// CaptureSHA256 names its source capture when applicable. No member hashes itself.
type Member struct {
	Path          string `json:"path"`
	Kind          string `json:"kind"`
	Bytes         int64  `json:"bytes"`
	SHA256        string `json:"sha256"`
	CaptureSHA256 string `json:"capture_sha256,omitempty"`
}

// Bindings hashes exact comparison artifacts, including the official manifest
// and alignment receipt. The index is stored as .cudaprof-bindings.json, outside
// the Perfetto manifest's trace files. Internal hash consistency is not an
// authenticity signature; consumers also hash the exact archive bytes parsed.
// The zero value is not an index; Version and Schema must identify v1.
type Bindings struct {
	Version int      `json:"version"`
	Schema  string   `json:"schema"`
	Members []Member `json:"members"`
}

// Digest computes one archive member's hash. Name must be a safe relative path.
func Digest(name, kind, captureSHA256 string, data []byte) (Member, error) {
	member := Member{Path: name, Kind: kind, Bytes: int64(len(data)), CaptureSHA256: captureSHA256}
	hash := sha256.Sum256(data)
	member.SHA256 = hex.EncodeToString(hash[:])
	if err := member.validate(); err != nil {
		return Member{}, err
	}
	return member, nil
}

func (m Member) validate() error {
	if m.Path == "" || path.IsAbs(m.Path) || path.Clean(m.Path) != m.Path || m.Path == ".." || strings.HasPrefix(m.Path, "../") || strings.Contains(m.Path, "\\") || m.Path == ".cudaprof-bindings.json" {
		return fmt.Errorf("trace binding: unsafe or self-referential member path")
	}
	if m.Kind == "" || m.Bytes < 0 || !validHash(m.SHA256) || (m.CaptureSHA256 != "" && !validHash(m.CaptureSHA256)) {
		return fmt.Errorf("trace binding: invalid member identity")
	}
	if m.Kind == "trace" && m.CaptureSHA256 == "" {
		return fmt.Errorf("trace binding: trace lacks source capture identity")
	}
	return nil
}

// JSON returns a canonical member index sorted by path. Duplicate paths,
// unsupported versions and absent manifest/alignment members are refused.
func (b Bindings) JSON() ([]byte, error) {
	if len(b.Members) > 4096 {
		return nil, fmt.Errorf("trace binding: member limit exceeded")
	}
	if b.Version != 1 || b.Schema != "cudaprof.bindings.v1" {
		return nil, fmt.Errorf("trace binding: unsupported schema or version")
	}
	b.Members = append([]Member(nil), b.Members...)
	sort.Slice(b.Members, func(i, j int) bool { return b.Members[i].Path < b.Members[j].Path })
	seen := map[string]bool{}
	for _, m := range b.Members {
		if err := m.validate(); err != nil {
			return nil, err
		}
		if seen[m.Path] {
			return nil, fmt.Errorf("trace binding: duplicate member %q", m.Path)
		}
		seen[m.Path] = true
	}
	if !seen["perfetto_manifest.json"] || !seen[".alignment.json"] {
		return nil, fmt.Errorf("trace binding: missing manifest or alignment receipt")
	}
	return json.Marshal(b)
}

// Verify checks that every artifact member is bound to its exact bytes, with
// no omitted or added members. The index itself is excluded from self-hashing.
func (b Bindings) Verify(members map[string][]byte) error {
	if _, err := b.JSON(); err != nil {
		return err
	}
	bound := map[string]bool{}
	for _, m := range b.Members {
		data, ok := members[m.Path]
		if !ok || int64(len(data)) != m.Bytes {
			return fmt.Errorf("trace binding: missing member or size mismatch for %q", m.Path)
		}
		hash := sha256.Sum256(data)
		if hex.EncodeToString(hash[:]) != m.SHA256 {
			return fmt.Errorf("trace binding: hash mismatch for %q", m.Path)
		}
		bound[m.Path] = true
	}
	for name := range members {
		if name != ".cudaprof-bindings.json" && !bound[name] {
			return fmt.Errorf("trace binding: unbound member %q", name)
		}
	}
	return nil
}
