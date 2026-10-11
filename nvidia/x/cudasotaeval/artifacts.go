package cudasotaeval

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ArtifactManifest records all generated artifacts and their cryptographically verified digests.
type ArtifactManifest struct {
	Version        int               `json:"version"`
	Timestamp      string            `json:"timestamp"`
	Files          map[string]string `json:"files"`
	ManifestSHA256 string            `json:"manifest_sha256"`
}

// WriteReportArtifacts persists the evaluation report, claim, protocol, and receipts to outDir,
// producing an ArtifactManifest with SHA256 digests.
func WriteReportArtifacts(outDir string, report *EvaluationReport) (*ArtifactManifest, error) {
	if report == nil {
		return nil, fmt.Errorf("artifacts: nil report")
	}
	if outDir == "" {
		outDir = "."
	}
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return nil, fmt.Errorf("artifacts: mkdir: %w", err)
	}

	files := make(map[string]string)

	// 1. evaluation_report.json
	repBytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("artifacts: marshal report: %w", err)
	}
	repPath := filepath.Join(outDir, "evaluation_report.json")
	if err := os.WriteFile(repPath, repBytes, 0644); err != nil {
		return nil, fmt.Errorf("artifacts: write report: %w", err)
	}
	files["evaluation_report.json"] = hashBytes(repBytes)

	// 2. protocol.json
	proBytes, err := json.MarshalIndent(report.Protocol, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("artifacts: marshal protocol: %w", err)
	}
	proPath := filepath.Join(outDir, "protocol.json")
	if err := os.WriteFile(proPath, proBytes, 0644); err != nil {
		return nil, fmt.Errorf("artifacts: write protocol: %w", err)
	}
	files["protocol.json"] = hashBytes(proBytes)

	// 3. claim.txt and claim.json
	if report.Claim != nil {
		claimTxt := report.Claim.Format()
		txtPath := filepath.Join(outDir, "claim.txt")
		if err := os.WriteFile(txtPath, []byte(claimTxt), 0644); err != nil {
			return nil, fmt.Errorf("artifacts: write claim.txt: %w", err)
		}
		files["claim.txt"] = hashBytes([]byte(claimTxt))

		claimJSON, err := report.Claim.JSON()
		if err != nil {
			return nil, fmt.Errorf("artifacts: marshal claim: %w", err)
		}
		jsonPath := filepath.Join(outDir, "claim.json")
		if err := os.WriteFile(jsonPath, claimJSON, 0644); err != nil {
			return nil, fmt.Errorf("artifacts: write claim.json: %w", err)
		}
		files["claim.json"] = hashBytes(claimJSON)
	}

	// 4. Compute overall manifest hash
	keys := make([]string, 0, len(files))
	for k := range files {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	mH := sha256.New()
	for _, k := range keys {
		mH.Write([]byte(k + ":" + files[k] + "\n"))
	}
	manifestHash := hex.EncodeToString(mH.Sum(nil))

	manifest := &ArtifactManifest{
		Version:        1,
		Timestamp:      report.Timestamp.Format("2006-01-02T15:04:05Z07:00"),
		Files:          files,
		ManifestSHA256: manifestHash,
	}

	manBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("artifacts: marshal manifest: %w", err)
	}
	manPath := filepath.Join(outDir, "manifest.json")
	if err := os.WriteFile(manPath, manBytes, 0644); err != nil {
		return nil, fmt.Errorf("artifacts: write manifest.json: %w", err)
	}

	return manifest, nil
}

const (
	maxManifestBytes      = 1 << 20
	maxArtifactBytes      = 64 << 20
	maxArtifactTotalBytes = 256 << 20
	maxArtifactFiles      = 1024
)

// VerifyManifest verifies a version 1 manifest and its declared regular files.
// It refuses symlinks, non-local paths, more than 1024 files, manifests over
// 1 MiB, files over 64 MiB, and total artifact contents over 256 MiB.
func VerifyManifest(dir string) (*ArtifactManifest, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, fmt.Errorf("verify manifest: open directory: %w", err)
	}
	defer root.Close()
	f, err := openArtifact(root, "manifest.json")
	if err != nil {
		return nil, fmt.Errorf("verify manifest: open: %w", err)
	}
	data, err := io.ReadAll(io.LimitReader(f, maxManifestBytes+1))
	f.Close()
	if err != nil {
		return nil, fmt.Errorf("verify manifest: read: %w", err)
	}
	if len(data) > maxManifestBytes {
		return nil, fmt.Errorf("verify manifest: manifest exceeds size limit")
	}
	if err := checkArtifactJSON(json.NewDecoder(bytes.NewReader(data))); err != nil {
		return nil, fmt.Errorf("verify manifest: decode: %w", err)
	}
	var man ArtifactManifest
	if err := json.Unmarshal(data, &man); err != nil {
		return nil, fmt.Errorf("verify manifest: decode: %w", err)
	}
	if man.Version != 1 {
		return nil, fmt.Errorf("verify manifest: unsupported version %d", man.Version)
	}
	if len(man.Files) == 0 || len(man.Files) > maxArtifactFiles {
		return nil, fmt.Errorf("verify manifest: invalid file count")
	}
	keys := make([]string, 0, len(man.Files))
	for name, digest := range man.Files {
		if !filepath.IsLocal(name) || filepath.Clean(name) != name || strings.ContainsAny(name, "\\:\n\r") || name == "manifest.json" {
			return nil, fmt.Errorf("verify manifest: invalid artifact path %q", name)
		}
		decoded, err := hex.DecodeString(digest)
		if err != nil || len(decoded) != sha256.Size || strings.ToLower(digest) != digest {
			return nil, fmt.Errorf("verify manifest: invalid digest for %q", name)
		}
		keys = append(keys, name)
	}
	sort.Strings(keys)
	h := sha256.New()
	for _, name := range keys {
		fmt.Fprintf(h, "%s:%s\n", name, man.Files[name])
	}
	if hex.EncodeToString(h.Sum(nil)) != man.ManifestSHA256 {
		return nil, fmt.Errorf("verify manifest: manifest digest mismatch")
	}
	var total int64
	for _, name := range keys {
		f, err := openArtifact(root, name)
		if err != nil {
			return nil, fmt.Errorf("verify artifact %q: %w", name, err)
		}
		info, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("verify artifact %q: %w", name, err)
		}
		if info.Size() > maxArtifactBytes || info.Size() > maxArtifactTotalBytes-total {
			f.Close()
			return nil, fmt.Errorf("verify artifact %q: contents exceed size limit", name)
		}
		h := sha256.New()
		limit := int64(maxArtifactBytes)
		if remaining := int64(maxArtifactTotalBytes) - total; remaining < limit {
			limit = remaining
		}
		n, err := io.Copy(h, io.LimitReader(f, limit+1))
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("verify artifact %q: %w", name, err)
		}
		total += n
		if n > maxArtifactBytes || total > maxArtifactTotalBytes {
			return nil, fmt.Errorf("verify artifact %q: contents exceed size limit", name)
		}
		if hex.EncodeToString(h.Sum(nil)) != man.Files[name] {
			return nil, fmt.Errorf("verify artifact %q: digest mismatch", name)
		}
	}
	return &man, nil
}

// Root bounds path resolution even if directory entries change during verification.
func openArtifact(root *os.Root, name string) (*os.File, error) {
	parts := strings.Split(name, string(filepath.Separator))
	for i := range parts {
		info, err := root.Lstat(filepath.Join(parts[:i+1]...))
		if err != nil {
			return nil, err
		}
		if i == len(parts)-1 && !info.Mode().IsRegular() {
			return nil, fmt.Errorf("artifact is not a regular file")
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("symlink is not an artifact")
		}
	}
	f, err := root.Open(name)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	if !info.Mode().IsRegular() {
		f.Close()
		return nil, fmt.Errorf("artifact is not a regular file")
	}
	return f, nil
}

func hashBytes(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// checkArtifactJSON rejects duplicate keys before decoding into maps and structs.
func checkArtifactJSON(d *json.Decoder) error {
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok {
				return fmt.Errorf("invalid object key")
			}
			if seen[name] {
				return fmt.Errorf("duplicate object key %q", name)
			}
			seen[name] = true
			if err := checkArtifactJSON(d); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := checkArtifactJSON(d); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("invalid JSON delimiter")
	}
	_, err = d.Token()
	return err
}
