package cudasotaeval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
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

		claimJSON, _ := report.Claim.JSON()
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

// VerifyManifest checks that every file declared in manifest.json exists in dir
// and matches its declared SHA256 digest.
func VerifyManifest(dir string) (*ArtifactManifest, error) {
	manPath := filepath.Join(dir, "manifest.json")
	data, err := os.ReadFile(manPath)
	if err != nil {
		return nil, fmt.Errorf("verify manifest: read: %w", err)
	}

	var man ArtifactManifest
	if err := json.Unmarshal(data, &man); err != nil {
		return nil, fmt.Errorf("verify manifest: unmarshal: %w", err)
	}

	for relPath, expectedHash := range man.Files {
		fullPath := filepath.Join(dir, relPath)
		actualHash, err := hashFile(fullPath)
		if err != nil {
			return nil, fmt.Errorf("verify file %s: %w", relPath, err)
		}
		if actualHash != expectedHash {
			return nil, fmt.Errorf("digest mismatch on %s: expected %s, got %s", relPath, expectedHash, actualHash)
		}
	}

	return &man, nil
}

func hashBytes(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
