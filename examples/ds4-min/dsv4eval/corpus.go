package dsv4eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Generate into an explicitly selected external directory, never this tree.
//go:generate go run ./internal/gensmoke

// SmokeCorpusManifest identifies an external, source-generated question corpus.
type SmokeCorpusManifest struct {
	OracleRevision   string            `json:"oracle_revision"`
	CorpusSHA256     string            `json:"corpus_sha256"`
	SourceSHA256     map[string]string `json:"source_sha256"`
	LicenseSHA256    string            `json:"license_sha256"`
	ExtractorSHA256  string            `json:"extractor_sha256"`
	ExtractorVersion int               `json:"extractor_version"`
	BinarySHA256     string            `json:"extraction_binary_sha256"`
	BuildCommand     []string          `json:"build_command"`
}

const SmokeOracleRevision = "11e8cad4473716a41baf4d62e92ed9764a00fce6"
const SmokeCorpusSHA256 = "5689dd9cf208fd67b8fef54b8400c6226c0c5a457965e45c3f90ea922253287e"
const SmokeLicenseSHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
const SmokeExtractorSHA256 = "e155034b2c827b3a4c18d496a1a3fe9986c2d8d8f7166579a7002a804e751cfe"

// SmokeSourceHashes returns the pinned oracle sources used by the extractor.
func SmokeSourceHashes() map[string]string {
	return map[string]string{
		"ds4_eval.c":       "4092ce91692e99b6cbb846401f075969362de5d2ea62662513979e18480f9a60",
		"ds4_eval_cases.c": "9e61d65d5c4db4fd312955b9ce4aed487b0487748b2cb8f91cf6b80b9b5cc764",
		"ds4_eval_cases.h": "39bc494d1a95ec095a16a0ed8c52326e6cb67cb11c20613aeccfc2108fa545a2",
	}
}

// LoadSmokeCorpus admits only the exact external twelve-case corpus generated
// from the pinned oracle. Questions and third-party licenses are not embedded.
func LoadSmokeCorpus(dir string) ([]Case, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("load smoke manifest: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var manifest SmokeCorpusManifest
	if err := decoder.Decode(&manifest); err != nil {
		return nil, err
	}
	var extra any
	if decoder.Decode(&extra) != io.EOF {
		return nil, fmt.Errorf("load smoke manifest: trailing data")
	}
	if manifest.OracleRevision != SmokeOracleRevision || manifest.CorpusSHA256 != SmokeCorpusSHA256 || manifest.LicenseSHA256 != SmokeLicenseSHA256 {
		return nil, fmt.Errorf("load smoke manifest: unqualified oracle or corpus")
	}
	if manifest.ExtractorVersion != 1 || manifest.ExtractorSHA256 != SmokeExtractorSHA256 || !validSHA(manifest.BinarySHA256) || len(manifest.BuildCommand) < 2 {
		return nil, fmt.Errorf("load smoke manifest: missing qualified extractor receipt")
	}
	expected := SmokeSourceHashes()
	if len(manifest.SourceSHA256) != len(expected) {
		return nil, fmt.Errorf("load smoke manifest: wrong source inventory")
	}
	for name, hash := range expected {
		if manifest.SourceSHA256[name] != hash {
			return nil, fmt.Errorf("load smoke manifest: source %s is unqualified", name)
		}
	}
	for _, item := range []struct{ name, hash string }{{"hard-smoke.json", SmokeCorpusSHA256}, {"Apache-2.0.txt", SmokeLicenseSHA256}} {
		value, err := os.ReadFile(filepath.Join(dir, item.name))
		if err != nil {
			return nil, err
		}
		if fmt.Sprintf("%x", sha256.Sum256(value)) != item.hash {
			return nil, fmt.Errorf("load smoke corpus: %s hash mismatch", item.name)
		}
		if item.name == "hard-smoke.json" {
			data = value
		}
	}
	var cases []Case
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, err
	}
	if err := validateRuntimeCases(cases, RunOptions{Build: SmokeOracleRevision, ModelSHA256: SmokeCorpusSHA256, Context: 32768, Soft: 1024, Hard: 512, Rank: 3}); err != nil {
		return nil, err
	}
	return cases, nil
}
