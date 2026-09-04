package inferbench

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// PathManifest is a reconstructible description of a directory identity.
type PathManifest struct {
	SHA256 string         `json:"sha256"`
	Files  []FileArtifact `json:"files"`
}

// FileArtifact identifies one regular file by relative name, mode, size, and content.
type FileArtifact struct {
	Name   string `json:"name"`
	Mode   string `json:"mode"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// ManifestPath records every regular file used by HashPath. Symlinks must
// resolve to regular files and are described by their resolved contents.
func ManifestPath(root string) (PathManifest, error) {
	rootHash, err := HashPath(root)
	if err != nil {
		return PathManifest{}, err
	}
	var files []FileArtifact
	err = filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		info, err := os.Stat(path)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("manifest path: %s is not a regular file", path)
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		sum, err := HashFile(path)
		if err != nil {
			return err
		}
		files = append(files, FileArtifact{Name: filepath.ToSlash(rel), Mode: strings.TrimSpace(info.Mode().String()), Size: info.Size(), SHA256: sum})
		return nil
	})
	if err != nil {
		return PathManifest{}, fmt.Errorf("manifest path: %w", err)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Name < files[j].Name })
	return PathManifest{SHA256: rootHash, Files: files}, nil
}

// VerifyPathManifest proves that root has exactly the files in manifest and
// reconstructs the directory hash without trusting HashPath's result alone.
func VerifyPathManifest(root string, manifest PathManifest) error {
	got, err := ManifestPath(root)
	if err != nil {
		return err
	}
	if got.SHA256 != manifest.SHA256 || len(got.Files) != len(manifest.Files) {
		return errors.New("path manifest differs")
	}
	for i := range got.Files {
		if got.Files[i] != manifest.Files[i] {
			return fmt.Errorf("path manifest file %d differs", i)
		}
	}
	return nil
}

// BuildReceipt binds an executable to the exact source, dependency, toolchain,
// and command inputs needed to reproduce it.
type BuildReceipt struct {
	BinarySHA256       string   `json:"binary_sha256"`
	SourceSHA256       string   `json:"source_sha256"`
	DependenciesSHA256 string   `json:"dependencies_sha256"`
	Toolchain          string   `json:"toolchain"`
	Command            []string `json:"command"`
}

// Validate checks that a build receipt contains independently comparable inputs.
func (r BuildReceipt) Validate() error {
	for _, sum := range []string{r.BinarySHA256, r.SourceSHA256, r.DependenciesSHA256} {
		if !validSHA256(sum) {
			return errors.New("build receipt has invalid hash")
		}
	}
	if r.Toolchain == "" || len(r.Command) == 0 {
		return errors.New("build receipt is incomplete")
	}
	return nil
}

// MutationProbe retains the inputs that prove a semantic comparator can fail.
type MutationProbe struct {
	ComparatorSHA256 string `json:"comparator_sha256"`
	InputSHA256      string `json:"input_sha256"`
	ExpectedSHA256   string `json:"expected_sha256"`
}

func (p MutationProbe) validate() error {
	for _, sum := range []string{p.ComparatorSHA256, p.InputSHA256, p.ExpectedSHA256} {
		if !validSHA256(sum) {
			return errors.New("mutation probe is incomplete")
		}
	}
	return nil
}

// SentinelRecord binds a stability measurement to its immediately preceding
// raw interval and to every prior sentinel in an append-only log.
type SentinelRecord struct {
	Sequence          int           `json:"sequence"`
	AfterIndex        int           `json:"after_index"`
	CapturedAt        time.Time     `json:"captured_at"`
	DeviceID          string        `json:"device_id"`
	PredecessorSHA256 string        `json:"predecessor_sha256"`
	PreviousSHA256    string        `json:"previous_sha256,omitempty"`
	Duration          time.Duration `json:"duration_ns"`
	SHA256            string        `json:"sha256"`
}

func hashSentinel(record SentinelRecord) (string, error) {
	record.SHA256 = ""
	return HashJSON(record)
}

// AppendSentinel appends and synchronizes one chained record. Callers invoke it
// from the raw-interval writer before allowing the next timed interval.
func AppendSentinel(path string, record SentinelRecord) error {
	record.CapturedAt = record.CapturedAt.Round(0)
	if record.Sequence < 0 || record.AfterIndex < 0 || record.CapturedAt.IsZero() || record.DeviceID == "" || !validSHA256(record.PredecessorSHA256) || record.Duration <= 0 {
		return errors.New("sentinel record is incomplete")
	}
	if record.Sequence == 0 && record.PreviousSHA256 != "" || record.Sequence > 0 && !validSHA256(record.PreviousSHA256) {
		return errors.New("sentinel chain is incomplete")
	}
	var err error
	record.SHA256, err = hashSentinel(record)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("append sentinel: %w", err)
	}
	b, err := json.Marshal(record)
	if err == nil {
		b = append(b, '\n')
		_, err = f.Write(b)
	}
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

// ReadSentinelChain verifies an append-only sentinel log and its raw predecessors.
func ReadSentinelChain(path, rawDir, deviceID string, every, intervals int) ([]SentinelRecord, error) {
	if every <= 0 || intervals <= 0 {
		return nil, errors.New("invalid sentinel cadence")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var records []SentinelRecord
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var record SentinelRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return nil, fmt.Errorf("sentinel %d: %w", len(records), err)
		}
		wantIndex := (len(records)+1)*every - 1
		if wantIndex >= intervals {
			wantIndex = intervals - 1
		}
		previous := ""
		if len(records) != 0 {
			previous = records[len(records)-1].SHA256
		}
		raw, err := HashFile(filepath.Join(rawDir, fmt.Sprintf("%06d.json", wantIndex)))
		if err != nil {
			return nil, err
		}
		var predecessor IntervalReceipt
		if err := readJSON(filepath.Join(rawDir, fmt.Sprintf("%06d.json", wantIndex)), &predecessor); err != nil {
			return nil, fmt.Errorf("sentinel %d predecessor: %w", len(records), err)
		}
		wantHash, err := hashSentinel(record)
		if err != nil || record.Sequence != len(records) || record.AfterIndex != wantIndex || record.DeviceID != deviceID || record.PreviousSHA256 != previous || record.PredecessorSHA256 != raw || record.SHA256 != wantHash || record.CapturedAt.IsZero() || record.Duration <= 0 || record.CapturedAt.Before(predecessor.Sample.EndedAt) {
			return nil, fmt.Errorf("sentinel %d differs", len(records))
		}
		if wantIndex+1 < intervals {
			var successor IntervalReceipt
			if err := readJSON(filepath.Join(rawDir, fmt.Sprintf("%06d.json", wantIndex+1)), &successor); err != nil {
				return nil, fmt.Errorf("sentinel %d successor: %w", len(records), err)
			}
			if record.CapturedAt.After(successor.Sample.StartedAt) {
				return nil, fmt.Errorf("sentinel %d is outside interval bracket", len(records))
			}
		}
		records = append(records, record)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	want := (intervals + every - 1) / every
	if len(records) != want || records[len(records)-1].AfterIndex != intervals-1 {
		return nil, errors.New("sentinel chain does not cover campaign")
	}
	return records, nil
}

// SealManifest binds the independently reconstructible provenance and liveness inputs.
type SealManifest struct {
	Build          BuildReceipt  `json:"build"`
	Model          PathManifest  `json:"model"`
	WeightsSHA256  string        `json:"weights_sha256"`
	Mutation       MutationProbe `json:"mutation"`
	SentinelSHA256 string        `json:"sentinel_sha256"`
}

// Validate checks the closed-set identity contract.
func (m SealManifest) Validate() error {
	if err := m.Build.Validate(); err != nil {
		return err
	}
	if !validSHA256(m.Model.SHA256) || len(m.Model.Files) == 0 || !validSHA256(m.WeightsSHA256) || !validSHA256(m.SentinelSHA256) {
		return errors.New("seal manifest is incomplete")
	}
	return m.Mutation.validate()
}
