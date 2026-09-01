package hfsync

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Store provides operations on the local Hugging Face model cache repository.
type Store struct {
	rootDir string
	mu      sync.RWMutex
}

// NewStore creates a Store backed by the provided root cache directory.
// If rootDir is empty, it defaults to ~/.cache/huggingface/hub.
func NewStore(rootDir string) *Store {
	if rootDir == "" {
		home, _ := os.UserHomeDir()
		rootDir = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	return &Store{rootDir: rootDir}
}

// RootDir returns the root cache directory path.
func (s *Store) RootDir() string {
	return s.rootDir
}

// ModelDirName transforms a Hugging Face repo ID (e.g. "unsloth/Llama-3.2-1B")
// into its canonical cache directory name ("models--unsloth--Llama-3.2-1B").
func ModelDirName(modelID string) string {
	parts := strings.Split(modelID, "/")
	if len(parts) == 1 {
		return "models--" + parts[0]
	}
	return "models--" + strings.Join(parts, "--")
}

// ModelPath returns the absolute path to the model's directory in the hub.
func (s *Store) ModelPath(modelID string) string {
	return filepath.Join(s.rootDir, ModelDirName(modelID))
}

// ScanModel inspects the local store and builds a Manifest of available snapshot files.
func (s *Store) ScanModel(modelID string) (*Manifest, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	modelDir := s.ModelPath(modelID)
	snapshotsDir := filepath.Join(modelDir, "snapshots")
	if _, err := os.Stat(snapshotsDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("model %q not found in cache: %w", modelID, os.ErrNotExist)
	}

	entries, err := os.ReadDir(snapshotsDir)
	if err != nil || len(entries) == 0 {
		return nil, fmt.Errorf("no snapshots found for model %q", modelID)
	}

	// Pick the first valid snapshot directory
	var snapshotID string
	for _, e := range entries {
		if e.IsDir() {
			snapshotID = e.Name()
			break
		}
	}
	if snapshotID == "" {
		return nil, fmt.Errorf("no snapshot directory found for %q", modelID)
	}

	snapshotPath := filepath.Join(snapshotsDir, snapshotID)
	var files []FileInfo

	err = filepath.Walk(snapshotPath, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() {
			return nil
		}

		relPath, err := filepath.Rel(snapshotPath, path)
		if err != nil {
			return err
		}

		hash, err := fileSHA256(path)
		if err != nil {
			return fmt.Errorf("hash file %s: %w", path, err)
		}

		files = append(files, FileInfo{
			Path:   relPath,
			Size:   info.Size(),
			SHA256: hash,
		})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("scan snapshot %s: %w", snapshotID, err)
	}

	return &Manifest{
		ModelID:  modelID,
		Revision: snapshotID,
		Files:    files,
	}, nil
}

// OpenFileReader opens a file within the model's active snapshot for reading.
func (s *Store) OpenFileReader(modelID, relPath string) (*os.File, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	manifest, err := s.ScanModel(modelID)
	if err != nil {
		return nil, err
	}

	filePath := filepath.Join(s.ModelPath(modelID), "snapshots", manifest.Revision, relPath)
	return os.Open(filePath)
}

// WriteChunk writes a data chunk to the target file at offset, creating parent directories if needed.
func (s *Store) WriteChunk(modelID, revision, relPath string, totalSize, offset int64, data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	targetDir := filepath.Join(s.ModelPath(modelID), "snapshots", revision, filepath.Dir(relPath))
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("create parent directory: %w", err)
	}

	targetPath := filepath.Join(s.ModelPath(modelID), "snapshots", revision, relPath)
	f, err := os.OpenFile(targetPath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("open target file: %w", err)
	}
	defer f.Close()

	if _, err := f.WriteAt(data, offset); err != nil {
		return fmt.Errorf("write at offset %d: %w", offset, err)
	}
	return nil
}

// VerifyFile checks if a local file matches the expected SHA256 checksum.
func (s *Store) VerifyFile(modelID, revision, relPath, expectedSHA256 string) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	targetPath := filepath.Join(s.ModelPath(modelID), "snapshots", revision, relPath)
	if _, err := os.Stat(targetPath); err != nil {
		return false, nil
	}

	actualHash, err := fileSHA256(targetPath)
	if err != nil {
		return false, err
	}
	return strings.EqualFold(actualHash, expectedSHA256), nil
}

func fileSHA256(path string) (string, error) {
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
