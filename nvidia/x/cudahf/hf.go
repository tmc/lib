package cudahf

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ModelFiles contains resolved filepaths for a local Hugging Face model repository.
type ModelFiles struct {
	RepoID              string
	SnapshotDir         string
	ConfigPath          string
	TokenizerPath       string
	TokenizerConfigPath string
	IndexPath           string
	WeightsPaths        []string
}

// DefaultCacheDir returns the standard Hugging Face hub cache directory.
func DefaultCacheDir() string {
	if hfHome := os.Getenv("HF_HOME"); hfHome != "" {
		return filepath.Join(hfHome, "hub")
	}
	if cacheHome := os.Getenv("XDG_CACHE_HOME"); cacheHome != "" {
		return filepath.Join(cacheHome, "huggingface", "hub")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// Resolve looks up the latest local snapshot directory for repoID.
func Resolve(repoID string) (*ModelFiles, error) {
	return ResolveIn(DefaultCacheDir(), repoID)
}

// ResolveIn resolves repoID within a specific hub cache directory.
func ResolveIn(cacheDir, repoID string) (*ModelFiles, error) {
	if cacheDir == "" {
		return nil, fmt.Errorf("cudahf: cache directory not found")
	}

	repoDirName := "models--" + strings.ReplaceAll(repoID, "/", "--")
	repoPath := filepath.Join(cacheDir, repoDirName)

	snapshotsDir := filepath.Join(repoPath, "snapshots")
	entries, err := os.ReadDir(snapshotsDir)
	if err != nil {
		return nil, fmt.Errorf("cudahf: read snapshots for %s: %w", repoID, err)
	}

	if len(entries) == 0 {
		return nil, fmt.Errorf("cudahf: no snapshots found for %s", repoID)
	}

	// Use latest snapshot directory entry
	latestSnapshot := entries[len(entries)-1].Name()
	snapshotPath := filepath.Join(snapshotsDir, latestSnapshot)

	mf := &ModelFiles{
		RepoID:      repoID,
		SnapshotDir: snapshotPath,
	}

	// Locate standard files
	cfg := filepath.Join(snapshotPath, "config.json")
	if _, err := os.Stat(cfg); err == nil {
		mf.ConfigPath = cfg
	}

	tok := filepath.Join(snapshotPath, "tokenizer.json")
	if _, err := os.Stat(tok); err == nil {
		mf.TokenizerPath = tok
	}

	tokCfg := filepath.Join(snapshotPath, "tokenizer_config.json")
	if _, err := os.Stat(tokCfg); err == nil {
		mf.TokenizerConfigPath = tokCfg
	}

	idx := filepath.Join(snapshotPath, "model.safetensors.index.json")
	if _, err := os.Stat(idx); err == nil {
		mf.IndexPath = idx
	}

	// Find .safetensors files
	files, err := os.ReadDir(snapshotPath)
	if err == nil {
		for _, f := range files {
			if strings.HasSuffix(f.Name(), ".safetensors") {
				mf.WeightsPaths = append(mf.WeightsPaths, filepath.Join(snapshotPath, f.Name()))
			}
		}
	}

	return mf, nil
}
