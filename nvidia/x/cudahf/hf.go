package cudahf

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ModelFiles holds the paths of the files in one cached snapshot of a model
// repository. A path is empty if the snapshot does not have that file.
type ModelFiles struct {
	RepoID              string
	SnapshotDir         string
	ConfigPath          string
	TokenizerPath       string
	TokenizerConfigPath string
	IndexPath           string
	WeightsPaths        []string
}

// DefaultCacheDir returns the hub cache directory huggingface_hub uses:
// $HF_HUB_CACHE, else $HF_HOME/hub, else $XDG_CACHE_HOME/huggingface/hub,
// else ~/.cache/huggingface/hub. It returns "" if none can be determined.
func DefaultCacheDir() string {
	if dir := os.Getenv("HF_HUB_CACHE"); dir != "" {
		return dir
	}
	if home := os.Getenv("HF_HOME"); home != "" {
		return filepath.Join(home, "hub")
	}
	if cache := os.Getenv("XDG_CACHE_HOME"); cache != "" {
		return filepath.Join(cache, "huggingface", "hub")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// Resolve resolves repoID in [DefaultCacheDir], as [ResolveIn] does.
func Resolve(repoID string) (*ModelFiles, error) {
	return ResolveIn(DefaultCacheDir(), repoID)
}

// ResolveIn finds the cached snapshot of repoID in the hub cache cacheDir.
// It uses the snapshot that refs/main names. Without refs/main, the
// repository must have exactly one snapshot.
func ResolveIn(cacheDir, repoID string) (*ModelFiles, error) {
	if cacheDir == "" {
		return nil, errors.New("cudahf: no hub cache directory")
	}
	repo := filepath.Join(cacheDir, "models--"+strings.ReplaceAll(repoID, "/", "--"))
	snapshot, err := snapshotName(repo)
	if err != nil {
		return nil, fmt.Errorf("cudahf: resolve %s: %w", repoID, err)
	}
	dir := filepath.Join(repo, "snapshots", snapshot)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("cudahf: resolve %s: %w", repoID, err)
	}

	mf := &ModelFiles{RepoID: repoID, SnapshotDir: dir}
	for _, e := range entries {
		path := filepath.Join(dir, e.Name())
		switch name := e.Name(); {
		case name == "config.json":
			mf.ConfigPath = path
		case name == "tokenizer.json":
			mf.TokenizerPath = path
		case name == "tokenizer_config.json":
			mf.TokenizerConfigPath = path
		case name == "model.safetensors.index.json":
			mf.IndexPath = path
		case strings.HasSuffix(name, ".safetensors"):
			mf.WeightsPaths = append(mf.WeightsPaths, path)
		}
	}
	return mf, nil
}

// snapshotName returns the name of the snapshot directory to use in repo.
func snapshotName(repo string) (string, error) {
	ref, err := os.ReadFile(filepath.Join(repo, "refs", "main"))
	if err == nil {
		name := strings.TrimSpace(string(ref))
		if name == "" || filepath.Base(name) != name || !filepath.IsLocal(name) {
			return "", fmt.Errorf("refs/main holds %q, not a snapshot name", name)
		}
		return name, nil
	}
	if !errors.Is(err, fs.ErrNotExist) {
		return "", err
	}
	entries, err := os.ReadDir(filepath.Join(repo, "snapshots"))
	if err != nil {
		return "", err
	}
	switch len(entries) {
	case 0:
		return "", errors.New("no snapshots")
	case 1:
		return entries[0].Name(), nil
	}
	return "", fmt.Errorf("%d snapshots and no refs/main", len(entries))
}
