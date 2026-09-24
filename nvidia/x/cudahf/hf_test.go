package cudahf_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahf"
)

const repoID = "meta-llama/Llama-3.2-1B-Instruct"

// writeCache writes files, named relative to the repository directory, into
// a new hub cache and returns the cache directory.
func writeCache(t *testing.T, files map[string]string) string {
	t.Helper()
	cache := t.TempDir()
	repo := filepath.Join(cache, "models--meta-llama--Llama-3.2-1B-Instruct")
	for name, data := range files {
		path := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return cache
}

func TestResolveIn(t *testing.T) {
	tests := []struct {
		name     string
		files    map[string]string
		snapshot string
		weights  int
	}{
		{
			"refs/main",
			map[string]string{
				"refs/main":                                      "bbb\n",
				"snapshots/aaa/model.safetensors":                "",
				"snapshots/bbb/config.json":                      "{}",
				"snapshots/bbb/tokenizer.json":                   "{}",
				"snapshots/bbb/tokenizer_config.json":            "{}",
				"snapshots/bbb/model.safetensors.index.json":     "{}",
				"snapshots/bbb/model-00001-of-00002.safetensors": "",
				"snapshots/bbb/model-00002-of-00002.safetensors": "",
				"snapshots/ccc/model.safetensors":                "",
			},
			"bbb", 2,
		},
		{
			"one snapshot",
			map[string]string{"snapshots/aaa/model.safetensors": ""},
			"aaa", 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mf, err := cudahf.ResolveIn(writeCache(t, tt.files), repoID)
			if err != nil {
				t.Fatal(err)
			}
			if got := filepath.Base(mf.SnapshotDir); got != tt.snapshot {
				t.Errorf("snapshot = %s, want %s", got, tt.snapshot)
			}
			if len(mf.WeightsPaths) != tt.weights {
				t.Errorf("WeightsPaths = %v, want %d paths", mf.WeightsPaths, tt.weights)
			}
			for _, p := range mf.WeightsPaths {
				if filepath.Dir(p) != mf.SnapshotDir {
					t.Errorf("weights path %s is outside the snapshot", p)
				}
			}
			if tt.snapshot == "bbb" && (mf.ConfigPath == "" || mf.TokenizerPath == "" || mf.TokenizerConfigPath == "" || mf.IndexPath == "") {
				t.Errorf("missing paths: %+v", mf)
			}
		})
	}
}

func TestResolveInRejects(t *testing.T) {
	tests := []struct {
		name  string
		files map[string]string
		want  string
	}{
		{"no repository", nil, "no such file"},
		{"two snapshots", map[string]string{"snapshots/aaa/x": "", "snapshots/bbb/x": ""}, "2 snapshots and no refs/main"},
		{"escaping ref", map[string]string{"refs/main": "../../x", "snapshots/aaa/x": ""}, "not a snapshot name"},
		{"missing ref target", map[string]string{"refs/main": "bbb", "snapshots/aaa/x": ""}, "no such file"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := cudahf.ResolveIn(writeCache(t, tt.files), repoID)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ResolveIn error = %v, want %q", err, tt.want)
			}
		})
	}
	if _, err := cudahf.ResolveIn("", repoID); err == nil {
		t.Error("ResolveIn accepted an empty cache directory")
	}
}

func TestDefaultCacheDir(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", "/hub")
	t.Setenv("HF_HOME", "/hf")
	t.Setenv("XDG_CACHE_HOME", "/xdg")
	if got := cudahf.DefaultCacheDir(); got != "/hub" {
		t.Errorf("with HF_HUB_CACHE: %s", got)
	}
	t.Setenv("HF_HUB_CACHE", "")
	if got := cudahf.DefaultCacheDir(); got != filepath.Join("/hf", "hub") {
		t.Errorf("with HF_HOME: %s", got)
	}
	t.Setenv("HF_HOME", "")
	if got := cudahf.DefaultCacheDir(); got != filepath.Join("/xdg", "huggingface", "hub") {
		t.Errorf("with XDG_CACHE_HOME: %s", got)
	}
}
