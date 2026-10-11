package cudasotaeval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// Construct manifests independently of WriteReportArtifacts so verifier tests
// do not inherit the writer's path or hash assumptions.
func writeTestManifest(t *testing.T, dir string, m ArtifactManifest) {
	t.Helper()
	keys := make([]string, 0, len(m.Files))
	for name := range m.Files {
		keys = append(keys, name)
	}
	sort.Strings(keys)
	var records strings.Builder
	for _, name := range keys {
		records.WriteString(name + ":" + m.Files[name] + "\n")
	}
	digest := sha256.Sum256([]byte(records.String()))
	if m.ManifestSHA256 == "" {
		m.ManifestSHA256 = hex.EncodeToString(digest[:])
	}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyManifestRefusals(t *testing.T) {
	const abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	tests := []struct {
		name    string
		change  func(*ArtifactManifest)
		prepare func(*testing.T, string)
	}{
		{name: "file count", change: func(m *ArtifactManifest) {
			m.Files = make(map[string]string)
			for i := 0; i <= maxArtifactFiles; i++ {
				m.Files[fmt.Sprint(i)] = abc
			}
		}},
		{name: "empty", change: func(m *ArtifactManifest) { m.Files = nil }},
		{name: "version", change: func(m *ArtifactManifest) { m.Version = 2 }},
		{name: "manifest digest", change: func(m *ArtifactManifest) { m.ManifestSHA256 = strings.Repeat("0", 64) }},
		{name: "file digest", change: func(m *ArtifactManifest) { m.Files["data"] = strings.Repeat("0", 64) }},
		{name: "malformed digest", change: func(m *ArtifactManifest) { m.Files["data"] = "xyz" }},
		{name: "parent path", change: func(m *ArtifactManifest) { m.Files = map[string]string{"../data": abc} }},
		{name: "absolute path", change: func(m *ArtifactManifest) { m.Files = map[string]string{"/data": abc} }},
		{name: "unclean path", change: func(m *ArtifactManifest) { m.Files = map[string]string{"sub/../data": abc} }},
		{name: "self reference", change: func(m *ArtifactManifest) { m.Files = map[string]string{"manifest.json": abc} }},
		{name: "missing file", prepare: func(t *testing.T, dir string) {
			if err := os.Remove(filepath.Join(dir, "data")); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "directory", prepare: func(t *testing.T, dir string) {
			os.Remove(filepath.Join(dir, "data"))
			if err := os.Mkdir(filepath.Join(dir, "data"), 0700); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "file symlink", prepare: func(t *testing.T, dir string) {
			os.Remove(filepath.Join(dir, "data"))
			if err := os.Symlink("other", filepath.Join(dir, "data")); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "directory symlink", change: func(m *ArtifactManifest) { m.Files = map[string]string{"sub/data": abc} }, prepare: func(t *testing.T, dir string) {
			if err := os.Symlink(".", filepath.Join(dir, "sub")); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "oversized artifact", prepare: func(t *testing.T, dir string) {
			f, err := os.OpenFile(filepath.Join(dir, "data"), os.O_WRONLY, 0600)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if err := f.Truncate(maxArtifactBytes + 1); err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "data"), []byte("abc"), 0600); err != nil {
				t.Fatal(err)
			}
			m := ArtifactManifest{Version: 1, Files: map[string]string{"data": abc}}
			if tt.change != nil {
				tt.change(&m)
			}
			if tt.prepare != nil {
				tt.prepare(t, dir)
			}
			writeTestManifest(t, dir, m)
			if _, err := VerifyManifest(dir); err == nil {
				t.Fatal("accepted invalid artifact manifest")
			}
		})
	}
}

func TestVerifyManifestIndependentFixture(t *testing.T) {
	dir := t.TempDir()
	const digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sub", "data"), []byte("abc"), 0600); err != nil {
		t.Fatal(err)
	}
	writeTestManifest(t, dir, ArtifactManifest{Version: 1, Files: map[string]string{"sub/data": digest}})
	if _, err := VerifyManifest(dir); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyManifestSizeAndSymlink(t *testing.T) {
	t.Run("manifest size", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(strings.Repeat(" ", maxManifestBytes+1)), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := VerifyManifest(dir); err == nil {
			t.Fatal("accepted oversized manifest")
		}
	})
	t.Run("manifest symlink", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "other"), []byte(`{}`), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink("other", filepath.Join(dir, "manifest.json")); err != nil {
			t.Fatal(err)
		}
		if _, err := VerifyManifest(dir); err == nil {
			t.Fatal("accepted symlink manifest")
		}
	})
}

func TestVerifyManifestDuplicateKeys(t *testing.T) {
	for _, input := range []string{
		`{"version":1,"version":2,"files":{}}`,
		`{"version":1,"files":{"data":"first","data":"second"}}`,
		`{"version":1,"files":{},"manifest_sha256":"first","manifest_sha256":"second"}`,
	} {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(input), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := VerifyManifest(dir); err == nil || !strings.Contains(err.Error(), "duplicate object key") {
			t.Fatalf("duplicate accepted or wrong refusal: %v", err)
		}
	}
}

func TestVerifyManifestCanonicalFields(t *testing.T) {
	const abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	for _, tt := range []struct {
		name   string
		mutate func(string) string
	}{
		{"case alias", func(s string) string { return strings.Replace(s, `"version":1`, `"version":1,"Version":1`, 1) }},
		{"sole case alias", func(s string) string { return strings.Replace(s, `"version":1`, `"Version":1`, 1) }},
		{"unknown", func(s string) string { return strings.Replace(s, `"version":1`, `"version":1,"ignored":true`, 1) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "Data"), []byte("abc"), 0600); err != nil {
				t.Fatal(err)
			}
			writeTestManifest(t, dir, ArtifactManifest{Version: 1, Files: map[string]string{"Data": abc}})
			path := filepath.Join(dir, "manifest.json")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(tt.mutate(string(data))), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := VerifyManifest(dir); err == nil {
				t.Fatal("admitted ambiguous manifest")
			}
		})
	}
	// Artifact paths are caller-owned and may have upper-case spelling.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "Data"), []byte("abc"), 0600); err != nil {
		t.Fatal(err)
	}
	writeTestManifest(t, dir, ArtifactManifest{Version: 1, Files: map[string]string{"Data": abc}})
	if _, err := VerifyManifest(dir); err != nil {
		t.Fatal(err)
	}
}
