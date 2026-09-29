package cuptipcsampling_test

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func copyReferenceSampleTree(t *testing.T, tutorialRoot, sampleDir string) (string, error) {
	t.Helper()
	rel, err := filepath.Rel(tutorialRoot, sampleDir)
	if err != nil {
		return "", err
	}
	if filepath.IsAbs(rel) || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("sample directory %q is outside tutorial root %q", sampleDir, tutorialRoot)
	}
	destination := filepath.Join(t.TempDir(), "tutorial")
	if err := copyReferenceTree(tutorialRoot, destination); err != nil {
		return "", err
	}
	return filepath.Join(destination, rel), nil
}

func copyReferenceTree(source, destination string) error {
	return filepath.WalkDir(source, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() && entry.Name() == ".git" {
			return fs.SkipDir
		}
		rel, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destination, rel)
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode().Perm())
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("unsupported reference sample file %q", path)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, info.Mode().Perm())
	})
}

func TestCopyReferenceSampleTreeIsIsolated(t *testing.T) {
	tutorialRoot := t.TempDir()
	files := map[string]string{
		"pc_sampling_start_stop/Makefile": "NVCCFLAGS += -I../common\nall:\n\t@touch built\n",
		"pc_sampling_start_stop/main.cpp": "#include \"helper_cupti.h\"\n",
		"common/helper_cupti.h":           "/* common helper */\n",
		".git/config":                     "source-only git metadata\n",
	}
	for name, content := range files {
		path := filepath.Join(tutorialRoot, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	sourceSample := filepath.Join(tutorialRoot, "pc_sampling_start_stop")
	copiedSample, err := copyReferenceSampleTree(t, tutorialRoot, sourceSample)
	if err != nil {
		t.Fatal(err)
	}
	for _, rel := range []string{"Makefile", "main.cpp", "../common/helper_cupti.h"} {
		sourcePath := filepath.Join(sourceSample, rel)
		want, err := os.ReadFile(sourcePath)
		if err != nil {
			t.Fatalf("read source %s: %v", rel, err)
		}
		got, err := os.ReadFile(filepath.Join(copiedSample, rel))
		if err != nil {
			t.Fatalf("read copied %s: %v", rel, err)
		}
		if string(got) != string(want) {
			t.Errorf("copied %s = %q, want %q", rel, got, want)
		}
	}
	if _, err := os.Stat(filepath.Join(copiedSample, "../.git/config")); !os.IsNotExist(err) {
		t.Fatalf("copied tutorial tree includes .git/config: err=%v", err)
	}
	if err := os.WriteFile(filepath.Join(copiedSample, "Makefile"), []byte("built\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(sourceSample, "Makefile"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != files["pc_sampling_start_stop/Makefile"] {
		t.Fatalf("source sample changed to %q", got)
	}
}
