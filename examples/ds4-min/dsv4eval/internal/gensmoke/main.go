// Command gensmoke extracts external evaluation data from hash-qualified
// upstream source. It never writes questions or licenses into the repository.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
)

type record struct {
	Source, ID, Domain, Title, Question, Answer string
	Kind                                        dsv4eval.AnswerKind
	Suites                                      dsv4eval.Suite
	MaxTokens                                   int
	Choices, Aliases                            []string
}

func main() {
	if err := generate(); err != nil {
		fmt.Fprintln(os.Stderr, "gensmoke:", err)
		os.Exit(1)
	}
}
func generate() error {
	source, out := os.Getenv("DS4_ORACLE_SOURCE"), os.Getenv("DS4_EVAL_CORPUS_OUT")
	if source == "" || out == "" {
		return fmt.Errorf("set DS4_ORACLE_SOURCE and external DS4_EVAL_CORPUS_OUT")
	}
	abs, err := filepath.Abs(out)
	if err != nil {
		return err
	}
	if err := externalOutput(abs); err != nil {
		return err
	}
	if _, err := os.Lstat(abs); err == nil {
		return fmt.Errorf("output already exists; preserve it and choose a new directory")
	} else if !os.IsNotExist(err) {
		return err
	}
	extractor, err := os.ReadFile("testdata/extract_smoke.c")
	if err != nil {
		return err
	}
	if fmt.Sprintf("%x", sha256.Sum256(extractor)) != dsv4eval.SmokeExtractorSHA256 {
		return fmt.Errorf("unqualified extractor")
	}
	for name, want := range dsv4eval.SmokeSourceHashes() {
		data, err := os.ReadFile(filepath.Join(source, name))
		if err != nil {
			return err
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != want {
			return fmt.Errorf("unqualified oracle source %s", name)
		}
	}
	license, err := os.ReadFile(filepath.Join(source, "licenses", "Apache-2.0.txt"))
	if err != nil {
		return err
	}
	if fmt.Sprintf("%x", sha256.Sum256(license)) != dsv4eval.SmokeLicenseSHA256 {
		return fmt.Errorf("unqualified source license")
	}
	if err = os.MkdirAll(filepath.Dir(abs), 0700); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(filepath.Dir(abs), ".smoke-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	binary := filepath.Join(tmp, "extract")
	cmd := exec.Command("cc", "-I", source, "testdata/extract_smoke.c", filepath.Join(source, "ds4_eval_cases.c"), "-o", binary)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("compile extractor: %w: %s", err, output)
	}
	data, err := exec.Command(binary).Output()
	if err != nil {
		return fmt.Errorf("extract corpus: %w", err)
	}
	var records []record
	if err := json.Unmarshal(data, &records); err != nil {
		return err
	}
	var canonical bytes.Buffer
	encoder := json.NewEncoder(&canonical)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(records); err != nil {
		return err
	}
	if fmt.Sprintf("%x", sha256.Sum256(canonical.Bytes())) != dsv4eval.SmokeCorpusSHA256 {
		return fmt.Errorf("generated corpus differs from pinned twelve-case identity/budget/content contract")
	}
	binaryBytes, err := os.ReadFile(binary)
	if err != nil {
		return err
	}
	manifest := dsv4eval.SmokeCorpusManifest{OracleRevision: dsv4eval.SmokeOracleRevision, CorpusSHA256: dsv4eval.SmokeCorpusSHA256, SourceSHA256: dsv4eval.SmokeSourceHashes(), LicenseSHA256: dsv4eval.SmokeLicenseSHA256, ExtractorVersion: 1, ExtractorSHA256: dsv4eval.SmokeExtractorSHA256, BinarySHA256: fmt.Sprintf("%x", sha256.Sum256(binaryBytes)), BuildCommand: cmd.Args}
	m, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	m = append(m, '\n')
	for name, data := range map[string][]byte{"hard-smoke.json": canonical.Bytes(), "Apache-2.0.txt": license, "manifest.json": m} {
		if err := os.WriteFile(filepath.Join(tmp, name), data, 0600); err != nil {
			return err
		}
	}
	if _, err = dsv4eval.LoadSmokeCorpus(tmp); err != nil {
		return err
	}
	if err = os.Remove(binary); err != nil {
		return err
	}
	return os.Rename(tmp, abs)
}

func externalOutput(path string) error {
	// Resolve existing ancestors before creating anything, including symlinks.
	for dir := path; ; dir = filepath.Dir(dir) {
		resolved, err := filepath.EvalSymlinks(dir)
		if err == nil {
			for ancestor := resolved; ; ancestor = filepath.Dir(ancestor) {
				if _, err := os.Lstat(filepath.Join(ancestor, ".git")); err == nil {
					return fmt.Errorf("output is inside a git worktree")
				}
				if ancestor == filepath.Dir(ancestor) {
					break
				}
			}
			return nil
		}
		if !os.IsNotExist(err) {
			return err
		}
		if dir == filepath.Dir(dir) {
			return err
		}
	}
}
