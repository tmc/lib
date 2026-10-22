package dsv4eval

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCorpusManifestMutations(t *testing.T) {
	for _, name := range []string{"revision", "extractor", "binary", "binary-nonhex", "sources", "license", "corpus", "files"} {
		t.Run(name, func(t *testing.T) {
			m := SmokeCorpusManifest{OracleRevision: SmokeOracleRevision, CorpusSHA256: SmokeCorpusSHA256, LicenseSHA256: SmokeLicenseSHA256, SourceSHA256: SmokeSourceHashes(), ExtractorVersion: 1, ExtractorSHA256: SmokeExtractorSHA256, BinarySHA256: strings.Repeat("a", 64), BuildCommand: []string{"cc", "extract.c"}}
			want := "unqualified oracle or corpus"
			switch name {
			case "revision":
				m.OracleRevision = "other"
			case "extractor":
				m.ExtractorVersion = 2
				want = "missing qualified extractor receipt"
			case "binary":
				m.BinarySHA256 = ""
				want = "missing qualified extractor receipt"
			case "binary-nonhex":
				m.BinarySHA256 = strings.Repeat("g", 64)
				want = "missing qualified extractor receipt"
			case "sources":
				delete(m.SourceSHA256, "ds4_eval.c")
				want = "wrong source inventory"
			case "license":
				m.LicenseSHA256 = "other"
			case "corpus":
				m.CorpusSHA256 = "other"
			case "files":
				want = "no such file"
			}
			dir := t.TempDir()
			data, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "manifest.json"), data, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadSmokeCorpus(dir); err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("mutation error=%v, want %q", err, want)
			}
		})
	}
}

func TestExternalCorpusAdmission(t *testing.T) {
	dir := os.Getenv("DS4_EVAL_CORPUS_OUT")
	if dir == "" {
		t.Skip("set DS4_EVAL_CORPUS_OUT to the source-generated external corpus")
	}
	cases, err := LoadSmokeCorpus(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) != 12 {
		t.Fatalf("admitted cases=%d, want 12", len(cases))
	}
}
