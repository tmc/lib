//go:build linux || darwin

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func watchWriteCPUArtifacts(t *testing.T, dir string, e watchEvidence) {
	t.Helper()
	for i, c := range e.Compilations {
		for suffix, body := range map[string][]byte{"cu": []byte(c.Source.Code), "ptx": []byte("synthetic PTX"), "cubin": []byte("synthetic CUBIN")} {
			if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("compile-%d.%s", i, suffix)), body, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestWatchRetainedCompilationBodies(t *testing.T) {
	for index := range 4 {
		for _, suffix := range []string{"cu", "ptx", "cubin"} {
			for _, name := range []string{"positive", "corrupt", "truncate", "digest", "extent", "missing", "symlink", "oversized_metadata"} {
				t.Run(fmt.Sprintf("compile-%d/%s/%s", index, suffix, name), func(t *testing.T) {
					e := watchEvidenceCPUFixture()
					dir := t.TempDir()
					watchWriteCPUArtifacts(t, dir, e)
					p := filepath.Join(dir, fmt.Sprintf("compile-%d.%s", index, suffix))
					switch name {
					case "corrupt":
						body, err := os.ReadFile(p)
						if err != nil {
							t.Fatal(err)
						}
						body[0] ^= 1
						if err := os.WriteFile(p, body, 0600); err != nil {
							t.Fatal(err)
						}
					case "truncate":
						if err := os.Truncate(p, 1); err != nil {
							t.Fatal(err)
						}
					case "digest":
						switch suffix {
						case "cu":
							e.Compilations[index].SourceSHA256 = watchDigest([]byte("contradictory body"))
						case "ptx":
							e.Compilations[index].PTXSHA256 = watchDigest([]byte("contradictory body"))
						case "cubin":
							e.Compilations[index].CUBINSHA256 = watchDigest([]byte("contradictory body"))
						}
					case "extent":
						switch suffix {
						case "cu":
							e.Compilations[index].Source.Code += " "
						case "ptx":
							e.Compilations[index].PTXBytes++
						case "cubin":
							e.Compilations[index].CUBINBytes++
						}
					case "missing":
						if err := os.Remove(p); err != nil {
							t.Fatal(err)
						}
					case "symlink":
						if err := os.Rename(p, p+".actual"); err != nil {
							t.Fatal(err)
						}
						if err := os.Symlink(p+".actual", p); err != nil {
							t.Fatal(err)
						}
					case "oversized_metadata":
						switch suffix {
						case "cu":
							e.Compilations[index].Source.Code = string(make([]byte, (8<<20)+1))
						case "ptx":
							e.Compilations[index].PTXBytes = (8 << 20) + 1
						case "cubin":
							e.Compilations[index].CUBINBytes = (64 << 20) + 1
						}
					}
					err := validateWatchArtifacts(dir, e)
					if (err == nil) != (name == "positive") {
						t.Fatalf("retained body validation: %v", err)
					}
				})
			}
		}
	}
}

func TestWatchPairedTimingTiesAndMixed(t *testing.T) {
	for _, ns := range []int64{10, 20, 30} {
		e := watchEvidenceCPUFixture()
		r := e.Events[2].Event.Result.Report.Replay
		for i := range r.Trials {
			if r.Trials[i].Arm == "candidate" {
				r.Trials[i].DeviceNanos = &ns
			}
		}
		r.CandidateWins, r.BaselineWins = 0, 0
		p := float64(1)
		if ns < 20 {
			r.CandidateWins, p = 5, 0.0625
		}
		if ns > 20 {
			r.BaselineWins, p = 5, 0.0625
		}
		r.SignP = &p
		if err := validateWatchEvidence(e); err != nil {
			t.Fatalf("valid timing %d: %v", ns, err)
		}
	}
	e := watchEvidenceCPUFixture()
	r := e.Events[2].Event.Result.Report.Replay
	count := 0
	for i := range r.Trials {
		if r.Trials[i].Arm != "candidate" {
			continue
		}
		ns := []int64{10, 30, 20, 10, 10}[count]
		count++
		r.Trials[i].DeviceNanos = &ns
	}
	r.CandidateWins, r.BaselineWins = 3, 1
	p := float64(0.625)
	r.SignP = &p
	if err := validateWatchEvidence(e); err != nil {
		t.Fatal(err)
	}
}
