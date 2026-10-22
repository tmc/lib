package sampler

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestProbabilitiesPinnedC(t *testing.T) {
	fixture, err := os.ReadFile("testdata/probabilities_11e8cad.c")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(fixture)); got != "ba0edc0600697fd5a0fb90999a783e499aae1645b58721038a18cb2c89672e8e" {
		t.Fatalf("pinned C fixture hash changed: %s", got)
	}
	lines := strings.SplitAfter(string(fixture), "\n")
	if len(lines) < 331 {
		t.Fatal("pinned C source body truncated")
	}
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Join(lines[10:331], "")))); got != "28e7d2465b3cd14ff3d6cef02d367e33566e95f6a62257776add5e2eb0a1e8ec" {
		t.Fatalf("pinned C body hash changed: %s", got)
	}
	name := os.Getenv("DS4_PROBABILITIES_CC")
	wantHash := os.Getenv("DS4_PROBABILITIES_CC_SHA256")
	if name == "" {
		if wantHash != "" {
			t.Fatal("compiler hash requires an explicit compiler")
		}
		name = "clang"
	} else if !filepath.IsAbs(name) || len(wantHash) != 64 {
		t.Fatal("explicit probability compiler requires an absolute path and SHA256")
	}
	compiler, err := exec.LookPath(name)
	if err != nil {
		t.Fatalf("pinned probabilities compiler %q: %v", name, err)
	}
	compiler, err = filepath.EvalSymlinks(compiler)
	if err != nil {
		t.Fatal(err)
	}
	compilerBytes, err := os.ReadFile(compiler)
	if err != nil {
		t.Fatal(err)
	}
	compilerHash := fmt.Sprintf("%x", sha256.Sum256(compilerBytes))
	if wantHash != "" && compilerHash != wantHash {
		t.Fatalf("probability compiler hash: got %s want %s", compilerHash, wantHash)
	}
	version, err := exec.Command(compiler, "--version").CombinedOutput()
	if err != nil {
		t.Fatalf("probability compiler version: %v\n%s", err, version)
	}
	flags := []string{"-std=c11", "-O0", "-fno-fast-math", "-ffp-contract=off"}
	t.Logf("probability compiler=%s sha256=%s flags=%q version=%s", compiler, compilerHash, flags, strings.TrimSpace(string(version)))
	binary := filepath.Join(t.TempDir(), "probabilities")
	args := append(append([]string(nil), flags...), "testdata/probabilities_11e8cad.c", "-lm", "-o", binary)
	if out, err := exec.Command(compiler, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile C probability control: %v\n%s", err, out)
	}
	cases := 0
	for _, tc := range []struct {
		n    int
		kind string
	}{{3, "mixed"}, {17, "mixed"}, {513, "mixed"}, {1100, "mixed"}, {513, "uniform"}, {1025, "uniform"}, {17, "masked"}, {17, "all masked"}, {1, "mixed"}} {
		n := tc.n
		logits := make([]float32, n)
		var input strings.Builder
		for i := range logits {
			logits[i] = float32((i*17)%53-26) * 0.25
			switch tc.kind {
			case "uniform":
				logits[i] = 0
			case "masked":
				if i%3 != 1 {
					logits[i] = negInf
				}
			case "all masked":
				logits[i] = negInf
			}
			fmt.Fprintf(&input, "%.9g\n", logits[i])
		}
		for _, p := range []Params{{Temperature: 1, TopP: 1}, {Temperature: 0.7, TopP: 0.95, MinP: 0.05}, {Temperature: 1, TopK: 2, TopP: 1}, {Temperature: 1, TopK: 1025, TopP: 0.5}, {Temperature: 1, TopP: 1, MinP: 0.4}, {Temperature: 0}, {Temperature: 1, TopK: 1, TopP: 1}, {Temperature: 1, TopK: 511, TopP: 1}, {Temperature: 1, TopK: 512, TopP: 1}, {Temperature: 1, TopK: 513, TopP: 1}, {Temperature: 1, TopK: 1024, TopP: 1}, {Temperature: 1, TopK: -1, TopP: 2, MinP: -1}, {Temperature: 1, TopP: 1, MinP: 2}} {
			cmd := exec.Command(binary, strconv.Itoa(n), fmt.Sprint(p.Temperature), strconv.Itoa(p.TopK), fmt.Sprint(p.TopP), fmt.Sprint(p.MinP))
			cmd.Stdin = strings.NewReader(input.String())
			out, err := cmd.Output()
			if err != nil {
				t.Fatal(err)
			}
			fields := strings.Fields(string(out))
			if len(fields) != n {
				t.Fatal("source probability extent mismatch")
			}
			got, err := New(n).Probabilities(logits, p)
			if err != nil {
				t.Fatal(err)
			}
			var goSum, cSum float64
			goSupport, cSupport := 0, 0
			for i, field := range fields {
				want, err := strconv.ParseFloat(field, 64)
				if err != nil {
					t.Fatal(err)
				}
				if math.IsNaN(want) || math.IsInf(want, 0) || !isFinite(got[i]) || want < 0 || got[i] < 0 || want > 1.000001 || got[i] > 1.000001 {
					t.Fatalf("nonfinite or invalid probability at token %d: Go%g C%g", i, got[i], want)
				}
				goSum += float64(got[i])
				cSum += want
				if got[i] > 0 {
					goSupport++
				}
				if want > 0 {
					cSupport++
				}
				if (want == 0) != (got[i] == 0) || math.Abs(float64(got[i])-want) > 2e-6 {
					t.Fatalf("vocab%d params%+v token%d Go%g C%g", n, p, i, got[i], want)
				}
			}
			if goSupport == 0 || goSupport != cSupport || math.Abs(goSum-1) > 2e-5 || math.Abs(cSum-1) > 2e-5 {
				t.Fatalf("distribution extent/normalization Go%g/%d C%g/%d", goSum, goSupport, cSum, cSupport)
			}
			cases++
		}
	}
	if cases != 117 {
		t.Fatalf("probability control cases: got %d want 117", cases)
	}
	t.Logf("probability control cases=%d", cases)
}
