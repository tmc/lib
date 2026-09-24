package trainstat

import (
	"bufio"
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestReportHelper is run by TestReport in a child test binary.
func TestReportHelper(t *testing.T) {
	if os.Getenv("TRAINSTAT_HELPER") == "" {
		t.Skip("run by TestReport")
	}
	Report(t, "", Value{0.25, "loss"})
	Report(t, "Final/split=val", Value{0.8119, "accuracy"}, Value{1.0 / 3, "ece"})
	w := Artifact(t)
	w.Config("seed", "1")
	w.Step(1, Value{1.5, "loss"})
}

func BenchmarkReportHelper(b *testing.B) {
	if os.Getenv("TRAINSTAT_HELPER") == "" {
		b.Skip("run by TestReport")
	}
	for b.Loop() {
	}
	Report(b, "", Value{0.75, "accuracy"})
}

func TestReport(t *testing.T) {
	for _, artifacts := range []bool{false, true} {
		out := t.TempDir()
		args := []string{"-test.run=^TestReportHelper$", "-test.bench=^BenchmarkReportHelper$", "-test.benchtime=1x", "-test.v", "-test.outputdir=" + out}
		if artifacts {
			args = append(args, "-test.artifacts")
		}
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), "TRAINSTAT_HELPER=1")
		stdout, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%v\n%s", err, stdout)
		}

		attrs := make(map[string]string)
		var bench, dir string
		sc := bufio.NewScanner(bytes.NewReader(stdout))
		for sc.Scan() {
			line := sc.Text()
			if rest, ok := strings.CutPrefix(line, "=== ATTR  "); ok {
				f := strings.Fields(rest)
				attrs[f[0]+" "+f[1]] = f[2]
			}
			if rest, ok := strings.CutPrefix(line, "=== ARTIFACTS "); ok {
				dir = strings.Fields(rest)[1]
			}
			if strings.HasPrefix(line, "BenchmarkReportHelper") && strings.Contains(line, "accuracy") {
				bench = line
			}
		}
		for key, want := range map[string]string{
			"TestReportHelper trainstat:loss":                     "0.25",
			"TestReportHelper trainstat:Final/split=val:accuracy": "0.8119",
			"TestReportHelper trainstat:Final/split=val:ece":      "0.3333333333333333",
			"BenchmarkReportHelper trainstat:accuracy":            "0.75",
		} {
			if attrs[key] != want {
				t.Errorf("attr %s = %q, want %q\n%s", key, attrs[key], want, stdout)
			}
		}
		if !strings.Contains(bench, "0.7500 accuracy") {
			t.Errorf("benchmark line %q does not report accuracy", bench)
		}

		if !artifacts {
			if dir != "" {
				t.Errorf("artifacts directory %s without -test.artifacts", dir)
			}
			continue
		}
		if dir == "" {
			t.Fatalf("no artifacts directory\n%s", stdout)
		}
		b, err := os.ReadFile(filepath.Join(dir, "trainstat.txt"))
		if err != nil {
			t.Fatal(err)
		}
		want := "seed: 1\nUnit loss better=lower assume=normal\nBenchmarkTrain/step=1 1 1.5 loss\n"
		if string(b) != want {
			t.Errorf("trainstat.txt = %q, want %q", b, want)
		}
	}
}
