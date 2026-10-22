package main

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
	"time"
)

const powerFixtureSHA = "813766fc02149e470cdf88fb4396e401f1eb24fbaa576b0431e6431fcd9e04d6"

func TestPowerLiteralC(t *testing.T) {
	data, err := os.ReadFile("testdata/power-oracle-11e8cad.c")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != powerFixtureSHA {
		t.Fatalf("oracle source identity %s", got)
	}
	var source strings.Builder
	source.WriteString("#include <math.h>\n#include <stdint.h>\n#include <stdio.h>\n#define DS4_N_LAYER 43\nstruct ds4_gpu_graph { uint32_t power_percent; double prefill_layer_avg_sec[43], decode_token_avg_sec; };\nstatic double last_sleep;\nstatic void sleep_sec(double x) { last_sleep=x; }\n")
	for _, name := range []string{"graph_power_throttle_enabled", "graph_power_update_avg", "graph_power_sleep", "graph_power_note_prefill_layer", "graph_power_note_decode_token"} {
		text := string(data)
		at := strings.Index(text, name+"(")
		if at < 0 {
			t.Fatalf("missing %s", name)
		}
		begin := strings.LastIndex(text[:at], "\n") + 1
		brace := strings.Index(text[at:], "{") + at
		depth, end := 0, -1
		for i := brace; i < len(text); i++ {
			if text[i] == '{' {
				depth++
			}
			if text[i] == '}' {
				depth--
				if depth == 0 {
					end = i + 1
					break
				}
			}
		}
		if end < 0 {
			t.Fatalf("unterminated %s", name)
		}
		source.WriteString(text[begin:end] + "\n")
	}
	source.WriteString("int main() { for(int power=1;power<=100;power+=11) { ds4_gpu_graph g={};g.power_percent=power;double values[]={.008,.016,0,-1,NAN,INFINITY,.004};for(double v:values){last_sleep=0;graph_power_note_prefill_layer(&g,0,v);printf(\"%d %.17g %.17g\\n\",power,g.prefill_layer_avg_sec[0],last_sleep);} } }\n")
	dir := t.TempDir()
	cpp := filepath.Join(dir, "literal.cc")
	bin := filepath.Join(dir, "literal")
	if err := os.WriteFile(cpp, []byte(source.String()), 0600); err != nil {
		t.Fatal(err)
	}
	compiler := os.Getenv("CXX")
	if compiler == "" {
		compiler = "c++"
	}
	if out, err := exec.Command(compiler, "-std=c++17", cpp, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("literal C compile: %v\n%s", err, out)
	}
	out, err := exec.Command(bin).Output()
	if err != nil {
		t.Fatal(err)
	}
	values := []float64{.008, .016, 0, -1, math.NaN(), math.Inf(1), .004}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	line := 0
	for percent := 1; percent <= 100; percent += 11 {
		p := newPowerControl(percent, 43)
		var slept time.Duration
		if p != nil {
			p.sleep = func(d time.Duration) { slept = d }
		}
		for _, v := range values {
			fields := strings.Fields(lines[line])
			line++
			avg, err := strconv.ParseFloat(fields[1], 64)
			if err != nil {
				t.Fatal(err)
			}
			seconds, err := strconv.ParseFloat(fields[2], 64)
			if err != nil {
				t.Fatal(err)
			}
			slept = 0
			got := 0.0
			if p != nil {
				p.layers[0] = powerAverage(p.layers[0], v)
				p.pause(p.layers[0])
				got = p.layers[0]
			}
			if got != avg || slept != time.Duration(seconds*1e9) {
				t.Fatalf("literal power=%d sample=%g: avg %g/%g sleep %v/%v", percent, v, got, avg, slept, time.Duration(seconds*1e9))
			}
		}
	}
}

func TestPowerDisabledPacingMutation(t *testing.T) {
	source, err := os.ReadFile("power_control.go")
	if err != nil {
		t.Fatal(err)
	}
	tests, err := os.ReadFile("power_control_test.go")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	// The policy fixture injects every clock/sleeper. This default is never
	// invoked and is not a native interrupted-sleep qualification.
	contract, err := os.ReadFile("power_sleep_contract.go")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sleep_contract.go"), contract, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sleep_stub.go"), []byte("package main\nimport \"time\"\nfunc powerSleep(time.Duration) { panic(\"uninjected sleeper\") }\nfunc powerSleepSupported() error { return nil }\n"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []bool{false, true} {
		body := string(source)
		if mutate {
			needle := "p.sleep(time.Duration(nanos))"
			if strings.Count(body, needle) != 1 {
				t.Fatal("mutation seam")
			}
			body = strings.Replace(body, needle, "_ = nanos", 1)
		}
		if err := os.WriteFile(filepath.Join(dir, "power.go"), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "power_test.go"), tests, 0600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("go", "test", "-timeout=60s", "-count=1", "-run", "^TestPowerMeasuredBoundaries$", "power.go", "power_test.go", "sleep_stub.go", "sleep_contract.go")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if !mutate && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		exit, ok := err.(*exec.ExitError)
		if mutate && (!ok || exit.ExitCode() != 1 || !strings.Contains(string(out), "--- FAIL: TestPowerMeasuredBoundaries") || !strings.Contains(string(out), "measured sleeps") || strings.Contains(string(out), "[build failed]") || strings.Contains(string(out), "panic:")) {
			t.Fatalf("disabled pacing mutant did not fail intended assertion: %v\n%s", err, out)
		}
	}
}
