//go:build darwin && arm64

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestMetalDSparkRawConfidenceSchedulerPinnedC(t *testing.T) {
	source, err := os.ReadFile("testdata/dspark_scheduler_11e8cad.c")
	if err != nil {
		t.Fatal(err)
	}
	const site = "s.dspark_last_confidence0=0.5f;"
	if strings.Count(string(source), site) != 1 {
		t.Fatal("pinned raw confidence fixture site is missing or duplicated")
	}
	body := strings.Replace(string(source), site, "s.dspark_last_confidence0=strtof(argv[2],NULL);", 1)
	dir := t.TempDir()
	path, binary := filepath.Join(dir, "confidence.c"), filepath.Join(dir, "confidence")
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("clang", "-std=c11", path, "-lm", "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile pinned confidence scheduler: %v\n%s", err, out)
	}
	for _, logit := range []float32{.25, .5, 1} {
		cmd := exec.Command(binary, "0", strconv.FormatFloat(float64(logit), 'g', -1, 32))
		for _, entry := range os.Environ() {
			if !strings.HasPrefix(entry, "DS4_DSPARK_") {
				cmd.Env = append(cmd.Env, entry)
			}
		}
		out, err := cmd.Output()
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		if len(lines) != 70 {
			t.Fatal("missing scheduler rows")
		}
		c := &metalDSparkCycle{schedule: dsparkSchedule{policy: newDSparkSchedulePolicy(false, false, func(string) string { return "" })}, targetMS: 2}
		c.noteConfidence(0, 4, logit)
		fields := strings.Fields(lines[0])
		want, err := strconv.ParseUint(fields[4], 10, 32)
		if err != nil || c.schedule.skip != uint32(want) {
			t.Fatalf("raw logit%g Go skip%d C skip%s", logit, c.schedule.skip, fields[4])
		}
		t.Logf("raw_logit=%g C_skip=%d Go_skip=%d", logit, want, c.schedule.skip)
		if logit == .25 {
			mutant := &metalDSparkCycle{schedule: dsparkSchedule{policy: c.schedule.policy}, targetMS: 2}
			mutant.noteConfidence(0, 4, float32(1/(1+math.Exp(-float64(logit)))))
			if mutant.schedule.skip == c.schedule.skip {
				t.Fatal("probability-as-logit mutation was not discriminated")
			}
			t.Logf("sigmoid_mutant_skip=%d expected_raw_skip=%d", mutant.schedule.skip, c.schedule.skip)
		}
	}
}

func TestMetalDSparkRawConfidenceSource(t *testing.T) {
	source, err := os.ReadFile("dspark_metal_cycle_darwin.go")
	if err != nil {
		t.Fatal(err)
	}
	check := func(source string) bool {
		f, err := parser.ParseFile(token.NewFileSet(), "cycle.go", source, 0)
		if err != nil {
			return false
		}
		found := 0
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Name.Name != "note" {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				method, ok := call.Fun.(*ast.SelectorExpr)
				if !ok || method.Sel.Name != "noteConfidence" || len(call.Args) != 3 {
					return true
				}
				ast.Inspect(call.Args[2], func(n ast.Node) bool {
					if selector, ok := n.(*ast.SelectorExpr); ok && selector.Sel.Name == "confidenceLogits" {
						found++
					}
					return true
				})
				return true
			})
		}
		return found == 1
	}
	if !check(string(source)) {
		t.Fatal("actual cycle does not consume explicit raw confidence readout")
	}
	mutant := strings.Replace(string(source), "c.drafter.forward.confidenceLogits.HostSlice()[0]", "c.drafter.forward.confidence.HostSlice()[0]", 1)
	if check(mutant) {
		t.Fatal("probability readout source mutation survived")
	}
}
