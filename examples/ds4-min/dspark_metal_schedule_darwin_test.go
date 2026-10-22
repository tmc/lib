//go:build darwin && arm64

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestMetalDSparkSchedulePinnedC(t *testing.T) {
	source, err := os.ReadFile("testdata/dspark_scheduler_11e8cad.c")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	// Alter fixture platform facts only. Every copied scheduler body stays
	// literal; the Metal build excludes the Spark long-accept override.
	for _, change := range [][2]string{
		{"#undef __APPLE__", "#define __APPLE__ 1"},
		{"ds4_gpu_device_is_m5_apple_silicon(void) {return false;}", "ds4_gpu_device_is_m5_apple_silicon(void) {return true;}"},
		{"ds4_engine engine={.backend=1};", "ds4_engine engine={.backend=2};"},
	} {
		if strings.Count(text, change[0]) != 1 {
			t.Fatalf("fixture fact %q not unique", change[0])
		}
		text = strings.Replace(text, change[0], change[1], 1)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "metal-scheduler.c")
	binary := filepath.Join(dir, "scheduler")
	if err := os.WriteFile(path, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("clang", "-std=c11", path, "-lm", "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile Metal C facts: %v\n%s", err, out)
	}
	cmd := exec.Command(binary, "0")
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
		t.Fatalf("rows %d", len(lines))
	}
	s := dsparkSchedule{policy: newMetalDSparkSchedulePolicy(true, func(string) string { return "" })}
	accepted := []uint32{0, 0, 0, 1, 0, 3, 0, 0, 1, 1, 1, 1, 3, 0, 0, 0, 3, 3, 3, 3}
	for i, line := range lines {
		skipped := s.shouldSkip()
		n := accepted[i%len(accepted)]
		s.note(n, n == 0, 4, 2, .5, true)
		got := fmt.Sprintf("%t %d %d %d %d %d %t", skipped, s.cycles, s.accepted, s.noDraft, s.skip, s.lifetime, s.longAccept)
		fields := strings.Fields(line)
		fields[0] = strconv.FormatBool(fields[0] == "1")
		fields[6] = strconv.FormatBool(fields[6] == "1")
		if want := strings.Join(fields, " "); got != want {
			t.Fatalf("cycle %d Metal=%s C=%s", i, got, want)
		}
	}
}
