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

func TestDSparkSchedulerPinnedC(t *testing.T) {
	compiler, err := exec.LookPath("clang")
	if err != nil {
		t.Fatal("pinned scheduler control requires clang")
	}
	binary := filepath.Join(t.TempDir(), "scheduler")
	if out, err := exec.Command(compiler, "-std=c11", "testdata/dspark_scheduler_11e8cad.c", "-lm", "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile pinned C scheduler: %v\n%s", err, out)
	}
	cases := []struct {
		name  string
		spark bool
		env   map[string]string
	}{
		{"default", false, nil},
		{"spark", true, nil},
		{"disabled", true, map[string]string{"DS4_DSPARK_SCHEDULER": "0"}},
		{"no pauses", false, map[string]string{"DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP": "0", "DS4_DSPARK_SCHEDULER_SKIP": "0", "DS4_DSPARK_SCHEDULER_SLOW_SKIP": "0"}},
		{"short window", true, map[string]string{"DS4_DSPARK_SCHEDULER_WINDOW": "2", "DS4_DSPARK_SCHEDULER_SKIP": "1"}},
		{"timing", true, map[string]string{"DS4_DSPARK_SCHEDULER_MAX_MS_PER_ACCEPT_MILLI": "1000"}},
		{"break even", true, map[string]string{"DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW": "2", "DS4_DSPARK_SCHEDULER_MAX_EXTRA_SAVED_RATIO_MILLI": "1"}},
		{"numeric prefix", false, map[string]string{"DS4_DSPARK_SCHEDULER_WINDOW": "  +2tail", "DS4_DSPARK_SCHEDULER_SKIP": "-1", "DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP": "4294967296"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cmd := exec.Command(binary, strconv.FormatBool(tc.spark))
			if tc.spark {
				cmd.Args[1] = "1"
			} else {
				cmd.Args[1] = "0"
			}
			for _, entry := range os.Environ() {
				if !strings.HasPrefix(entry, "DS4_DSPARK_") {
					cmd.Env = append(cmd.Env, entry)
				}
			}
			for k, v := range tc.env {
				cmd.Env = append(cmd.Env, k+"="+v)
			}
			out, err := cmd.Output()
			if err != nil {
				t.Fatal(err)
			}
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			if len(lines) != 70 {
				t.Fatalf("source trace rows=%d", len(lines))
			}
			s := dsparkSchedule{policy: newDSparkSchedulePolicy(tc.spark, tc.spark, func(k string) string { return tc.env[k] })}
			accepted := []uint32{0, 0, 0, 1, 0, 3, 0, 0, 1, 1, 1, 1, 3, 0, 0, 0, 3, 3, 3, 3}
			for i, line := range lines {
				skipped := s.shouldSkip()
				n := accepted[i%len(accepted)]
				s.note(n, n == 0, 4, 2, 0.5, true)
				got := fmt.Sprintf("%t %d %d %d %d %d %t", skipped, s.cycles, s.accepted, s.noDraft, s.skip, s.lifetime, s.longAccept)
				fields := strings.Fields(line)
				fields[0] = strconv.FormatBool(fields[0] == "1")
				fields[6] = strconv.FormatBool(fields[6] == "1")
				want := strings.Join(fields, " ")
				if got != want {
					t.Fatalf("cycle%d Go=%s C=%s", i, got, want)
				}
			}
		})
	}
}

func TestDSparkSchedulerEnvUint(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  uint32
	}{
		{"", 7}, {"0", 0}, {"-0", 0}, {"-1", 7}, {"  +12tail", 12},
		{"4294967295", 4294967295}, {"4294967296", 7}, {"abc", 7}, {"0x10", 0},
		{"\u200312", 7},
	} {
		if got := dsparkEnvUint(tc.value, 7); got != tc.want {
			t.Errorf("%q=%d want%d", tc.value, got, tc.want)
		}
	}
}
