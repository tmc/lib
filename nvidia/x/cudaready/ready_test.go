package cudaready

import (
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"
)

// TestHostRace runs testdata/ready_test.cc, which drives ready.cuh with
// host threads standing in for blocks: 300 epochs of producers that start
// in random order after random delays, with counters never reset. Every
// slot a consumer reads must hold its epoch's value. Built with NO_WAIT,
// the consumer skips the wait, and the test must then see stale slots, or
// it could not see a broken wait either. It checks the counting, not
// memory ordering: dropping a fence does not make it fail.
func TestHostRace(t *testing.T) {
	cxx, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("no c++")
	}
	tests := []struct {
		name   string
		flags  []string
		ok     bool
		stales func(n int) bool
	}{
		{"wait", nil, true, func(n int) bool { return n == 0 }},
		{"no wait", []string{"-DNO_WAIT"}, false, func(n int) bool { return n > 0 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bin := filepath.Join(t.TempDir(), "ready_test")
			args := append([]string{"-std=c++20", "-O2", "-pthread", "-o", bin}, tt.flags...)
			args = append(args, filepath.Join("testdata", "ready_test.cc"))
			if out, err := exec.Command(cxx, args...).CombinedOutput(); err != nil {
				t.Fatalf("c++: %v\n%s", err, out)
			}
			out, err := exec.Command(bin).CombinedOutput()
			if (err == nil) != tt.ok {
				t.Fatalf("ran with err = %v, want ok %v\n%s", err, tt.ok, out)
			}
			m := regexp.MustCompile(`(\d+) slots checked, (\d+) stale`).FindSubmatch(out)
			if m == nil {
				t.Fatalf("no tally in output:\n%s", out)
			}
			stale, _ := strconv.Atoi(string(m[2]))
			if !tt.stales(stale) {
				t.Errorf("%s slots checked, %d stale", m[1], stale)
			}
			t.Logf("%s", out)
		})
	}
}
