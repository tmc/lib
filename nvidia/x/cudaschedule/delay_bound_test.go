package cudaschedule_test

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// singleVisitDelayBound is for the fixture's bounded inputs. It sums delays
// across threads and sites, with two injection positions and the PTX [0, 2*t]
// nanosleep range. It does not bound repeated visits or other kernel work.
func singleVisitDelayBound(threads, sites, ceiling uint64) uint64 {
	return threads * sites * 2 * 2 * ceiling
}

func TestSingleVisitDelayBound(t *testing.T) {
	for _, tt := range []struct {
		name                    string
		threads, sites, ceiling uint64
		want                    uint64
	}{
		{"no threads", 0, 3, 64000, 0},
		{"no sites", 64, 0, 64000, 0},
		{"zero ceiling", 64, 3, 0, 0},
		{"one site", 1, 1, 1, 4},
		{"illustrative three sites", 64, 3, 64000, 49152000},
		{"illustrative seven sites", 64, 7, 64000, 114688000},
		{"below watchdog", 1, 1, 749999999, 2999999996},
		{"at watchdog", 1, 1, 750000000, 3000000000},
		{"above watchdog", 1, 1, 750000001, 3000000004},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := singleVisitDelayBound(tt.threads, tt.sites, tt.ceiling); got != tt.want {
				t.Fatalf("bound=%d, want %d", got, tt.want)
			}
		})
	}
}

func TestRaceFixtureSingleVisitDelayBound(t *testing.T) {
	data, err := os.ReadFile("testdata/schedule_race.ptx")
	if err != nil {
		t.Fatal(err)
	}
	sites, err := cudaschedule.Analyze(string(data))
	if err != nil {
		t.Fatal(err)
	}
	if len(sites) == 0 {
		t.Fatal("fixture has no perturbation sites")
	}
	// Timer/output fixture revisions can add sites. The native calculation
	// must use the actual count while retaining both conservative factors.
	want := uint64(64) * uint64(len(sites)) * 4 * uint64(cudaschedule.DefaultMaxSleepNs)
	got := singleVisitDelayBound(64, uint64(len(sites)), uint64(cudaschedule.DefaultMaxSleepNs))
	if got != want {
		t.Fatalf("fixture bound=%d ns for %d sites, want %d", got, len(sites), want)
	}
	if got >= uint64((3 * time.Second).Nanoseconds()) {
		t.Fatalf("fixture single-visit bound=%d ns reaches watchdog", got)
	}
}

func TestNativeDelayBoundUsesConservativeArithmetic(t *testing.T) {
	data, err := os.ReadFile("native_test.go")
	if err != nil {
		t.Fatal(err)
	}
	// Bind the host-tested arithmetic and strict watchdog boundary to the
	// native test without executing its device-dependent workload.
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "native_test.go", data, 0)
	if err != nil {
		t.Fatal(err)
	}
	render := func(node ast.Node) string {
		var buf bytes.Buffer
		if err := format.Node(&buf, fset, node); err != nil {
			t.Fatal(err)
		}
		return buf.String()
	}
	assignments, boundaries := 0, 0
	ast.Inspect(file, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.AssignStmt:
			if len(node.Lhs) == 1 && render(node.Lhs[0]) == "worstInjected" {
				assignments++
				want := "singleVisitDelayBound(uint64(threadsPerBlock), uint64(len(sites)), uint64(cudaschedule.DefaultMaxSleepNs))"
				if len(node.Rhs) != 1 || render(node.Rhs[0]) != want {
					t.Error("native bound does not use host-tested arithmetic")
				}
			}
		case *ast.IfStmt:
			if strings.HasPrefix(render(node.Cond), "worstInjected ") {
				boundaries++
				if render(node.Cond) != "worstInjected >= uint64((3 * time.Second).Nanoseconds())" {
					t.Error("native watchdog boundary must reject equality")
				}
			}
		}
		return true
	})
	if assignments != 1 || boundaries != 1 {
		t.Fatalf("native bound assignments=%d, watchdog boundaries=%d; want one each", assignments, boundaries)
	}
	if !strings.Contains(string(data), "2 positions x 2 PTX duration factor") {
		t.Error("native log does not explain both delay factors")
	}
	limit := uint64((3 * time.Second).Nanoseconds())
	if got := singleVisitDelayBound(1, 1, 750000000); got < limit {
		t.Fatalf("watchdog equality must be rejected: bound=%d, limit=%d", got, limit)
	}
}
