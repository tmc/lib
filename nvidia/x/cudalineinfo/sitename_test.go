package cudalineinfo

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestSiteNames(t *testing.T) {
	for _, tt := range []struct {
		name  string
		sites []Site
		want  []string
	}{{
		name:  "one file",
		sites: []Site{{"a/b/kernels.go", 12}, {"a/b/kernels.go", 14}},
		want:  []string{"kernels.go:12", "kernels.go:14"},
	}, {
		name:  "base names collide",
		sites: []Site{{"pkg/gate/kernels.go", 12}, {"pkg/norm/kernels.go", 12}},
		want:  []string{"gate/kernels.go:12", "norm/kernels.go:12"},
	}, {
		name:  "collision needs two elements",
		sites: []Site{{"x/gate/op/kernels.go", 1}, {"y/gate/op/kernels.go", 1}, {"z/main.go", 1}},
		want:  []string{"x/gate/op/kernels.go:1", "y/gate/op/kernels.go:1", "main.go:1"},
	}, {
		name:  "only the colliding file grows",
		sites: []Site{{"a/kernels.go", 1}, {"b/kernels.go", 1}, {"c/main.go", 2}},
		want:  []string{"a/kernels.go:1", "b/kernels.go:1", "main.go:2"},
	}, {
		name:  "path written on another system",
		sites: []Site{{`C:\gate\kernels.go`, 3}, {`C:\norm\kernels.go`, 3}},
		want:  []string{"gate/kernels.go:3", "norm/kernels.go:3"},
	}} {
		t.Run(tt.name, func(t *testing.T) {
			names := SiteNames(tt.sites)
			for i, s := range tt.sites {
				if got := names[s]; got != tt.want[i] {
					t.Errorf("SiteNames[%v] = %q, want %q", s, got, tt.want[i])
				}
			}
		})
	}
}

// TestSiteNamesAreUniquePerSite is the property the reports rely on: a row
// or a graph node names one site and no other.
func TestSiteNamesAreUniquePerSite(t *testing.T) {
	sites := []Site{{"a/kernels.go", 1}, {"b/kernels.go", 1}, {"a/kernels.go", 2}}
	names := SiteNames(sites)
	seen := map[string]Site{}
	for _, s := range sites {
		if other, ok := seen[names[s]]; ok {
			t.Errorf("%v and %v both print as %q", other, s, names[s])
		}
		seen[names[s]] = s
	}
}

// TestReportsDistinguishCollidingFiles checks the writers use the names
// rather than the base name a site prints on its own. Two helpers in
// different packages with the same file name are one row apiece.
func TestReportsDistinguishCollidingFiles(t *testing.T) {
	a := &Attribution{
		Unit: Unit{Name: "fused"},
		SASS: map[Site][]cudasass.Instruction{
			{"gate/kernels.go", 7}: {{Opcode: "FMUL"}},
			{"norm/kernels.go", 7}: {{Opcode: "FADD"}},
		},
	}
	var buf bytes.Buffer
	if err := WriteTable(&buf, a); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"gate/kernels.go:7", "norm/kernels.go:7"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("table does not name %s:\n%s", want, &buf)
		}
	}
	edges := a.Edges()
	if len(edges) != 2 {
		t.Fatalf("Edges() = %v, want one edge per site to the kernel", edges)
	}
	if edges[0].From == edges[1].From {
		t.Errorf("both sites are the same graph node %q", edges[0].From)
	}
}
