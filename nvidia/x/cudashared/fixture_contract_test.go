package cudashared_test

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashared"
)

func copyFixturePlan(t *testing.T, requests ...cudashared.Request) *cudashared.Plan {
	t.Helper()
	// Deliberately larger than the device record: the source consumer must
	// check that record rather than trusting this caller-supplied budget.
	p, err := cudashared.NewPlan(cudashared.Limits{StaticMax: 200000, DynamicMax: 200000, TotalMax: 200000, Source: "contract test"})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range requests {
		if _, err := p.Add(r); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

func TestStagedCopySourceResources(t *testing.T) {
	limits, err := cudashared.LimitsFromDevice("sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, target string
		elements     int
		wantError    string
	}{
		{"one", "sm_121a", 1, ""},
		{"device boundary", "sm_121a", int(limits.DynamicMax / 4), ""},
		{"beyond device", "sm_121a", int(limits.DynamicMax/4) + 1, "exceeds target"},
		{"overflow", "sm_121a", math.MaxInt, "positive and bounded"},
		{"empty", "sm_121a", 0, "positive and bounded"},
		{"negative", "sm_121a", -1, "positive and bounded"},
		{"unprobed", "sm_100a", 1, "target"},
		{"unknown", "sm_999a", 1, "unsupported target"},
		{"injection", "sm_121a\n#error", 1, "unsupported target"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			src, err := cudashared.StagedCopyFixtureSource(tt.target, tt.elements)
			if tt.wantError != "" {
				if err == nil || src != "" || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("source = %q, error = %v; want %q", src, err, tt.wantError)
				}
				return
			}
			if err != nil || !strings.Contains(src, "__align__(4)") {
				t.Fatalf("source lacks float32 base alignment: %q, %v", src, err)
			}
		})
	}
}

func TestStagedCopyPlanResources(t *testing.T) {
	limits, err := cudashared.LimitsFromDevice("sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, target string
		requests     []cudashared.Request
		wantError    string
	}{
		{"dynamic limit", "sm_121a", []cudashared.Request{{Name: "stage0", Size: limits.DynamicMax + 1, Alignment: 16, Dynamic: true}}, "exceeds target"},
		{"static limit", "sm_121a", []cudashared.Request{{Name: "static", Size: limits.StaticMax + 1, Alignment: 16}, {Name: "stage0", Size: 4, Alignment: 4, Dynamic: true}}, "exceeds target"},
		{"total limit", "sm_121a", []cudashared.Request{{Name: "static", Size: limits.StaticMax, Alignment: 16}, {Name: "stage0", Size: limits.DynamicMax, Alignment: 16, Dynamic: true}}, "exceeds target"},
		{"static pool", "sm_121a", []cudashared.Request{{Name: "static", Size: 16, Alignment: 16}, {Name: "stage0", Size: 4, Alignment: 4, Dynamic: true}}, "only the dynamic"},
		{"unprobed", "sm_100a", []cudashared.Request{{Name: "stage0", Size: 4, Alignment: 4, Dynamic: true}}, "target"},
		{"unknown", "sm_999a", []cudashared.Request{{Name: "stage0", Size: 4, Alignment: 4, Dynamic: true}}, "unsupported target"},
		{"missing stage", "sm_121a", nil, "lacks dynamic stage0"},
		{"static stage", "sm_121a", []cudashared.Request{{Name: "stage0", Size: 4, Alignment: 4}}, "only the dynamic"},
		{"small stage", "sm_121a", []cudashared.Request{{Name: "stage0", Size: 3, Alignment: 4, Dynamic: true}}, "too small"},
		{"unaligned stage", "sm_121a", []cudashared.Request{{Name: "prefix", Size: 1, Alignment: 1, Dynamic: true}, {Name: "stage0", Size: 4, Alignment: 1, Dynamic: true}}, "not aligned"},
		{"stage alignment", "sm_121a", []cudashared.Request{{Name: "stage0", Size: 4, Alignment: 256, Dynamic: true}}, "alignment"},
		{"later alignment", "sm_121a", []cudashared.Request{{Name: "stage0", Size: 4, Alignment: 4, Dynamic: true}, {Name: "later", Size: 1, Alignment: 256, FirstStage: 2, LastStage: 2, Dynamic: true}}, "alignment"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := copyFixturePlan(t, tt.requests...)
			src, launchBytes, err := cudashared.StagedCopyFixture(p, tt.target, 1)
			if err == nil || src != "" || launchBytes != 0 || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("source = %q, bytes = %d, error = %v; want %q", src, launchBytes, err, tt.wantError)
			}
		})
	}
	if src, n, err := cudashared.StagedCopyFixture(nil, "sm_121a", 1); err == nil || src != "" || n != 0 {
		t.Fatal("nil plan accepted")
	}
	p := copyFixturePlan(t, cudashared.Request{Name: "stage0", Size: 4, Alignment: 4, Dynamic: true})
	for _, elements := range []int{-1, 0, 2, math.MaxInt} {
		if src, n, err := cudashared.StagedCopyFixture(p, "sm_121a", elements); err == nil || src != "" || n != 0 {
			t.Fatalf("invalid element extent %d accepted", elements)
		}
	}
}

func TestStagedCopyPlanAlignmentAndTail(t *testing.T) {
	for _, alignment := range []int64{4, 16, 32, 64, 128} {
		t.Run(fmt.Sprint(alignment), func(t *testing.T) {
			p := copyFixturePlan(t,
				cudashared.Request{Name: "prefix", Size: 17, Alignment: 4, Dynamic: true},
				cudashared.Request{Name: "stage0", Size: 65 * 4, Alignment: 4, Dynamic: true},
				// This follows stage0 in the description and reuses memory at
				// a later stage. Its alignment still constrains the common base.
				cudashared.Request{Name: "later", Size: 1, Alignment: alignment, FirstStage: 2, LastStage: 2, Dynamic: true})
			src, launchBytes, err := cudashared.StagedCopyFixture(p, "sm_121a", 65)
			if err != nil {
				t.Fatal(err)
			}
			constant := func(pattern string) int {
				t.Helper()
				match := regexp.MustCompile(pattern).FindStringSubmatch(src)
				if len(match) != 2 {
					t.Fatal("missing generated index", pattern)
				}
				n, err := strconv.Atoi(match[1])
				if err != nil {
					t.Fatal(err)
				}
				return n
			}
			baseAlignment := constant(`__align__\((\d+)\)`)
			offset := constant(`raw \+ (\d+)`)
			bound := constant(`if \(i < (\d+)\)`)
			if int64(baseAlignment) != alignment || offset != 20 || bound != 65 || launchBytes != p.DynamicBytes() {
				t.Fatalf("alignment=%d offset=%d bound=%d bytes=%d", baseAlignment, offset, bound, launchBytes)
			}
			if !strings.Contains(src, "tile[threadIdx.x] = in[i];") || !strings.Contains(src, "out[i] = tile[threadIdx.x];") || strings.Count(src, "if (i < 65)") != 2 {
				t.Fatal("producer or consumer indexing/guards changed")
			}
			producer, barrier, consumer := strings.Index(src, "= in[i]"), strings.Index(src, "__syncthreads();"), strings.Index(src, "out[i] =")
			if producer < 0 || barrier <= producer || consumer <= barrier {
				t.Fatal("missing uniform producer/barrier/consumer order")
			}
			// Interpret the emitted bounded indices for three CTAs, including
			// a one-element final tail. Poison every untouched shared slot.
			out := make([]int, 65)
			for block := 0; block < 3; block++ {
				shared := make([]int, int(launchBytes)/4)
				for i := range shared {
					shared[i] = -1
				}
				for thread := 0; thread < 32; thread++ {
					i := block*32 + thread
					if i < bound {
						shared[offset/4+thread] = i*13 + 7
					}
				}
				for thread := 0; thread < 32; thread++ {
					i := block*32 + thread
					if i < bound {
						out[i] = shared[offset/4+thread]
					}
				}
			}
			for i, got := range out {
				if want := i*13 + 7; got != want {
					t.Fatalf("tail output[%d] = %d, want %d", i, got, want)
				}
			}
		})
	}
}
