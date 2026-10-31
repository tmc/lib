package numopt

import (
	"math"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func TestPlanKLoop(t *testing.T) {
	tiles := outlierLayer()
	_, frontier, err := SolveParetoFrontier(tiles, "sm_100a", math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}
	p := frontier[len(frontier)/2]
	groups, err := PlanKLoop(tiles, p.Assignments, "sm_100a")
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckGroups(groups, len(tiles)); err != nil {
		t.Fatal(err)
	}
	for _, g := range groups {
		for _, k := range g.Tiles {
			if f := p.Assignments[tiles[k].TileID]; f != g.Format {
				t.Errorf("tile %d is %v but issued in the %v group", k, f, g.Format)
			}
		}
	}
	if _, err := PlanKLoop(tiles, map[string]TilePrecisionFormat{}, "sm_100a"); err == nil {
		t.Error("PlanKLoop accepted an empty assignment")
	}
}

func TestCheckGroupsMutants(t *testing.T) {
	good := []KGroup{{PrecMXFP8E4M3, []int{1, 3}}, {PrecNVFP4ScaleVec4X, []int{0, 2}}}
	if err := CheckGroups(good, 4); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		groups []KGroup
	}{
		{"tile dropped", []KGroup{{PrecMXFP8E4M3, []int{1, 3}}, {PrecNVFP4ScaleVec4X, []int{0}}}},
		{"tile issued twice", []KGroup{{PrecMXFP8E4M3, []int{1, 3, 0}}, {PrecNVFP4ScaleVec4X, []int{0, 2}}}},
		{"tile out of range", []KGroup{{PrecMXFP8E4M3, []int{1, 3}}, {PrecNVFP4ScaleVec4X, []int{0, 2, 4}}}},
		{"format split", []KGroup{{PrecMXFP8E4M3, []int{1}}, {PrecNVFP4ScaleVec4X, []int{0, 2}}, {PrecMXFP8E4M3, []int{3}}}},
		{"empty group", append(good, KGroup{PrecBF16, nil})},
	}
	for _, tt := range tests {
		if err := CheckGroups(tt.groups, 4); err == nil {
			t.Errorf("%s: accepted", tt.name)
		}
	}
}

var asmRE = regexp.MustCompile(`setp\.ne\.b32 p, %(\d+), 0;\\n"\s*"([^"]*)\\n}\\n"\s*:: ([^:]*), "r"\(issued\)`)

func TestEmitKLoop(t *testing.T) {
	groups := []KGroup{
		{PrecBF16, []int{4}},
		{PrecMXFP8E4M3, []int{0, 5}},
		{PrecNVFP4ScaleVec4X, []int{1}},
		{PrecRubin3BitLUT, []int{2}},
		{PrecSparse2To4NVFP4, []int{3}},
	}
	src, err := EmitKLoop("eg_mixed", "sm_107a", groups, 6)
	if err != nil {
		t.Fatal(err)
	}
	m := asmRE.FindAllStringSubmatch(src, -1)
	if len(m) != len(groups) {
		t.Fatalf("found %d MMA statements, want %d:\n%s", len(m), len(groups), src)
	}
	kinds := []string{"kind::f16", "kind::mxf8f6f4.block_scale.scale_vec::1X", "kind::mxf4nvf4.block_scale.scale_vec::4X",
		"decompress::lut::b.block32", "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4"}
	for i, s := range m {
		if !strings.Contains(s[2], kinds[i]) {
			t.Errorf("group %d issues %q, want %s", i, s[2], kinds[i])
		}
		// The predicate operand must be the one after the instruction's own
		// operands, or enable_input_d reads the wrong register.
		pred, _ := strconv.Atoi(s[1])
		if n := strings.Count(s[3], `"(`); pred != n {
			t.Errorf("group %d: predicate is %%%d, but %d operands precede it", i, pred, n)
		}
		if maxRef := highestOperand(s[2]); maxRef >= pred {
			t.Errorf("group %d: instruction references %%%d, not below the predicate %%%d", i, maxRef, pred)
		}
	}
	if !strings.Contains(src, "__device__ const int eg_mixed_g1[2] = {0, 5};") {
		t.Errorf("group table missing:\n%s", src)
	}
	if strings.Count(src, "++issued;") != len(groups) {
		t.Error("accumulate counter not advanced once per MMA")
	}

	for _, tt := range []struct {
		name, arch string
		groups     []KGroup
	}{
		{"lut on sm_100a", "sm_100a", []KGroup{{PrecRubin3BitLUT, []int{0}}}},
		{"ue5m3 has no spelling", "sm_107a", []KGroup{{PrecRubinUE5M3, []int{0}}}},
		{"sm_121a", "sm_121a", []KGroup{{PrecBF16, []int{0}}}},
		{"bad groups", "sm_100a", []KGroup{{PrecBF16, []int{1}}}},
	} {
		if _, err := EmitKLoop("x", tt.arch, tt.groups, 1); err == nil {
			t.Errorf("%s: accepted", tt.name)
		}
	}
}

func highestOperand(instr string) int {
	n := -1
	for _, s := range regexp.MustCompile(`%(\d+)`).FindAllStringSubmatch(instr, -1) {
		v, _ := strconv.Atoi(s[1])
		n = max(n, v)
	}
	return n
}
