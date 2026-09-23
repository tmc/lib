package cudalineinfo

import (
	"bytes"
	"strings"
	"testing"
)

// TestDiffFastMath compares the fixture with its -use_fast_math build. Fast
// math replaces the IEEE division with an approximate reciprocal, so the Silu
// line must lose its slowpath call and the slowpath itself must be gone from
// the remainder. The loop's 64-bit integer division is not floating point
// and stays.
func TestDiffFastMath(t *testing.T) {
	base, fast := loadFixture(t, "fused_gate", ""), loadFixture(t, "fused_gate", "-fast")
	diffs := Diff(Side{Attribution: base}, Side{Attribution: fast})
	if len(diffs) != 4 {
		t.Fatalf("got %d site diffs, want 4", len(diffs))
	}
	silu := diffs[2]
	if silu.Site != (Site{kernelsFile, 14}) || silu.Base != 117 || silu.Target != 25 {
		t.Errorf("silu diff = %+v, want kernels.go:14 117 -> 25", silu)
	}
	// Opcodes compare by base mnemonic, so the slowpath call is CALL and
	// the range check FCHK. Both builds keep a MUFU exponential and
	// reciprocal, so MUFU is on neither side.
	removed := " " + strings.Join(silu.Removed, " ") + " "
	for _, op := range []string{" CALL ", " FCHK ", " BSSY "} {
		if !strings.Contains(removed, op) {
			t.Errorf("fast math kept %s on the silu line; removed %s", strings.TrimSpace(op), removed)
		}
	}
	if strings.Contains(removed, " MUFU ") {
		t.Errorf("fast math lost a MUFU on the silu line; removed %s", removed)
	}
	for name := range fast.Remainder.Subroutines {
		if strings.HasSuffix(name, divSlowpath) {
			t.Errorf("fast build still has %s", name)
		}
	}
	// The input line keeps its instructions. Scheduling may still move a
	// convergence barrier or a constant load across it, so only the
	// arithmetic is required to be untouched.
	x := diffs[0]
	if x.Base != x.Target {
		t.Errorf("input load line changed size under fast math: %+v", x)
	}
	for _, op := range append(x.Removed, x.Added...) {
		if strings.HasPrefix(op, "F") || op == "MUFU" {
			t.Errorf("input load line changed arithmetic under fast math: %+v", x)
			break
		}
	}

	var buf bytes.Buffer
	if err := WriteDiff(&buf, Side{Attribution: base, Label: "ieee"}, Side{Attribution: fast, Label: "fast"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "kernels.go:14") || !strings.Contains(buf.String(), "CALL") {
		t.Errorf("diff output lacks the silu row:\n%s", buf.String())
	}
}

// TestDiffProfiles gives each side a profile: the measured column is what
// says whether the instructions a line lost were instructions the device was
// spending its time on, and a side that was not run reads as unmeasured
// rather than as zero.
func TestDiffProfiles(t *testing.T) {
	base, fast := loadFixture(t, "fused_gate", ""), loadFixture(t, "fused_gate", "-fast")
	silu := Site{kernelsFile, 14}
	baseSide := Side{Attribution: base, Label: "ieee", Profile: &Profile{
		Measure: "pc_samples",
		Sites:   map[Site]int64{silu: 900},
	}}
	fastSide := Side{Attribution: fast, Label: "fast", Profile: &Profile{
		Measure: "pc_samples",
		Sites:   map[Site]int64{silu: 120},
	}}
	for _, d := range Diff(baseSide, fastSide) {
		if d.Site != silu {
			continue
		}
		if d.BaseValue != 900 || d.TargetValue != 120 {
			t.Errorf("silu measured %d -> %d, want 900 -> 120", d.BaseValue, d.TargetValue)
		}
	}
	var buf bytes.Buffer
	if err := WriteDiff(&buf, baseSide, fastSide); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "IEEE PC_SAMPLES") || !strings.Contains(buf.String(), "900") {
		t.Errorf("no measured column:\n%s", buf.String())
	}
	// One side run and the other not: the column is there, and the side
	// that was not run says so.
	buf.Reset()
	if err := WriteDiff(&buf, baseSide, Side{Attribution: fast, Label: "fast"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "900") || !strings.Contains(buf.String(), "-") {
		t.Errorf("an unmeasured side did not read as unmeasured:\n%s", buf.String())
	}
	// Neither side run: no column at all.
	buf.Reset()
	if err := WriteDiff(&buf, Side{Attribution: base, Label: "ieee"}, Side{Attribution: fast, Label: "fast"}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(buf.String(), "PC_SAMPLES") {
		t.Errorf("a measured column with nothing measured:\n%s", buf.String())
	}
}

func TestDiffSiteOnOneSide(t *testing.T) {
	base := Attribute(fixtureUnit(t, "fused_gate"), "", nil)
	target := loadFixture(t, "fused_gate", "")
	for _, d := range Diff(Side{Attribution: base}, Side{Attribution: target}) {
		if d.Base != 0 || len(d.Removed) != 0 {
			t.Errorf("site %v has base instructions in an attribution without SASS: %+v", d.Site, d)
		}
		if d.Target != len(target.SASS[d.Site]) {
			t.Errorf("site %v target = %d, want %d", d.Site, d.Target, len(target.SASS[d.Site]))
		}
	}
}

func TestSummarize(t *testing.T) {
	for _, tt := range []struct {
		ops  []string
		want string
	}{
		{nil, "-"},
		{[]string{"FFMA"}, "FFMA"},
		{[]string{"FFMA", "FFMA", "FFMA", "MUFU.EX2"}, "FFMA×3 MUFU.EX2"},
	} {
		if got := summarize(tt.ops); got != tt.want {
			t.Errorf("summarize(%v) = %q, want %q", tt.ops, got, tt.want)
		}
	}
}
