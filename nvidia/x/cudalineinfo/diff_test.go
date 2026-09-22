package cudalineinfo

import (
	"bytes"
	"strings"
	"testing"
)

// TestDiffFastMath compares the fixture with its -use_fast_math build. Fast
// math replaces the IEEE division with an approximate reciprocal, so the Silu
// line must lose its slowpath call and the slowpath itself must be gone from
// the remainder.
func TestDiffFastMath(t *testing.T) {
	base, fast := loadFixture(t, "fused_gate", ""), loadFixture(t, "fused_gate", "-fast")
	diffs := Diff(base, fast)
	if len(diffs) != 4 {
		t.Fatalf("got %d site diffs, want 4", len(diffs))
	}
	silu := diffs[2]
	if silu.Site != (Site{kernelsFile, 14}) || silu.Base != 23 || silu.Target != 5 {
		t.Errorf("silu diff = %+v, want kernels.go:14 23 -> 5", silu)
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
	if len(fast.Remainder.Subroutines) != 0 {
		t.Errorf("fast build still has subroutines: %v", fast.Remainder.Subroutines)
	}
	if diffs[0].Base != diffs[0].Target || len(diffs[0].Removed) != 0 {
		t.Errorf("input load line changed under fast math: %+v", diffs[0])
	}

	var buf bytes.Buffer
	if err := WriteDiff(&buf, base, fast, "ieee", "fast"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "kernels.go:14") || !strings.Contains(buf.String(), "CALL") {
		t.Errorf("diff output lacks the silu row:\n%s", buf.String())
	}
}

func TestDiffSiteOnOneSide(t *testing.T) {
	base := Attribute(fixtureUnit(t, "fused_gate"), "", nil)
	target := loadFixture(t, "fused_gate", "")
	for _, d := range Diff(base, target) {
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
