package cudacensus

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// testdata/ds4-weight-bytes.txt is documentation, not code, so nothing executes
// it and every number in it is one typo away from being wrong in a way that
// still reads plausibly. These tests check the file against itself: the three
// summaries it states (by family, by residency, by position regime) are all
// derivable from its itemized lines, so a mistake in any one of them shows up
// as a disagreement rather than as a number nobody can check.
//
// This is the same failure the package exists to make visible. A byte count
// cannot be validated by arithmetic once it reaches a Coster; the only place it
// can be validated is here, against the shapes it was computed from.

var (
	ds4Item   = regexp.MustCompile(`^\s*([\d,]+)\s+\d+\.\d+%\s+(\S+)\s+(.*)$`)
	ds4Family = regexp.MustCompile(`^  (Q8_0|Q2_K|IQ2_XXS|F16|F32|mixed)\s+([\d,]+)\s+\d+\.\d+%$`)
	ds4Total  = regexp.MustCompile(`^([\d,]+)\s+TOTAL`)
	ds4Resid  = regexp.MustCompile(`^  (host page table|device)\s+([\d,]+)\s`)
)

func ds4Num(t *testing.T, s string) uint64 {
	t.Helper()
	n, err := strconv.ParseUint(strings.ReplaceAll(s, ",", ""), 10, 64)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return n
}

type ds4Budget struct {
	items    map[string]uint64 // tensor description -> bytes
	family   map[string]uint64 // as itemized
	stated   map[string]uint64 // as summarized in the "By family" block
	residenc map[string]uint64
	total    uint64
}

func readDS4Budget(t *testing.T) ds4Budget {
	t.Helper()
	b, err := os.ReadFile("testdata/ds4-weight-bytes.txt")
	if err != nil {
		t.Fatalf("read budget: %v", err)
	}
	got := ds4Budget{
		items:    map[string]uint64{},
		family:   map[string]uint64{},
		stated:   map[string]uint64{},
		residenc: map[string]uint64{},
	}
	for _, line := range strings.Split(string(b), "\n") {
		switch {
		case ds4Total.MatchString(line):
			got.total = ds4Num(t, ds4Total.FindStringSubmatch(line)[1])
		case ds4Family.MatchString(line):
			m := ds4Family.FindStringSubmatch(line)
			got.stated[m[1]] = ds4Num(t, m[2])
		case ds4Resid.MatchString(line):
			m := ds4Resid.FindStringSubmatch(line)
			got.residenc[m[1]] = ds4Num(t, m[2])
		case ds4Item.MatchString(line):
			m := ds4Item.FindStringSubmatch(line)
			n := ds4Num(t, m[1])
			got.items[m[3]] = n
			got.family[m[2]] += n
		}
	}
	if got.total == 0 || len(got.items) == 0 {
		t.Fatalf("parsed nothing useful: total=%d items=%d", got.total, len(got.items))
	}
	return got
}

// TestDS4ItemizedLinesSumToTotal checks the stated TOTAL against the lines it
// claims to total.
func TestDS4ItemizedLinesSumToTotal(t *testing.T) {
	got := readDS4Budget(t)
	var sum uint64
	for _, n := range got.items {
		sum += n
	}
	if sum != got.total {
		t.Errorf("itemized lines sum to %d, file states TOTAL %d (off by %d)", sum, got.total, int64(sum)-int64(got.total))
	}
}

// TestDS4FamilySummaryMatchesLines checks the "By family" block against the
// lines it summarizes. A family total that drifts from its own lines is the
// shape that put 9.5599 and 9.601 in circulation as rivals.
func TestDS4FamilySummaryMatchesLines(t *testing.T) {
	got := readDS4Budget(t)
	if len(got.stated) == 0 {
		t.Fatal("no By-family block parsed")
	}
	for fam, want := range got.stated {
		if got.family[fam] != want {
			t.Errorf("family %s: lines sum to %d, block states %d", fam, got.family[fam], want)
		}
	}
	for fam := range got.family {
		if _, ok := got.stated[fam]; !ok {
			t.Errorf("family %s appears in the lines but not in the By-family block", fam)
		}
	}
}

// TestDS4ResidencyPartitionIsNotTheWeightSplit is the assertion the file's own
// prose argues for. The host-page-table figure must be TOTAL minus the single
// device-resident line, and it must NOT equal the four weight families --
// because those two populations differ by the norms/sinks/bias and mixed lines,
// and a model that reaches for "weights" when it means "host-page-table
// resident" is right only by coincidence of size.
func TestDS4ResidencyPartitionIsNotTheWeightSplit(t *testing.T) {
	got := readDS4Budget(t)
	hpt, device := got.residenc["host page table"], got.residenc["device"]
	if hpt == 0 || device == 0 {
		t.Fatalf("residency block not parsed: %v", got.residenc)
	}
	if hpt+device != got.total {
		t.Errorf("residency partition sums to %d, TOTAL is %d", hpt+device, got.total)
	}

	var kv uint64
	for desc, n := range got.items {
		if strings.Contains(desc, "KV/compressed-cache") {
			kv = n
		}
	}
	if kv == 0 {
		t.Fatal("no KV/compressed-cache line found; the residency exception is unstated")
	}
	if device != kv {
		t.Errorf("device-resident figure %d is not the KV line %d", device, kv)
	}

	weights := got.family["Q8_0"] + got.family["Q2_K"] + got.family["IQ2_XXS"] + got.family["F16"]
	if weights == hpt {
		t.Errorf("weights (%d) equal host-page-table bytes (%d); the file's central "+
			"distinction has been flattened -- a non-family mmap'd line was dropped", weights, hpt)
	}
	// They are close enough to be mistaken for each other, which is the point.
	if d := int64(hpt) - int64(weights); d <= 0 || float64(d)/float64(hpt) > 0.01 {
		t.Errorf("host-page-table bytes exceed weights by %d (%.3f%%); expected a small "+
			"positive gap (the norms/sinks/bias and mixed lines)", d, 100*float64(d)/float64(hpt))
	}
}

// TestDS4IndexerRegimeIsStatedForEveryTotal guards the caveat that cost the
// campaign a day: 9,559,926,784 and 9,601,439,048 are both indexer-ACTIVE, and
// each has an inactive counterpart 3.8% lower. Any of the four quoted without
// its position regime is one of the other three.
func TestDS4IndexerRegimeIsStatedForEveryTotal(t *testing.T) {
	b, err := os.ReadFile("testdata/ds4-weight-bytes.txt")
	if err != nil {
		t.Fatalf("read budget: %v", err)
	}
	got := readDS4Budget(t)

	var indexer uint64
	for desc, n := range got.items {
		if strings.Contains(desc, "indexer.attn_q_b") || strings.Contains(desc, "indexer.proj") {
			indexer += n
		}
	}
	if indexer == 0 {
		t.Fatal("no indexer lines found")
	}
	weights := got.family["Q8_0"] + got.family["Q2_K"] + got.family["IQ2_XXS"] + got.family["F16"]

	// Every one of the four totals must appear in the file with its regime, so a
	// reader cannot pick one up without the qualifier.
	for _, want := range []uint64{weights, got.total, weights - indexer, got.total - indexer} {
		s := ds4Commas(want)
		if !strings.Contains(string(b), s) {
			t.Errorf("total %s is derivable from the lines but never stated; a reader "+
				"who needs it will recompute it and lose the regime qualifier", s)
		}
	}
}

func ds4Commas(n uint64) string {
	s := strconv.FormatUint(n, 10)
	var out []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	return string(out)
}
