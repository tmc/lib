package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestNonFiniteChecksReportExtent asserts that every non-finite check over a
// logits vector in this package goes through nonFiniteExtent.
//
// The property it guards is that a diagnostic must distinguish a wholly
// poisoned vector from a single bad element at the front. Those are different
// bugs -- memory-pressure corruption against a layout or offset fault -- and a
// checker that returns at the first non-finite value prints the same thing for
// both. Thirteen such checks were written independently in this package before
// anyone noticed, and a fourteenth in a darwin-only file was found only because
// an unrelated build error forced a second look.
//
// That history is why this test reads the directory rather than the package.
// A grep's denominator is the build configuration it ran under: a scan that
// sees only the files the current GOOS compiles reports the package clean and
// is silent about the other half. os.ReadDir has no build tags, so every file
// is in the denominator whatever this test is built for.
func TestNonFiniteChecksReportExtent(t *testing.T) {
	// The old spellings. They are a finite set because there is now exactly
	// one message format; a new one appearing means a new copy of the check.
	old := regexp.MustCompile(`non-?finite logit ?\[?%d|logit %d = %g`)

	// isNaN/isInf pairs are legitimate outside a logits vector -- scores,
	// expert outputs, tolerances. The defect is specifically a per-element
	// scan over something vocabulary-sized that reports a position.
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	scanned := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		scanned++
		if name == filepath.Base("logit_extent_test.go") {
			continue // holds the patterns themselves
		}
		for i, line := range strings.Split(string(b), "\n") {
			if old.MatchString(line) {
				t.Errorf("%s:%d reports a non-finite position without its extent:\n\t%s\n"+
					"use nonFiniteExtent, whose format distinguishes one bad element from all of them",
					name, i+1, strings.TrimSpace(line))
			}
		}
	}
	// A scan that matched nothing because it read nothing is the failure this
	// whole test exists to prevent, so it states its own denominator.
	if scanned < 100 {
		t.Fatalf("scanned %d .go files, expected the whole package; the scan's "+
			"denominator is wrong and its silence means nothing", scanned)
	}
	t.Logf("scanned %d .go files under GOOS-independent directory read", scanned)
}
