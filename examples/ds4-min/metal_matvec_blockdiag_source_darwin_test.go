//go:build darwin

package main

import (
	"strings"
	"testing"
)

// TestMatvecQ80BlockDiagIsMatvecQ80WithBasePointers keeps the two kernels in
// lockstep at the source level.
//
// matvec_q8_0_blockdiag's whole claim is that it is matvec_q8_0 with three base
// pointers moved and no other change, which is what makes the fused dispatch bit
// identical to the eight it replaces rather than merely close. That claim is
// worth exactly as much as whatever enforces it, and until this test the only
// thing enforcing it was a sentence in a comment asking the next editor to keep
// them in step.
//
// The risk is specific and it is not hypothetical: 094936CE's first draft of the
// CUDA equivalent retyped one accumulation line as `d * dot * x.d` where the
// original reads `d * x.d * dot`. Float multiplication is not associative, so
// those disagree in about 35% of bit patterns. A reviewer reading for
// correctness would pass it.
//
// So: recover matvec_q8_0's body from matvec_q8_0_blockdiag's by undoing the
// three substitutions and dropping the four added lines, then require exact
// equality. Any edit to either kernel that is not one of those shows up here as a
// diff, including an edit that is individually correct but applied to only one of
// the two.
func TestMatvecQ80BlockDiagIsMatvecQ80WithBasePointers(t *testing.T) {
	plain := metalKernelBody(t, "matvec_q8_0")
	blockDiag := metalKernelBody(t, "matvec_q8_0_blockdiag")

	// The four lines the block-diagonal form adds: the group index and the three
	// bases derived from it.
	added := []string{
		"  const uint diag = group.y;",
		"  device const uchar *w = weights+uint64_t(diag)*uint64_t(rows)*blocks*34;",
		"  device const float *xb = x+uint64_t(diag)*cols;",
		"  device float *ob = out+uint64_t(diag)*rows;",
	}
	// The three places an existing line reads through one of those bases instead
	// of the buffer argument. Nothing else may differ.
	rebased := strings.NewReplacer(
		"kernel void matvec_q8_0_blockdiag(", "kernel void matvec_q8_0(",
		"xv[i]=xb[", "xv[i]=x[",
		"const device uchar *p=w+", "const device uchar *p=weights+",
		"ob[first_row+r]=total", "out[first_row+r]=total",
	)

	var recovered []string
	for _, line := range strings.Split(blockDiag, "\n") {
		if slicesContains(added, line) {
			continue
		}
		recovered = append(recovered, rebased.Replace(line))
	}
	got := strings.Join(recovered, "\n")
	if got != plain {
		t.Errorf("matvec_q8_0_blockdiag is no longer matvec_q8_0 with moved base pointers.\n"+
			"Recovered body:\n%s\n\nmatvec_q8_0 body:\n%s\n\n"+
			"If the change was intended, make it in BOTH kernels and update the substitutions above. "+
			"If it touched the accumulation order, the fusion is no longer bit identical and the "+
			"runtime swap needs the logits anchor rather than this gate.", got, plain)
	}
}

func slicesContains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

// metalKernelBody returns the source of one kernel in metalPinnedMSL, from its
// "kernel void <name>(" line through the closing brace in column zero.
func metalKernelBody(t *testing.T, name string) string {
	t.Helper()
	start := strings.Index(metalPinnedMSL, "kernel void "+name+"(")
	if start < 0 {
		t.Fatalf("kernel %q not found in metalPinnedMSL", name)
	}
	end := strings.Index(metalPinnedMSL[start:], "\n}\n")
	if end < 0 {
		t.Fatalf("kernel %q has no closing brace", name)
	}
	return metalPinnedMSL[start : start+end+2]
}
