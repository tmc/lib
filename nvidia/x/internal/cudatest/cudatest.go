// Package cudatest resolves the CUDA toolchain a native test gate builds with.
//
// The native gates in this tree are opt-in: they skip unless an environment
// variable enables them, and fail loudly once enabled. A gate that compiles
// must record which compiler it used, or a failure caused by the toolchain is
// indistinguishable from a broken fixture.
package cudatest

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// versionTimeout bounds the one execution this package performs. A toolkit
// binary that cannot answer --version in that long is not going to compile.
const versionTimeout = 30 * time.Second

// NVCC returns the nvcc a native gate compiles with and the toolkit version it
// reports, logging both so the artifact the gate qualifies carries the
// toolchain that produced it. The version is the zero [cudatool.Version] if
// nvcc named none, which is logged rather than failed: it is missing
// provenance, not a broken compiler.
//
// Resolution is [cudatool.LookTool]'s, which prefers a toolkit install over
// PATH; see that package's documentation for why PATH alone is the wrong
// source. A caller has already decided it must compile, so an nvcc that cannot
// be resolved, or cannot be run, fails the test.
func NVCC(t *testing.T) (string, cudatool.Version) {
	t.Helper()
	nvcc, err := cudatool.LookTool("nvcc", "")
	if err != nil {
		t.Fatalf("qualification enabled but nvcc unavailable: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), versionTimeout)
	defer cancel()
	out, err := cudatool.Run(ctx, nvcc, "", nil, "--version")
	if err != nil {
		t.Fatalf("nvcc --version: %v", err)
	}
	v, ok := cudatool.ParseVersion(append(out.Stdout, out.Stderr...))
	if !ok {
		t.Logf("nvcc %s reports no toolkit version:\n%s", nvcc, out.Stdout)
		return nvcc, v
	}
	t.Logf("nvcc %s (CUDA %s)", nvcc, v)
	return nvcc, v
}
