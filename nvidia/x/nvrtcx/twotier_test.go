package nvrtcx_test

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestNVRTCFlagSeparation(t *testing.T) {
	// 1. Check individual flag classification
	offlineFlags := []string{
		"-maxrregcount",
		"--maxrregcount",
		"-maxrregcount=64",
		"--maxrregcount=32",
		"-Xptxas",
		"-Xptxas=-v",
		"-Xptxas,-v",
		"-O0",
		"-O1",
		"-O2",
		"-O3",
		"-O4",
		"-Xcudafe",
		"-Xnvlink",
	}
	for _, f := range offlineFlags {
		if !nvrtcx.IsOfflineFlag(f) {
			t.Errorf("IsOfflineFlag(%q) = false, want true", f)
		}
	}

	sourceFlags := []string{
		"-std=c++20",
		"--std=c++20",
		"-I/usr/local/cuda/include",
		"--include-path=/usr/local/cuda/include",
		"-DFEATURE=1",
		"-UDEBUG",
		"--gpu-architecture=sm_121a",
		"--extra-device-vectorization",
		"--device-as-default-execution-space",
		"--use_fast_math",
		"-pch",
		"--create-pch=test.pch",
		"--use-pch=test.pch",
	}
	for _, f := range sourceFlags {
		if nvrtcx.IsOfflineFlag(f) {
			t.Errorf("IsOfflineFlag(%q) = true, want false (source-level flag)", f)
		}
	}

	// 2. Test ValidateNVRTCFlags
	if err := nvrtcx.ValidateNVRTCFlags(sourceFlags); err != nil {
		t.Errorf("ValidateNVRTCFlags returned error on valid source flags: %v", err)
	}
	mixedFlags := append([]string(nil), sourceFlags...)
	mixedFlags = append(mixedFlags, "-maxrregcount=64", "-Xptxas=-v", "-O3")
	if err := nvrtcx.ValidateNVRTCFlags(mixedFlags); err == nil {
		t.Fatal("ValidateNVRTCFlags expected error on mixed flags, got nil")
	}

	// 3. Test SeparateFlags
	src, off, jit, err := nvrtcx.SeparateFlags(mixedFlags)
	if err != nil {
		t.Fatalf("SeparateFlags: %v", err)
	}
	if !slices.Equal(src, sourceFlags) {
		t.Errorf("SeparateFlags source flags mismatch:\ngot:  %v\nwant: %v", src, sourceFlags)
	}
	if len(off) != 3 {
		t.Errorf("SeparateFlags offline flags count = %d, want 3", len(off))
	}
	if jit.MaxRegisters != 64 {
		t.Errorf("SeparateFlags jit.MaxRegisters = %d, want 64", jit.MaxRegisters)
	}
	if jit.OptLevel != 3 {
		t.Errorf("SeparateFlags jit.OptLevel = %d, want 3", jit.OptLevel)
	}

	// 4. Test StripOfflineFlags
	stripped, strippedJIT := nvrtcx.StripOfflineFlags(mixedFlags)
	if !slices.Equal(stripped, sourceFlags) {
		t.Errorf("StripOfflineFlags mismatch:\ngot:  %v\nwant: %v", stripped, sourceFlags)
	}
	if strippedJIT.MaxRegisters != 64 || strippedJIT.OptLevel != 3 {
		t.Errorf("StripOfflineFlags JIT mismatch: %+v", strippedJIT)
	}
}

func TestJITOptionTranslation(t *testing.T) {
	jit := cudakernel.JITOptions{
		MaxRegisters:    64,
		ThreadsPerBlock: 256,
		OptLevel:        4,
	}

	opts, vals := jit.DriverOptions()
	if len(opts) != 3 || len(vals) != 3 {
		t.Fatalf("DriverOptions len = %d, %d; want 3, 3", len(opts), len(vals))
	}

	// Verify driver option IDs
	expectedOpts := []driver.JITOption{
		driver.JITMaxRegisters,
		driver.JITThreadsPerBlock,
		driver.JITOptimizationLevel,
	}
	for i, want := range expectedOpts {
		if opts[i] != want {
			t.Errorf("opts[%d] = %v, want %v", i, opts[i], want)
		}
	}

	// Verify driver values
	expectedVals := []uintptr{64, 256, 4}
	for i, want := range expectedVals {
		if vals[i] != want {
			t.Errorf("vals[%d] = %v, want %v", i, vals[i], want)
		}
	}

	// Test partial JIT options
	partialJIT := cudakernel.JITOptions{MaxRegisters: 32}
	pOpts, pVals := partialJIT.DriverOptions()
	if len(pOpts) != 1 || pOpts[0] != driver.JITMaxRegisters || pVals[0] != 32 {
		t.Errorf("partial JIT options mismatch: %+v, %+v", pOpts, pVals)
	}
}

func TestExternalInputIdentity(t *testing.T) {
	tmpDir := t.TempDir()
	headerPath := filepath.Join(tmpDir, "ext_header.h")

	// Write version 1 of the external header
	if err := os.WriteFile(headerPath, []byte("#define VALUE 100\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	src := `#include <ext_header.h>
extern "C" __global__ void test_k(int* out) {
    *out = VALUE;
}
`
	// Check identity change when external header changes
	flags := []string{"--include-path=" + tmpDir, "-std=c++20"}
	deps1, allTracked1 := nvrtcx.ExportResolveExternalDependencies(src, flags, nil)
	if !allTracked1 {
		t.Fatal("allTracked1 must be true for valid header on disk")
	}
	if len(deps1) != 1 || deps1[0].Path != headerPath {
		t.Fatalf("deps1 mismatch: %+v", deps1)
	}
	key1 := nvrtcx.ExportComputeDependencyKey(deps1)

	// Modify header on disk to version 2
	if err := os.WriteFile(headerPath, []byte("#define VALUE 200\n"), 0o644); err != nil {
		t.Fatalf("WriteFile v2: %v", err)
	}

	deps2, allTracked2 := nvrtcx.ExportResolveExternalDependencies(src, flags, nil)
	if !allTracked2 {
		t.Fatal("allTracked2 must be true for valid header on disk")
	}
	key2 := nvrtcx.ExportComputeDependencyKey(deps2)

	if key1 == key2 {
		t.Fatal("external header change did not alter dependency compilation key; would cause silent cache reuse")
	}
	if deps1[0].Digest == deps2[0].Digest {
		t.Fatal("header digests should differ between v1 and v2")
	}

	// Negative control: unresolvable external header must report untracked
	missingSrc := `#include "nonexistent_header_12345.h"
extern "C" __global__ void missing_k() {}
`
	missingDeps, missingAllTracked := nvrtcx.ExportResolveExternalDependencies(missingSrc, flags, nil)
	if missingAllTracked {
		t.Fatal("negative control failed: unresolvable external header marked as tracked")
	}
	if len(missingDeps) == 0 || missingDeps[0].Tracked {
		t.Fatalf("expected untracked dependency, got: %+v", missingDeps)
	}
}
