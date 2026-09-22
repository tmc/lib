package cudacapture_test

import (
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// The main.* fixtures are one module compiled from two files, which is what a
// compilation of a file that includes others produces: one CUBIN and one PTX,
// whose line directives name every file that took part.
//
// The kernel is defined in impl.cu, not in the primary main.cu, so resolving it
// requires searching the whole unit rather than the primary alone.
const multiKernel = "_Z10accumulatePfPKfi"

// multiCapture builds the two-file capture. sources is what CodeFiles.Sources
// is set to; passing nil leaves it empty, which is how a capture written before
// the field says "the unit is Source alone". files replaces the bundled .cu
// text, so that a test can leave one out or bundle it twice.
func multiCapture(t *testing.T, sources []string, files map[string]string) *cudacapture.Capture {
	t.Helper()
	read := func(name string) []byte {
		t.Helper()
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return b
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(read("main.cubin.hex"))))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	bundle := map[string][]byte{
		"main.ptx":   read("main.ptx"),
		"main.cubin": cubin,
		"main.sass":  read("main.sass"),
	}
	for name, text := range files {
		bundle[name] = []byte(text)
	}
	return &cudacapture.Capture{
		Code: cudacapture.CodeFiles{
			Source:  "main.cu",
			Sources: sources,
			PTX:     "main.ptx",
			CUBIN:   "main.cubin",
			SASS:    "main.sass",
		},
		Files: bundle,
	}
}

func defaultMultiFiles(t *testing.T) map[string]string {
	t.Helper()
	files := make(map[string]string, 2)
	for _, name := range []string{"main.cu", "impl.cu"} {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		files[name] = string(b)
	}
	return files
}

var lastSASSFile = regexp.MustCompile(`(?m)^\t//## File "[^"]*", line \d+$`)

// strayFileSASS returns the fixture disassembly with its last source directive
// replaced, so the module's tail is attributed to a file the PTX never
// mentions. Whether that is a finding depends entirely on whether the capture
// supplies the file, which is what the two tests below separate.
func strayFileSASS(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "main.sass"))
	if err != nil {
		t.Fatalf("read main.sass: %v", err)
	}
	locs := lastSASSFile.FindAllIndex(b, -1)
	if len(locs) == 0 {
		t.Fatal("main.sass names no source file; the directive spelling has moved")
	}
	last := locs[len(locs)-1]
	out := append([]byte{}, b[:last[0]]...)
	out = append(out, "\t//## File \"elsewhere.cu\", line 900"...)
	return append(out, b[last[1]:]...)
}

// TestMultiSourceKernelInSecondFile is the blocker case: a kernel defined in a
// file other than the primary must resolve, and must report which file it came
// from. Before Sources existed, LoadCode demanded every kernel's definition in
// the one named source and refused the module outright.
func TestMultiSourceKernelInSecondFile(t *testing.T) {
	c := multiCapture(t, []string{"main.cu", "impl.cu"}, defaultMultiFiles(t))
	code, err := cudacapture.LoadCode(c)
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	k := code.Kernel(multiKernel)
	if k == nil {
		t.Fatalf("%s not decoded", multiKernel)
	}
	if k.SourceFile != "impl.cu" {
		t.Errorf("SourceFile = %q, want impl.cu", k.SourceFile)
	}
	if k.SourceLine != 11 {
		t.Errorf("SourceLine = %d, want 11", k.SourceLine)
	}
	// Code.SourceFile still names the unit's primary file, which is a
	// different thing from where any one kernel is defined.
	if code.SourceFile != "main.cu" {
		t.Errorf("Code.SourceFile = %q, want main.cu", code.SourceFile)
	}
	for _, inst := range k.Instructions {
		if inst.SourceFile != "impl.cu" {
			t.Errorf("instruction at %#x attributed to %q, want impl.cu", inst.Offset, inst.SourceFile)
		}
	}
}

// TestMultiSourceUnsuppliedFileStillErrors is the negative the requester asked
// for: naming fewer files than the unit has must fail, not silently resolve the
// kernel somewhere else. A module that loads with its definitions unfound is
// how a wrong line number reaches a report.
func TestMultiSourceUnsuppliedFileStillErrors(t *testing.T) {
	c := multiCapture(t, []string{"main.cu"}, defaultMultiFiles(t))
	_, err := cudacapture.LoadCode(c)
	if err == nil {
		t.Fatal("LoadCode succeeded with the defining file left out of Sources")
	}
	// The symbol is mangled in the CUBIN and spelled "accumulate" in the
	// source; the diagnostic must name something the reader can look for.
	if !strings.Contains(err.Error(), "accumulate") {
		t.Errorf("error does not name the unresolved kernel: %v", err)
	}
}

// TestMultiSourceMissingBundledFile separates "not listed" from "listed but not
// bundled". The second is a malformed capture and must say so rather than
// reporting the kernel as undefined.
func TestMultiSourceMissingBundledFile(t *testing.T) {
	files := defaultMultiFiles(t)
	delete(files, "impl.cu")
	c := multiCapture(t, []string{"main.cu", "impl.cu"}, files)
	_, err := cudacapture.LoadCode(c)
	if err == nil {
		t.Fatal("LoadCode succeeded with a listed source absent from Files")
	}
	if !strings.Contains(err.Error(), "impl.cu") {
		t.Errorf("error does not name the missing file: %v", err)
	}
}

// TestMultiSourceSharedBaseNameRefused: two files of one unit that share a base
// name are addressed identically by every line directive, so one would shadow
// the other and half the unit's instructions would land on the wrong file's
// lines. There is no correct choice, so the capture is refused.
func TestMultiSourceSharedBaseNameRefused(t *testing.T) {
	files := defaultMultiFiles(t)
	impl := files["impl.cu"]
	delete(files, "impl.cu")
	files["a/impl.cu"], files["b/impl.cu"] = impl, impl
	c := multiCapture(t, []string{"main.cu", "a/impl.cu", "b/impl.cu"}, files)
	_, err := cudacapture.LoadCode(c)
	if err == nil {
		t.Fatal("LoadCode accepted two sources with the same base name")
	}
	if !strings.Contains(err.Error(), "base name") {
		t.Errorf("error does not explain the collision: %v", err)
	}
}

// TestMultiSourceAmbiguousDefinitionRefused: the same kernel defined in two
// files of the unit is refused rather than resolved to whichever sorts first.
// Searching several files must not turn a single file's refusal into a silent
// pick.
func TestMultiSourceAmbiguousDefinitionRefused(t *testing.T) {
	files := defaultMultiFiles(t)
	files["other.cu"] = files["impl.cu"]
	c := multiCapture(t, []string{"main.cu", "impl.cu", "other.cu"}, files)
	_, err := cudacapture.LoadCode(c)
	if err == nil {
		t.Fatal("LoadCode resolved a kernel defined in two files of the unit")
	}
	if !strings.Contains(err.Error(), "ambiguous") {
		t.Errorf("error does not report an ambiguity: %v", err)
	}
}

// TestUnsuppliedFileLinesAreNotUnclaimed is the guard's scope, and the reason
// ds4_math could not load at all. The SASS names files the capture does not
// bundle -- headers, and every file of a large unit that bundles a subset. The
// capture holds no line table to check those against, so their lines are
// unknowable rather than unclaimed, and counting them fails every multi-file
// module.
func TestUnsuppliedFileLinesAreNotUnclaimed(t *testing.T) {
	c := multiCapture(t, []string{"main.cu", "impl.cu"}, defaultMultiFiles(t))
	c.Files["main.sass"] = strayFileSASS(t)

	code, err := cudacapture.LoadCode(c)
	var fusion *cudacapture.SourceFusionError
	if errors.As(err, &fusion) {
		t.Fatalf("lines of an unsupplied file were counted against the capture: %v", fusion)
	}
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	// The instruction still carries the location the disassembler gave it: the
	// guard declining to check a file is not a reason to discard its
	// attribution.
	k := code.Kernel(multiKernel)
	last := k.Instructions[len(k.Instructions)-1]
	if last.SourceFile != "elsewhere.cu" || last.SourceLine != 900 {
		t.Errorf("last instruction attributed to %s:%d, want elsewhere.cu:900", last.SourceFile, last.SourceLine)
	}
}

// TestGuardStillFiresPerFile is the control for the test above: with the file
// supplied, the same disagreement must be caught. Without this, a guard that
// checked nothing at all would pass TestUnsuppliedFileLinesAreNotUnclaimed.
func TestGuardStillFiresPerFile(t *testing.T) {
	files := defaultMultiFiles(t)
	files["elsewhere.cu"] = "// nothing the PTX ever names\nvoid unused(int *a) {\n    *a = 0;\n}\n"
	c := multiCapture(t, []string{"main.cu", "impl.cu", "elsewhere.cu"}, files)
	c.Files["main.sass"] = strayFileSASS(t)

	_, err := cudacapture.LoadCode(c)
	var fusion *cudacapture.SourceFusionError
	if !errors.As(err, &fusion) {
		t.Fatalf("LoadCode error = %v, want *SourceFusionError", err)
	}
	if fusion.Source != "elsewhere.cu" {
		t.Errorf("Source = %q, want elsewhere.cu", fusion.Source)
	}
	if len(fusion.Unclaimed) != 1 || fusion.Unclaimed[0] != 900 {
		t.Errorf("Unclaimed = %v, want [900]", fusion.Unclaimed)
	}
	// impl.cu agrees between the two tiers and must not be dragged in by its
	// neighbour: the check is per file, not per unit.
	if len(fusion.Files) != 1 {
		var names []string
		for _, f := range fusion.Files {
			names = append(names, f.Source)
		}
		t.Errorf("Files = %v, want only elsewhere.cu", names)
	}
}

// TestSingleSourceCaptureUnchanged pins the compatibility promise: a capture
// that sets Source and no Sources behaves exactly as before the field existed.
func TestSingleSourceCaptureUnchanged(t *testing.T) {
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, "", ""))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	k := code.Kernel("_Z5scalePfPKff")
	if k == nil {
		t.Fatal("_Z5scalePfPKff not decoded")
	}
	if k.SourceLine != 11 {
		t.Errorf("SourceLine = %d, want 11", k.SourceLine)
	}
	if k.SourceFile != "kernels.cu" {
		t.Errorf("SourceFile = %q, want kernels.cu", k.SourceFile)
	}
}
