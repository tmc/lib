//go:build linux

package cudacapture_test

import (
	"context"
	"path"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const mapKernelSource = `
extern "C" __global__ void mapped(unsigned int *p, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		p[i] += 1;
	}
}
`

// TestSourceMapByBaseOnRecordedPaths is the native gate for the source
// mapping. Every other test in this package feeds SourceMapByBase directives
// written by hand, which cannot show what nvrtc actually records: it compiles
// through a temporary directory this process does not choose, so the path in
// the disassembly is one no caller could have predicted or written down.
//
// The test requires the recorded path to be exactly that — an absolute path
// that is not the bundled artifact name — and then requires SourceMapByBase to
// find it and RewriteSASSSources to replace it.
func TestSourceMapByBaseOnRecordedPaths(t *testing.T) {
	ctx := context.Background()
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(mapKernelSource,
		nvrtcx.WithName("kernels.cu"),
		nvrtcx.WithDeviceArch(dev),
		nvrtcx.WithFlag("--generate-line-info"),
		nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	cubin := compiled.CUBIN()
	if len(cubin) == 0 {
		t.Fatal("compiler produced no CUBIN")
	}

	d := cudasass.Disassembler{TempDir: t.TempDir()}
	assembly, err := d.Disassemble(ctx, cubin, cudasass.Options{
		PrintCode:   true,
		InlineLines: true,
		PTXLines:    true,
	})
	if err != nil {
		t.Fatalf("disassemble: %v", err)
	}

	recorded := cudacapture.SASSSourcePaths(assembly.Stdout)
	if len(recorded) == 0 {
		t.Fatal("disassembly records no source paths; --generate-line-info did not survive to the SASS")
	}
	t.Logf("recorded source paths: %q", recorded)

	var compiledThrough string
	for _, p := range recorded {
		if path.Base(p) == "kernels.cu" {
			compiledThrough = p
			break
		}
	}
	if compiledThrough == "" {
		t.Fatalf("no recorded path has base kernels.cu; got %q", recorded)
	}
	// The point of the fallback: the recorded path is not the name the caller
	// bundles, and not one it could have written into a map beforehand.
	if compiledThrough == "kernels.cu" {
		t.Fatal("recorded path is the bundled name; this build did not compile through a temporary directory, so the test proves nothing SourceMapByBase is for")
	}
	if !strings.HasPrefix(compiledThrough, "/") {
		t.Errorf("recorded path %q is not absolute", compiledThrough)
	}

	m, err := cudacapture.SourceMapByBase(assembly.Stdout, "kernels.cu")
	if err != nil {
		t.Fatalf("SourceMapByBase: %v", err)
	}
	if got := m[compiledThrough]; got != "kernels.cu" {
		t.Fatalf("map[%q] = %q, want kernels.cu", compiledThrough, got)
	}

	rewritten := cudacapture.RewriteSASSSources(assembly.Stdout, m)
	if strings.Contains(string(rewritten), compiledThrough) {
		t.Errorf("rewritten SASS still names the temporary path %q", compiledThrough)
	}
	if !strings.Contains(string(rewritten), `"kernels.cu"`) {
		t.Error("rewritten SASS does not name the bundled artifact")
	}
	// Paths the map does not cover keep their recorded spelling.
	for _, p := range recorded {
		if p == compiledThrough {
			continue
		}
		if !strings.Contains(string(rewritten), p) {
			t.Errorf("rewrite dropped unmapped recorded path %q", p)
		}
	}
}
