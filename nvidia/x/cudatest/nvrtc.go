package cudatest

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// probeKernel is the smallest program that proves the toolchain works. If this
// does not compile, nothing in the package under test will either, and the
// failure is in the host rather than in the kernel being gated.
const probeKernel = `extern "C" __global__ void probe(float *out) { out[0] = 1.0f; }`

// probeArches are the virtual architectures to try, newest first. Only the
// first one that libnvrtc accepts is used: a compile-only gate checks that
// kernel source is valid CUDA, which does not depend on the target, so the
// architecture only has to be one the installed toolkit knows.
var probeArches = [][2]int{{8, 0}, {7, 0}, {5, 0}}

// RequireNVRTC skips the test unless libnvrtc is installed and able to compile,
// and returns the architecture to compile against. It needs no GPU: nvrtc is a
// compiler, so a compile-only gate runs anywhere the CUDA toolkit is present.
func RequireNVRTC(t testing.TB) (major, minor int) {
	t.Helper()
	var err error
	for _, a := range probeArches {
		if _, err = CompileSource(probeKernel, a[0], a[1]); err == nil {
			return a[0], a[1]
		}
	}
	t.Skipf("nvrtc cannot compile on this host: %v", err)
	return 0, 0
}

// CompileSource compiles src for the given architecture and discards the
// result. It reports whether the source is valid CUDA, which is the only thing
// a compile-only gate asserts.
func CompileSource(src string, major, minor int) (*nvrtcx.CompiledModule, error) {
	return nvrtcx.Compile(src,
		nvrtcx.WithName("gate.cu"),
		nvrtcx.WithArch(major, minor),
		nvrtcx.WithNoCache(),
	)
}
