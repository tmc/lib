package nvrtcx

import (
	"testing"

	"github.com/tmc/lib/nvidia/nvrtc"
)

const vectorAddKernel = `
extern "C" __global__ void VectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
`

func TestCompileAndCache(t *testing.T) {
	useTestCache(t)
	if err := nvrtc.Load(""); err != nil {
		t.Skipf("NVRTC library not available: %v", err)
	}

	mod1, err := Compile(vectorAddKernel, WithArch(8, 0))
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}

	if mod1.PTX() == "" {
		t.Fatal("expected non-empty PTX string")
	}

	// Second compilation should hit the cache immediately
	mod2, err := Compile(vectorAddKernel, WithArch(8, 0))
	if err != nil {
		t.Fatalf("second nvrtcx.Compile failed: %v", err)
	}

	if mod1.Key() != mod2.Key() {
		t.Fatalf("cache keys differ: %s vs %s", mod1.Key(), mod2.Key())
	}
}

func TestClearCache(t *testing.T) {
	useTestCache(t)
	if err := ClearCache(); err != nil {
		t.Fatalf("ClearCache failed: %v", err)
	}
}

func useTestCache(t *testing.T) {
	t.Helper()
	old := cacheDirOverride
	cacheDirOverride = t.TempDir()
	t.Cleanup(func() { cacheDirOverride = old })
	memCacheMu.Lock()
	memCache = make(map[string]*CompiledModule)
	memCacheMu.Unlock()
}
