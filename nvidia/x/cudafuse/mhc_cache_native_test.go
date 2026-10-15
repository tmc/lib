package cudafuse

import (
	"context"
	"os"
	"reflect"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestMHCNativeCompilerCache requires separate compiler-host authorization.
// It compiles real source and never creates a CUDA device or loads an artifact.
func TestMHCNativeCompilerCache(t *testing.T) {
	if os.Getenv("CUDA_MHC_CACHE_TEST") != "1" {
		t.Skip("requires an admitted Linux compiler host and CUDA_MHC_CACHE_TEST=1")
	}
	if runtime.GOOS != "linux" {
		t.Fatal("the cache recipe requires verified Linux mappings")
	}
	ctx := context.Background()
	c := mhcCacheTestConfig()
	// Establish real compiler/builtins mappings. A mapping, rather than a
	// guessed library beside nvrtc, is required before cache admission.
	sources, err := MHCSource(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := compileMHCArtifact(ctx, sources[0], 12, 1, nvrtcx.WithArch(12, 1), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithNoCache()); err != nil {
		t.Fatal(err)
	}
	var tools []cudacache.ToolDigest
	for _, tool := range []struct{ name, opened string }{{"nvrtc", nvrtc.Path()}, {"nvrtc-builtins", "libnvrtc-builtins.so"}} {
		path, err := cudacache.LoadedLibraryPath(tool.opened)
		if err != nil {
			t.Fatal(err)
		}
		digest, err := cudacache.DigestTool(tool.name, path)
		if err != nil {
			t.Fatal(err)
		}
		tools = append(tools, digest)
		t.Logf("mapped %s: %s sha256=%s", tool.name, path, digest.SHA256)
	}
	declared := cudacache.ToolProvenance{Tools: tools}
	cfg := cudacache.Config{RootDir: t.TempDir()}
	cache, err := cudacache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileMHCCached(ctx, c, 12, 1, cudacache.NewNVRTCAdapter(cache), declared)
	if err != nil {
		t.Fatal(err)
	}
	fresh, err := cudacache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	adapter := cudacache.NewNVRTCAdapter(fresh)
	prov, major, minor, driverName, err := mhcCacheEnvironment(tools)
	if err != nil {
		t.Fatal(err)
	}
	requests, err := mhcCacheRequests(c, 12, 1, major, minor)
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range requests {
		id := r.identity(major, minor, driverName, prov)
		a, hit, err := adapter.Lookup(ctx, r.InputKey, id)
		if err != nil || !hit {
			t.Fatalf("fresh adapter entry %d: hit=%v error=%v", i, hit, err)
		}
		if err := checkMHCCacheArtifact(r, id, a); err != nil {
			t.Fatal(err)
		}
		if a.Kind != cudakernel.KindPTX || !reflect.DeepEqual(a, compiled[i]) {
			t.Fatal("real compiled artifact did not survive serialization")
		}
		t.Logf("entry=%s source-key=%s adapter-key=%s ptx-sha256=%s flags=%q toolchain=%s", a.EntryPoint, r.InputKey, cudacache.CompilerCacheKey(r.InputKey, id), a.Digest, a.CompilerOpts, a.Toolchain)
	}
	cached, err := CompileMHCCached(ctx, c, 12, 1, adapter, declared)
	if err != nil || !reflect.DeepEqual(cached, compiled) {
		t.Fatalf("cached production entry point: %v", err)
	}
}
