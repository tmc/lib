package stage

import (
	"context"
	"debug/elf"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
)

const hostStage = `int stage(struct stage_ctx *ctx) {
 if (!ctx || ctx->version != 1 || ctx->size != sizeof(*ctx)) return -1;
 return 0;
}`

func TestCompilerOutputBound(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("shell compiler control requires Unix")
	}
	for _, stream := range []string{"stdout", "stderr", "both"} {
		t.Run(stream, func(t *testing.T) {
			dir := t.TempDir()
			compiler := filepath.Join(dir, "noisy-compiler")
			redirect := ""
			if stream == "stderr" {
				redirect = " >&2"
			}
			body := "#!/bin/sh\ni=0\nwhile [ $i -lt 10000 ]; do\n echo abcdefghijklmnopqrstuvwxyz" + redirect + "\n"
			if stream == "both" {
				body += " echo abcdefghijklmnopqrstuvwxyz >&2\n"
			}
			body += " i=$((i+1))\ndone\n"
			if err := os.WriteFile(compiler, []byte(body), 0700); err != nil {
				t.Fatal(err)
			}
			builder, err := NewBuilder(filepath.Join(dir, "build"), compiler)
			if err != nil {
				t.Fatal(err)
			}
			artifact, err := builder.Build(context.Background(), Source{Code: hostStage})
			if artifact != nil || err == nil || !strings.Contains(err.Error(), "compiler output exceeds") {
				t.Fatalf("Build = %v, %v", artifact, err)
			}
			if len(err.Error()) > 1024 {
				t.Fatal("unbounded error")
			}
			logs, err := filepath.Glob(filepath.Join(dir, "build", "*", "compiler.log"))
			if err != nil || len(logs) != 1 {
				t.Fatalf("logs = %v, %v", logs, err)
			}
			info, err := os.Stat(logs[0])
			if err != nil || info.Size() != maxCompilerOutput {
				t.Fatalf("log size = %v, %v", info, err)
			}
		})
	}
}

func testBuilder(t *testing.T) *Builder {
	t.Helper()
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("native host shared libraries require Linux or Darwin")
	}
	if _, err := exec.LookPath("cc"); err != nil {
		t.Skip("host compiler cc unavailable")
	}
	b, err := NewBuilder(t.TempDir(), "cc")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func testLibrary(t *testing.T) *Library {
	t.Helper()
	a, err := testBuilder(t).Build(t.Context(), Source{Code: hostStage})
	if err != nil {
		t.Fatal(err)
	}
	l, err := Load(a)
	if l != nil {
		t.Cleanup(func() {
			if err := l.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func TestGenerationFileNamesAreUnique(t *testing.T) {
	b := testBuilder(t)
	a, err := b.Build(t.Context(), Source{Code: hostStage})
	if err != nil {
		t.Fatal(err)
	}
	c, err := b.Build(t.Context(), Source{Code: hostStage})
	if err != nil {
		t.Fatal(err)
	}
	if a.Path() == c.Path() || filepath.Base(a.Path()) == filepath.Base(c.Path()) || a.Generation() == c.Generation() || a.SourceDigest() != c.SourceDigest() {
		t.Fatalf("generation identity reused: %s, %s", a.Path(), c.Path())
	}
	if _, err := os.Stat(a.Path()); err != nil {
		t.Fatal("earlier generation was overwritten", err)
	}
}

func TestPinnedLibraryIsNotClosed(t *testing.T) {
	l := testLibrary(t)
	p, err := l.Pin()
	if err != nil {
		t.Fatal(err)
	}
	defer p.Release()
	oldClose := l.close
	calls := 0
	l.close = func(handle uintptr) error { calls++; return oldClose(handle) }
	if err := l.Close(); !errors.Is(err, ErrPinned) || calls != 0 {
		t.Fatalf("close while pinned = %v; actual unload calls %d", err, calls)
	}
	if _, err := l.Pin(); err == nil {
		t.Fatal("closing library admitted a new pin")
	}
	ctx, err := NewContext(nil, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Call(ctx); err != nil {
		t.Fatal("retained entry unavailable", err)
	}
	if err := p.Release(); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("unload calls = %d; want 1", calls)
	}
	if err := p.Call(ctx); err == nil {
		t.Fatal("released pin invoked unloaded code")
	}
	if err := l.Close(); err != nil || calls != 1 {
		t.Fatalf("repeat close = %v; calls %d", err, calls)
	}
}

func TestUnloadFailureRetainsOwner(t *testing.T) {
	l := testLibrary(t)
	close := l.close
	failure := errors.New("unload control")
	l.close = func(uintptr) error { return failure }
	if err := l.Close(); !errors.Is(err, failure) || l.handle == 0 {
		t.Fatalf("owner lost on failure: %v", err)
	}
	l.close = close
	if err := l.Close(); err != nil || l.handle != 0 {
		t.Fatalf("retry = %v", err)
	}
}

func TestEveryBoundEntryUsesPointerABI(t *testing.T) {
	for _, entry := range []any{(func(*Context) int32)(nil), (func(*abiValues) int32)(nil)} {
		if err := checkEntrySignature(reflect.TypeOf(entry)); err != nil {
			t.Fatal(err)
		}
	}
	type twelve struct{ A, B, C uint32 }
	for _, entry := range []any{(func(twelve) int32)(nil), (func(Context) int32)(nil), (func(*Context) Context)(nil), (func(*Context, uint64) int32)(nil), (func(...*Context) int32)(nil)} {
		if err := checkEntrySignature(reflect.TypeOf(entry)); err == nil {
			t.Fatalf("by-value or mismatched entry admitted: %T", entry)
		}
	}
}

func TestByValueCDeclarationIsRejected(t *testing.T) {
	_, err := testBuilder(t).Build(t.Context(), Source{Code: "int stage(struct stage_ctx ctx) { return 0; }"})
	if err == nil || !strings.Contains(err.Error(), "stage") {
		t.Fatalf("by-value C declaration = %v", err)
	}
}

func TestChangedCompilerOutputIsRejected(t *testing.T) {
	a, err := testBuilder(t).Build(t.Context(), Source{Code: hostStage})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(a.Path(), []byte("substituted"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(a); err == nil || !strings.Contains(err.Error(), "changed") {
		t.Fatalf("substituted output = %v", err)
	}
}

func TestActualCompilerABIMismatchIsRejected(t *testing.T) {
	a, err := testBuilder(t).Build(t.Context(), Source{Code: hostStage, Flags: []string{"-fpack-struct=1"}})
	if err != nil {
		t.Fatal(err)
	}
	l, err := Load(a)
	if l != nil {
		defer l.Close()
	}
	if err == nil || !strings.Contains(err.Error(), "stage ABI field") {
		t.Fatalf("packed compiler layout admitted: %v", err)
	}
}

func TestConcurrentPinsAndClose(t *testing.T) {
	l := testLibrary(t)
	retained, err := l.Pin()
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Release()
	ctx, err := NewContext(nil, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Go(func() {
			for j := 0; j < 16; j++ {
				p, err := l.Pin()
				if err != nil {
					return
				}
				if err := p.Call(ctx); err != nil {
					t.Error(err)
				}
				if err := p.Release(); err != nil {
					t.Error(err)
				}
			}
		})
	}
	if err := l.Close(); !errors.Is(err, ErrPinned) {
		t.Fatalf("pinned concurrent close = %v", err)
	}
	wg.Wait()
	if err := retained.Call(ctx); err != nil {
		t.Fatal(err)
	}
	if err := retained.Release(); err != nil {
		t.Fatal(err)
	}
}

func TestBuildCanceledBeforeCompiler(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := testBuilder(t).Build(ctx, Source{Code: hostStage}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled build = %v", err)
	}
}

func TestNVCCCompilerOnly(t *testing.T) {
	if os.Getenv("CUDAHOTSWAP_STAGE_NVCC_COMPILE") != "1" {
		t.Skip("explicit CPU-only nvcc compiler gate")
	}
	compiler := os.Getenv("CUDAHOTSWAP_STAGE_NVCC")
	if compiler == "" {
		t.Fatal("set exact CUDAHOTSWAP_STAGE_NVCC tool path")
	}
	dir := t.TempDir()
	if retain := os.Getenv("CUDAHOTSWAP_STAGE_NVCC_RETAIN"); retain != "" {
		if !filepath.IsAbs(retain) {
			t.Fatal("nvcc retained directory must be absolute")
		}
		dir = retain
	}
	b, err := NewBuilder(dir, compiler)
	if err != nil {
		t.Fatal(err)
	}
	a, err := b.Build(t.Context(), Source{Code: hostStage, CUDA: true, Flags: []string{"-arch=sm_121"}})
	if err != nil {
		t.Fatal(err)
	}
	if a.Generation() == 0 || a.SourceDigest() == "" {
		t.Fatal("missing actual compiler artifact identity")
	}
	if info, err := os.Stat(a.Path()); err != nil || info.Size() == 0 {
		t.Fatalf("nvcc output=%v %v", info, err)
	}
	t.Logf("compiler=%s output=%s sha256=%x", compiler, a.Path(), a.digest)
	// Deliberately no Load: nvcc libraries may have CUDA runtime constructors.
}

func TestCUDADriverLinkCompilerOnly(t *testing.T) {
	if os.Getenv("CUDAHOTSWAP_STAGE_CUDA_LINK_COMPILE") != "1" {
		t.Skip("explicit CPU-only CUDA driver link gate")
	}
	root := os.Getenv("CUDA_PATH")
	retain := os.Getenv("CUDAHOTSWAP_STAGE_CC_RETAIN")
	if root == "" || retain == "" || !filepath.IsAbs(root) || !filepath.IsAbs(retain) {
		t.Fatal("set exact CUDA_PATH and retained CPU compiler directory")
	}
	b, err := NewBuilder(retain, "cc")
	if err != nil {
		t.Fatal(err)
	}
	code := `#include <cuda.h>
int stage(struct stage_ctx *ctx) {
 if (!ctx || ctx->function_count != 1) return -1;
 return (int)cuLaunchKernel((CUfunction)ctx->functions[0].handle, 1,1,1,32,1,1,0,(CUstream)ctx->stream,0,0);
}`
	a, err := b.Build(t.Context(), Source{Code: code, Flags: []string{"-I" + filepath.Join(root, "include"), "-L" + filepath.Join(root, "lib64/stubs"), "-lcuda"}})
	if err != nil {
		t.Fatal(err)
	}
	f, err := elf.Open(a.Path())
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	libs, err := f.ImportedLibraries()
	if err != nil {
		t.Fatal(err)
	}
	linked := false
	for _, lib := range libs {
		if lib == "libcuda.so.1" {
			linked = true
		}
	}
	if !linked {
		t.Fatalf("actual CUDA stage omitted driver dependency: %v", libs)
	}
	t.Logf("compiler=%s output=%s sha256=%x libraries=%v", b.compiler, a.Path(), a.digest, libs)
	// Deliberately no Load or entry call: this is a linker contract only.
}
