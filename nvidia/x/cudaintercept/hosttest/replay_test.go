//go:build darwin || linux

package hosttest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// buildOwned links the same off-hook producer, matcher and exported launch
// boundary to a test-only driver seam. No public admission API is substituted.
// private only selects immutable CPU table rows at compile time. The ordinary
// build must refuse that same request without even resolving driver symbols.
func buildOwned(t *testing.T, private bool, replayEdits, registryEdits, shimEdits [][2]string, rawCPU ...bool) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "driver.image"), []byte("owned-test-driver"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, pair := range []struct {
		src, dst string
		edits    [][2]string
	}{
		{"../replay.c", "replay-source.c", replayEdits},
		{"../registry.c", "registry.c", registryEdits},
		{"../interpose.c", "interpose-source.c", shimEdits},
		{"testdata/replay.c", "fixture.c", nil},
		{"../registry.h", "registry.h", nil},
		{"../replay.h", "replay.h", nil},
		{"../observe.h", "observe.h", nil},
	} {
		b, err := os.ReadFile(pair.src)
		if err != nil {
			t.Fatal(err)
		}
		body := edit(t, string(b), pair.edits...)
		if err := os.WriteFile(filepath.Join(dir, pair.dst), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "owned-replay")
	args := []string{"-std=c11", "-pthread", "-Wall", "-Wextra", "-Werror", "-I", dir}
	if private {
		args = append(args, "-DCUDAINTERCEPT_PRIVATE_FIXTURE")
	}
	if len(rawCPU) != 0 && rawCPU[0] {
		args = append(args, "-DCUDAINTERCEPT_CPU_DRIVER")
	}
	args = append(args, filepath.Join(dir, "fixture.c"), filepath.Join(dir, "registry.c"), "-o", binary)
	if runtime.GOOS == "linux" {
		args = append(args, "-ldl")
	}
	t.Logf("CPU seam compile: %s %v", cc, args)
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("fixture did not compile; no assertion credit: %v\n%s", err, out)
	}
	return binary
}

func runOwned(t *testing.T, binary, mode string, wantPass bool) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, binary, mode, filepath.Join(filepath.Dir(binary), "driver.image")).CombinedOutput()
	t.Logf("CPU seam mode=%s timeout=%v exit=%v output=%s", mode, ctx.Err() != nil, err, out)
	if ctx.Err() != nil {
		t.Fatal("fixture timeout is not an assertion outcome")
	}
	want := "mode=" + mode + " bad="
	if wantPass {
		want += "0 "
	} else {
		want += "1 "
	}
	if (err == nil) != wantPass || !strings.Contains(string(out), want) {
		t.Fatal("owned producer/dispatch assertion outcome mismatch")
	}
	return string(out)
}

func TestOwnedReplayProductionRefusal(t *testing.T) {
	runOwned(t, buildOwned(t, false, nil, nil, nil), "production", true)
}

func TestOwnedReplayRawCannotPublish(t *testing.T) {
	runOwned(t, buildOwned(t, false, nil, nil, nil, true), "raw_only", true)
}

func TestOwnedReplayProducerAndDispatch(t *testing.T) {
	binary := buildOwned(t, true, nil, nil, nil)
	for _, mode := range []string{
		"positive", "qualifier", "adapter", "qualification", "image", "shape", "graph", "stream_special", "driver", "uuid", "major_error", "minor_error",
		"bad_y", "bad_x", "bad_w", "bad_guard", "nan", "noop", "partial_load", "sync_unknown", "cleanup_unknown", "tail_identity",
		"context", "stream_context", "capture", "population", "generation", "default", "process", "operation", "single_use",
		"frame_swap", "frame_owner", "frame_function", "frame_geometry", "frame_stream", "frame_extra", "cross_thread", "recursive", "concurrent", "legacy_binding", "legacy_invalid", "owner_reentrant", "bounded",
	} {
		t.Run(mode, func(t *testing.T) { runOwned(t, binary, mode, true) })
	}
}

func TestOwnedReplayOmissionControls(t *testing.T) {
	for _, tt := range []struct {
		name, mode             string
		replay, registry, shim [][2]string
	}{
		{"minor_query_error", "minor_error", [][2]string{{"o->api.attribute(&minor, CAPABILITY_MINOR, o->device)", "(o->api.attribute(&minor, CAPABILITY_MINOR, o->device), 0)"}}, nil, nil},
		{"y_readback", "bad_y", [][2]string{{"if (memcmp(got[1] + 16, want[1] + 16, 32))\n        return CUDAINTERCEPT_OUTPUT_MISMATCH;", ""}}, nil, nil},
		{"x_readback", "bad_x", [][2]string{{"if (memcmp(got[0] + 16, want[0] + 16, 32))\n        return CUDAINTERCEPT_OUTPUT_MISMATCH;", ""}}, nil, nil},
		{"stable_w", "bad_w", [][2]string{{"if (memcmp(got[2] + 16, want[2] + 16, 32))\n        return CUDAINTERCEPT_OUTPUT_MISMATCH;", ""}}, nil, nil},
		{"guards", "bad_guard", [][2]string{{"if (memcmp(got[j], want[j], 16) || memcmp(got[j] + 48, want[j] + 48, 16))\n            return CUDAINTERCEPT_OUTPUT_MISMATCH;", "(void)j;"}}, nil, nil},
		{"completion", "positive", [][2]string{{"uint32_t r = cudaintercept_replay_finish_v2(o);", "uint32_t r=0;atomic_store(&o->pending,0);"}}, nil, nil},
		{"cleanup_quiescence", "cleanup_unknown", [][2]string{{"atomic_store_explicit(&o->pending, 1, memory_order_release);", ""}}, nil, nil},
		{"independent_qualifier", "qualifier", [][2]string{{"!memcmp(a->qualifier, r->qualifier, 65)", "1"}}, nil, nil},
		{"image_identity", "image", [][2]string{{"if (memcmp(hashes[0], r.operation.baseline, 65) || memcmp(hashes[1], r.operation.candidate, 65))\n        return CUDAINTERCEPT_IMAGE_INVALID;", ""}}, nil, nil},
		{"caller_generation", "generation", [][2]string{{"if (generation != o->capsule.operation.caller_generation)\n        refusal = CUDAINTERCEPT_GENERATION;", "(void)generation;"}}, nil, nil},
		{"single_use", "single_use", [][2]string{{"if (atomic_exchange(&o->used, 1))\n        refusal = CUDAINTERCEPT_OWNER_REUSED;", ""}}, nil, nil},
		{"parameter_frame", "frame_swap", nil, [][2]string{{"params != f->params || !params || extra", "!params || extra"}}, nil},
		{"capsule_owner", "frame_owner", nil, [][2]string{{"!owned_capsule || f->owner != owned_capsule", "!owned_capsule"}}, nil},
		{"v1_refusal", "legacy_binding", nil, [][2]string{{"else if (v1_consumed) r = *reason;", ""}}, nil},
		{"off_hook_entry", "owner_reentrant", nil, nil, [][2]string{{"int cudaintercept_owned_in_hook_v2(void) { return resolving || launching; }", "int cudaintercept_owned_in_hook_v2(void) { return 0; }"}}},
		{"function", "frame_function", nil, [][2]string{{"else if (function != replay.function) r = CUDAINTERCEPT_REGISTRY_MISS;", ""}}, nil},
		{"stream", "frame_stream", nil, [][2]string{{"else if (stream != replay.stream) r = CUDAINTERCEPT_STREAM;", "else if ((void)stream, 0) r = CUDAINTERCEPT_STREAM;"}}, nil},
		{"tail_selection", "positive", nil, nil, [][2]string{{"f = (void *)(uintptr_t)dispatch;", "f = requested_f;"}}},
		{"tail_identity", "tail_identity", nil, nil, [][2]string{{"real_cuLaunchKernel == (cuLaunchKernel_t)(uintptr_t)expected_launch", "expected_launch != 0"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			binary := buildOwned(t, true, tt.replay, tt.registry, tt.shim)
			// Except the completion/tail omissions, the unaffected positive must
			// still pass in its own fresh process with the same compiled source.
			if tt.name != "completion" && tt.name != "tail_selection" {
				runOwned(t, binary, "positive", true)
			}
			out := runOwned(t, binary, tt.mode, false)
			if tt.name == "completion" && strings.Contains(out, "early_reads=0") {
				t.Fatal("completion omission did not reach early device-readback assertion")
			}
		})
	}
}

// This checks actual producer typedefs against the operator's real header
// closure. It is compile/static only and never authorizes an admission row.
func TestOwnedReplayRealHeaderABI(t *testing.T) {
	path, want := os.Getenv(cudaHeaderEnv), os.Getenv(cudaSHA256Env)
	if path == "" && want == "" {
		t.Skip("status=unavailable: authentic toolkit header closure not supplied")
	}
	if !filepath.IsAbs(path) || want == "" {
		t.Fatal("absolute real cuda.h and SHA256 required")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != strings.ToLower(want) {
		t.Fatal("real header identity mismatch")
	}
	header := t.TempDir()
	if err := os.WriteFile(filepath.Join(header, "cuda.h"), b, 0600); err != nil {
		t.Fatal(err)
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"-std=c11", "-pthread", "-Wall", "-Wextra", "-Werror", "-DCUDAINTERCEPT_REAL_HEADERS", "-I", header, "-I", filepath.Dir(path), "-fsyntax-only", "../replay.c"}
	t.Logf("real-header static-only: %s %v", cc, args)
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("real-header producer ABI: %v\n%s", err, out)
	}
}
