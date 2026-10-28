//go:build darwin || linux

package hosttest

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func runRegistry(t *testing.T, binary, mode string) (bool, error, string) {
	t.Helper()
	// The deadline includes host executable startup, not GPU or hook timing.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, binary, mode).CombinedOutput()
	return ctx.Err() != nil, err, string(out)
}

func TestReplayAPIABI(t *testing.T) {
	registry, err := os.ReadFile("../registry.h")
	if err != nil {
		t.Fatal(err)
	}
	if out, err := compileABI(t, readObserveH(t), map[string]string{"registry.h": string(registry)}, "-DCUDAINTERCEPT_HAVE_REGISTRY_H", "-fsyntax-only"); err != nil {
		t.Fatalf("actual registry API ABI assertions: %v\n%s", err, out)
	}
}

func TestReplayRegistry(t *testing.T) {
	binary := buildFixture(t, "registry", shimSource(t))
	for _, mode := range []string{
		"default", "process", "operation_disabled", "miss", "no_binding", "full",
		"operation", "caller", "generation", "context_generation", "module_generation",
		"unknown_generation", "context", "unknown_context", "function", "declared_function",
		"stream", "declared_stream", "abi", "pointer_intent", "shape", "extent", "layout",
		"stride", "dtype", "alignment", "unaligned", "regions", "overlap", "overflow",
		"workspace", "target", "nvrtc", "builtins", "flags", "baseline", "candidate",
		"capture_unknown", "capture", "graph_execution", "unbound_function", "ownership", "population", "grid", "block", "shared",
		"declared_grid", "params", "extra", "bad_registration", "binding_abi", "copy",
		"consumed", "switches", "result_max", "result_zero", "reentrant", "concurrent",
		"bounded", "decision_off",
	} {
		t.Run(mode, func(t *testing.T) {
			timedOut, err, out := runRegistry(t, binary, mode)
			t.Logf("timeout=%v exit=%v output=%s", timedOut, err, out)
			if timedOut || err != nil || !strings.Contains(out, "mode="+mode+" bad=0 calls=") {
				t.Fatal("registry dispatch guard assertion failed")
			}
		})
	}
}

// Causal controls remove actual guard decisions; each fixture must still
// compile, then fail its runtime assertions through cuLaunchKernel.
func TestReplayRegistryGuardOmissions(t *testing.T) {
	registry, err := os.ReadFile("../registry.c")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ name, old, replacement, mode string }{
		{"default", "if (!atomic_load_explicit(&replay_enabled, memory_order_acquire)) return CUDAINTERCEPT_DEFAULT_OFF;", "", "default"},
		{"process", "if (atomic_load_explicit(&process_disabled, memory_order_acquire)) return CUDAINTERCEPT_PROCESS_DISABLED;", "", "process"},
		{"operation_disable", "if (atomic_load_explicit(&replay_disabled, memory_order_acquire)) return CUDAINTERCEPT_OPERATION_DISABLED;", "", "operation_disabled"},
		{"baseline", "if (!TEXT(baseline, CUDAINTERCEPT_REPLAY_BASELINE)) return CUDAINTERCEPT_BASELINE_ARTIFACT;", "", "baseline"},
		{"candidate", "if (!TEXT(candidate, CUDAINTERCEPT_REPLAY_CANDIDATE)) return CUDAINTERCEPT_CANDIDATE_ARTIFACT;", "", "candidate"},
		{"generation", "return CUDAINTERCEPT_GENERATION;\n    if (e->context != replay.context)", "return CUDAINTERCEPT_ADAPTER_UNQUALIFIED;\n    if (e->context != replay.context)", "generation"},
		{"unbound_function", "if (!had_binding && function != replay.function) return CUDAINTERCEPT_REGISTRY_MISS;", "", "unbound_function"},
		{"unbound_invalid_binding_order", "if (invalid_binding) return CUDAINTERCEPT_METADATA_ABI;\n    if (!had_binding && function != replay.function) return CUDAINTERCEPT_REGISTRY_MISS;", "if (!had_binding && function != replay.function) return CUDAINTERCEPT_REGISTRY_MISS;\n    if (invalid_binding) return CUDAINTERCEPT_METADATA_ABI;", "unbound_function"},
		{"unbound_only_contract_order", "if (!had_binding && function != replay.function) return CUDAINTERCEPT_REGISTRY_MISS;", "if (function != replay.function) return CUDAINTERCEPT_REGISTRY_MISS;", "unbound_function"},
		{"graph_execution", "if (e->capture == 3) return CUDAINTERCEPT_GRAPH_EXECUTION;", "", "graph_execution"},
		{"graph_execution_conflate_capture", "if (e->capture != 1) return CUDAINTERCEPT_CAPTURE;", "if (e->capture != 1) return CUDAINTERCEPT_GRAPH_EXECUTION;", "graph_execution"},
		{"unqualified", "return CUDAINTERCEPT_ADAPTER_UNQUALIFIED;", "return CUDAINTERCEPT_REGISTRY_MISS;", "full"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mutant := edit(t, string(registry), [2]string{tt.old, tt.replacement})
			dir := t.TempDir()
			for name, content := range map[string]string{
				"registry.c": mutant, "interpose-source.c": shimSource(t),
			} {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0600); err != nil {
					t.Fatal(err)
				}
			}
			for _, name := range []string{"registry.h", "observe.h", "hosttest/testdata/registry.c"} {
				b, err := os.ReadFile(filepath.Join("..", name))
				if err != nil {
					t.Fatal(err)
				}
				base := filepath.Base(name)
				if strings.Contains(name, "testdata") {
					base = "fixture.c"
				}
				if err := os.WriteFile(filepath.Join(dir, base), b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			cc, err := exec.LookPath("cc")
			if err != nil {
				t.Fatal(err)
			}
			binary := filepath.Join(dir, "fixture")
			if out, err := exec.Command(cc, "-std=c11", "-pthread", "-Wall", "-Wextra", "-Werror", filepath.Join(dir, "fixture.c"), filepath.Join(dir, "registry.c"), "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("mutant did not compile; not an assertion kill: %v\n%s", err, out)
			}
			timedOut, err, out := runRegistry(t, binary, tt.mode)
			t.Logf("compiling omission runtime: timeout=%v exit=%v output=%s", timedOut, err, out)
			if timedOut || err == nil || !strings.Contains(out, "bad=1") {
				t.Fatal("guard omission did not fail actual dispatch assertions")
			}
		})
	}
}

func TestReplayForwardOmission(t *testing.T) {
	binary := buildFixture(t, "registry", shimSource(t, [2]string{
		"int result = real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);",
		"int result = 0;",
	}))
	timedOut, err, out := runRegistry(t, binary, "full")
	t.Logf("compiling forwarding omission runtime: timeout=%v exit=%v output=%s", timedOut, err, out)
	if timedOut || err == nil || !strings.Contains(out, "bad=1 calls=0") {
		t.Fatal("forwarding omission did not fail actual dispatch assertions")
	}
}
