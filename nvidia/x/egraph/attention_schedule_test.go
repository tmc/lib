package egraph

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaref"
)

func scheduleCandidate(t *testing.T, n, d, tile int, mask *AttentionRowMask) *OnlineAttentionCandidate {
	t.Helper()
	snapshot := maskedRegionSnapshot(t, n, d)
	scores, values := make([]float32, n), make([]float32, n*d)
	for key := range scores {
		scores[key] = float32(key*3 - 7)
		for column := range d {
			values[key*d+column] = float32((key+1)*(column%5+1) - 3)
		}
	}
	if mask != nil {
		for key, excluded := range mask.Dense() {
			if excluded {
				scores[key] = float32(math.Inf(-1))
			}
		}
	}
	inputs := map[string][]float32{"scores": scores, "values": values}
	policy := AttentionCandidatePolicy{Absolute: 2e-5, Relative: 2e-7}
	var candidate *OnlineAttentionCandidate
	var err error
	if mask == nil {
		candidate, err = NewCheckedOnlineAttentionCandidate(snapshot, inputs, tile, policy)
	} else {
		candidate, err = NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, *mask, tile, policy)
	}
	if err != nil {
		t.Fatal(err)
	}
	return candidate
}

func TestOnlineAttentionSchedulePackingAndShapes(t *testing.T) {
	for _, n := range []int{1, 2, 3, 5, 31, 64} {
		for _, d := range []int{1, 3, 31, 64} {
			widths := []int{1, max(1, n/2), n}
			for _, tile := range widths {
				candidate := scheduleCandidate(t, n, d, tile, nil)
				bound, err := NewOnlineAttentionSchedule(candidate, 2, "attention")
				if err != nil {
					t.Fatalf("n=%d d=%d tile=%d: %v", n, d, tile, err)
				}
				inputs := candidate.Inputs()
				packed := bound.PackedInput()
				words := int(bound.Schedule().PayloadBytes / 4)
				if len(packed) != len(candidate.Derivation().Tiles)*words {
					t.Fatal("packed extent mismatch")
				}
				for i, step := range candidate.Derivation().Tiles {
					row := packed[i*words : (i+1)*words]
					for key := step.Start; key < step.End; key++ {
						if math.Float32bits(row[key-step.Start]) != math.Float32bits(inputs["scores"][key]) {
							t.Fatal("score bits changed")
						}
						for col := range d {
							if math.Float32bits(row[tile+(key-step.Start)*d+col]) != math.Float32bits(inputs["values"][key*d+col]) {
								t.Fatal("value bits changed")
							}
						}
					}
					for index, word := range row {
						used := index < step.End-step.Start || index >= tile && index < tile+(step.End-step.Start)*d
						if !used && math.Float32bits(word) != 0 {
							t.Fatal("unread tail/padding is not zero")
						}
					}
				}
				grid, block := bound.Launch()
				if grid != [3]uint32{1, 1, 1} || block != [3]uint32{64, 1, 1} || bound.SharedBytes() > 48<<10 || bound.CandidateIdentity() != candidate.Identity() {
					t.Fatal("launch/provenance mismatch")
				}
			}
		}
	}
	// Assignment, rather than arithmetic, must preserve signed zero and -Inf.
	mask, _ := NewPrefixAttentionRowMask(1, 2)
	inputs := map[string][]float32{"scores": {math.Float32frombits(0x80000000), float32(math.Inf(-1))}, "values": {math.Float32frombits(0x80000000), 3}}
	candidate, err := NewCheckedMaskedOnlineAttentionCandidate(maskedRegionSnapshot(t, 2, 1), inputs, mask, 1, AttentionCandidatePolicy{Absolute: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	bound, err := NewOnlineAttentionSchedule(candidate, 2, "attention")
	if err != nil {
		t.Fatal(err)
	}
	packed := bound.PackedInput()
	if math.Float32bits(packed[0]) != 0x80000000 || math.Float32bits(packed[1]) != 0x80000000 || math.Float32bits(packed[4]) != 0xff800000 {
		t.Fatal("non-arithmetic packing changed input bits")
	}
}

func TestOnlineAttentionScheduleOwnershipAndRefusals(t *testing.T) {
	candidate := scheduleCandidate(t, 5, 3, 2, nil)
	a, err := NewOnlineAttentionSchedule(candidate, 2, "attention")
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewOnlineAttentionSchedule(candidate, 2, "attention")
	if err != nil || a.Identity() != b.Identity() {
		t.Fatalf("deterministic identity: %v", err)
	}
	changed, _ := NewOnlineAttentionSchedule(candidate, 3, "attention")
	if changed.Identity() == a.Identity() {
		t.Fatal("stage count absent from identity")
	}
	changed, _ = NewOnlineAttentionSchedule(candidate, 2, "different")
	if changed.Identity() == a.Identity() {
		t.Fatal("source absent from identity")
	}
	baseline := a.PackedInput()
	a.PackedInput()[0] = 999
	a.Schedule().Steps[0].Kind = "mutated"
	a.Schema().Params[0].Name = "mutated"
	if !slices.Equal(baseline, a.PackedInput()) || a.Schema().Params[0].Name != "packed" || len(a.Schedule().Validate()) != 0 {
		t.Fatal("accessor aliases retained binding")
	}
	for _, stages := range []int{-1, 0, 1, 9, 65} {
		if _, err := NewOnlineAttentionSchedule(candidate, stages, "attention"); err == nil {
			t.Fatalf("stage count %d admitted", stages)
		}
	}
	for _, candidate := range []*OnlineAttentionCandidate{nil, {}} {
		if _, err := NewOnlineAttentionSchedule(candidate, 2, "attention"); err == nil {
			t.Fatal("unchecked zero candidate admitted")
		}
	}
	large := scheduleCandidate(t, 64, 64, 64, nil)
	if _, err := NewOnlineAttentionSchedule(large, 3, "attention"); err == nil {
		t.Fatal("large triple-buffer source exceeded shared budget")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if artifact, _, err := a.Compile(ctx, 8, 0); artifact != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled compile: %v", err)
	}
	if artifact, _, err := a.Compile(nil, 8, 0); artifact != nil || err == nil {
		t.Fatal("nil compile context admitted")
	}
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for range 10 {
				a.Schedule().Steps[0].Stage = 99
				a.PackedInput()[0] = 99
				if !slices.Equal(baseline, a.PackedInput()) {
					t.Error("concurrent accessor changed retained input")
				}
			}
		})
	}
	wg.Wait()
}

// The shim executes the emitted CUDA C++ body on 64 real CPU threads. Static
// shared storage and C++20 barriers model this fixed one-CTA fixture only.
// It does not model SIMT execution, device math, CUDA or hardware barriers.
const attentionHostShim = `#pragma once
#include <barrier>
#include <bit>
#define __global__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
struct dim3 { unsigned int x, y, z; };
inline dim3 blockDim{64,1,1}, gridDim{1,1,1};
inline thread_local dim3 threadIdx{0,0,0};
inline std::barrier<> attention_barrier(64);
inline void __syncthreads() { attention_barrier.arrive_and_wait(); }
inline float __uint_as_float(unsigned int bits) { return std::bit_cast<float>(bits); }
`

func runAttentionSource(t *testing.T, source string, packed []float32, d int) []float32 {
	t.Helper()
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("emitted attention host execution requires a system C++ compiler")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "cuda_runtime.h"), []byte(attentionHostShim), 0600); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	b.WriteString(source)
	b.WriteString("\n#include <cstdio>\n#include <thread>\n#include <vector>\n#include <cstdint>\nint main() {\n  const uint32_t input_bits[] = {")
	for _, value := range packed {
		fmt.Fprintf(&b, "0x%08xu,", math.Float32bits(value))
	}
	b.WriteString("};\n")
	fmt.Fprintf(&b, "  alignas(16) float input[%d], output[%d];\n", len(packed)+8, d+8)
	fmt.Fprintf(&b, "  for (int i=0;i<%d;i++) input[i]=std::bit_cast<float>(0x4f123456u);\n", len(packed)+8)
	fmt.Fprintf(&b, "  for (int i=0;i<%d;i++) input[i+4]=std::bit_cast<float>(input_bits[i]);\n", len(packed))
	fmt.Fprintf(&b, "  for (int i=0;i<%d;i++) output[i]=std::bit_cast<float>(0x4f123456u);\n", d+8)
	b.WriteString("  std::vector<std::thread> threads;\n  for (unsigned int tid=0;tid<64;tid++) threads.emplace_back([&,tid] { threadIdx.x=tid; attention(input+4,output+4); });\n  for (auto& thread:threads) thread.join();\n")
	fmt.Fprintf(&b, "  for (int i=0;i<%d;i++) if (std::bit_cast<uint32_t>(input[i+4])!=input_bits[i]) return 2;\n", len(packed))
	fmt.Fprintf(&b, "  for (int i=0;i<4;i++) if (std::bit_cast<uint32_t>(input[i])!=0x4f123456u || std::bit_cast<uint32_t>(input[%d+i])!=0x4f123456u || std::bit_cast<uint32_t>(output[i])!=0x4f123456u || std::bit_cast<uint32_t>(output[%d+i])!=0x4f123456u) return 3;\n", len(packed)+4, d+4)
	fmt.Fprintf(&b, "  for (int d=0;d<%d;d++) std::printf(\"%%08x\\n\",std::bit_cast<uint32_t>(output[d+4]));\n  return 0;\n}\n", d)
	path := filepath.Join(dir, "attention.cc")
	if err := os.WriteFile(path, []byte(b.String()), 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "attention-host")
	if output, err := attentionHostOutput(25*time.Second, compiler, "-std=c++20", "-pthread", "-ffp-contract=off", "-I", dir, path, "-o", binary); err != nil {
		t.Fatalf("system C++ host compilation: %v\n%s", err, output)
	}
	output, err := attentionHostOutput(10*time.Second, binary)
	if err != nil {
		t.Fatalf("emitted-body host execution/guards: %v\n%s", err, output)
	}
	fields := strings.Fields(string(output))
	if len(fields) != d {
		t.Fatalf("emitted output extent %d, want %d: %s", len(fields), d, output)
	}
	got := make([]float32, d)
	for i, field := range fields {
		word, err := strconv.ParseUint(field, 16, 32)
		if err != nil {
			t.Fatal(err)
		}
		got[i] = math.Float32frombits(uint32(word))
	}
	return got
}

// WaitDelay bounds inherited-pipe waits after child exit/cancellation. It does
// not establish that every descendant has exited. A deadline or WaitDelay
// error leaves emitted-body execution failed, never qualified.
func attentionHostOutput(timeout time.Duration, path string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	command := exec.CommandContext(ctx, path, args...)
	command.WaitDelay = time.Second
	output, err := command.CombinedOutput()
	return output, errors.Join(err, ctx.Err())
}

func attentionScheduleMatches(got, want []float32, policy AttentionCandidatePolicy) bool {
	if len(got) != len(want) {
		return false
	}
	for i, value := range got {
		if math.IsNaN(float64(want[i])) || policy == (AttentionCandidatePolicy{}) {
			if math.Float32bits(value) != math.Float32bits(want[i]) {
				return false
			}
		} else if !finite64(float64(value)) || math.Abs(float64(value)-float64(want[i])) > policy.Absolute+policy.Relative*math.Abs(float64(want[i])) {
			return false
		}
	}
	return true
}

func TestOnlineAttentionScheduleEmittedHostReference(t *testing.T) {
	causal, _ := NewCausalAttentionRowMask(2, 5)
	tree, _ := NewTreeAttentionRowMask([]int{-1, -1, 1, 0, 2}, 2, 0)
	treePrefix, _ := NewTreeAttentionRowMask([]int{-1, 0, 0, 1, 2}, 3, 1)
	prefix, _ := NewPrefixAttentionRowMask(3, 5)
	for _, test := range []struct {
		name       string
		n, d, tile int
		mask       *AttentionRowMask
	}{
		{"single", 1, 1, 1, nil}, {"dense_tail", 5, 3, 2, nil}, {"causal", 5, 3, 2, &causal},
		{"tree_empty_tiles", 5, 3, 1, &tree}, {"tree_prefix", 6, 3, 2, &treePrefix}, {"prefix", 5, 3, 2, &prefix}, {"maximum", 64, 64, 64, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := scheduleCandidate(t, test.n, test.d, test.tile, test.mask)
			bound, err := NewOnlineAttentionSchedule(candidate, 2, "attention")
			if err != nil {
				t.Fatal(err)
			}
			refs, err := cudaref.EvaluateSnapshot(candidate.Snapshot(), candidate.Inputs())
			if err != nil {
				t.Fatal(err)
			}
			want := refs[candidate.Snapshot().Outputs()[0].Name]
			got := runAttentionSource(t, bound.Source(), bound.PackedInput(), test.d)
			if !attentionScheduleMatches(got, want, candidate.Derivation().Policy) {
				t.Fatalf("emitted body %v != independent source %v", got, want)
			}
		})
	}
}

func TestOnlineAttentionScheduleEmittedOmissionControls(t *testing.T) {
	candidate := scheduleCandidate(t, 5, 3, 2, nil)
	bound, err := NewOnlineAttentionSchedule(candidate, 2, "attention")
	if err != nil {
		t.Fatal(err)
	}
	for _, mutant := range []struct{ name, old, replacement string }{
		{"inverse_rescale", "exp(m - m_new)", "exp(m_new - m)"},
		{"omit_normalizer", "(float)(u[d]/l)", "(float)u[d]"},
		{"wrong_mask", "0x0000000000000000ULL", "0x0000000000000010ULL"},
		{"include_tail", "const int start = 4, count = 1;", "const int start = 4, count = 2;"},
	} {
		t.Run(mutant.name, func(t *testing.T) {
			if !strings.Contains(bound.Source(), mutant.old) {
				t.Fatal("mutation site missing")
			}
			source := strings.ReplaceAll(bound.Source(), mutant.old, mutant.replacement)
			got := runAttentionSource(t, source, bound.PackedInput(), 3)
			if attentionScheduleMatches(got, candidate.Reference(), candidate.Derivation().Policy) {
				t.Fatalf("emitted computation mutation survived: %v", got)
			}
		})
	}
	packed := bound.PackedInput()
	packed[2*int(bound.Schedule().PayloadBytes/4)] = -9
	if got := runAttentionSource(t, bound.Source(), packed, 3); attentionScheduleMatches(got, candidate.Reference(), candidate.Derivation().Policy) {
		t.Fatal("changed packed source binding survived reference")
	}
}

func TestOnlineAttentionScheduleFullyMaskedEmittedHost(t *testing.T) {
	snapshot := maskedRegionSnapshot(t, 5, 3)
	inputs := map[string][]float32{"scores": make([]float32, 5), "values": make([]float32, 15)}
	for i := range inputs["scores"] {
		inputs["scores"][i] = float32(math.Inf(-1))
	}
	for i := range inputs["values"] {
		inputs["values"][i] = float32(i - 7)
	}
	refs, err := cudaref.EvaluateSnapshot(snapshot, inputs)
	if err != nil {
		t.Fatal(err)
	}
	want := refs[snapshot.Outputs()[0].Name]
	if len(want) != 3 {
		t.Fatal("independent source extent mismatch")
	}
	canonical := true
	for _, value := range want {
		canonical = canonical && math.Float32bits(value) == 0x7fc00000
	}
	mask, _ := NewPrefixAttentionRowMask(0, 5)
	candidate, err := NewCheckedMaskedOnlineAttentionCandidate(snapshot, inputs, mask, 2, AttentionCandidatePolicy{Absolute: 1e9})
	if !canonical {
		if err == nil || !strings.Contains(err.Error(), "nan bits differ from source") {
			t.Fatalf("noncanonical source not refused: %v", err)
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	bound, err := NewOnlineAttentionSchedule(candidate, 2, "attention")
	if err != nil {
		t.Fatal(err)
	}
	got := runAttentionSource(t, bound.Source(), bound.PackedInput(), 3)
	if !attentionScheduleMatches(got, want, candidate.Derivation().Policy) {
		t.Fatal("emitted canonical nan bits differ from independent source")
	}
	mutant := strings.ReplaceAll(bound.Source(), "__uint_as_float(0x7fc00000u)", "0.0f")
	if got := runAttentionSource(t, mutant, bound.PackedInput(), 3); attentionScheduleMatches(got, want, candidate.Derivation().Policy) {
		t.Fatal("zero-finalizer mutation survived")
	}
}
