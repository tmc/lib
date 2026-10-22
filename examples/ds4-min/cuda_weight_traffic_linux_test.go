//go:build linux

package main

import (
	"os"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// TestCUDADecodeWeightTraffic reports the weight bytes one decode token moves.
//
// The reference implementation streams 9.56 GB a token. Whether our device time
// is near the bandwidth roofline or merely near it by arithmetic depends on
// whether we move the same bytes: a step that reads more is traffic-bound, and
// the remedy is to read less, not to read faster. Nothing before this measured
// the figure -- it was assumed.
func TestCUDADecodeWeightTraffic(t *testing.T) {
	const decodeSteps = 8

	// Graphs must be off. A graph replay does not re-enter the Go launch path,
	// so a counter placed there sees only the launches that were not captured
	// -- it would report a fraction of the traffic and look like a finding. The
	// bytes a decode step reads are a property of the work, not of how the
	// launches are issued, so measuring them un-graphed measures the right
	// thing; only the timing would differ, and this test does not time.
	if os.Getenv("DS4_DISABLE_CUDA_GRAPH") != "1" {
		t.Skip("DS4_DISABLE_CUDA_GRAPH=1 is required: a graph replay bypasses the Go launch path the counter sits on")
	}

	path := os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if path == "" {
		t.Skip("DS4_CUDA_PREFILL_TOKENS is required")
	}
	input, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	modelPath := os.Getenv("DS4_MODEL")
	if modelPath == "" {
		modelPath = "/home/tmc/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not accessible: %v", err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("cuda device 0 not available: %v", err)
	}
	defer dev.Close()
	cuCtx, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	defer cuCtx.Release()

	r, err := newCUDARuntime(modelPath, 4096)
	if err != nil {
		t.Fatalf("newCUDARuntime: %v", err)
	}
	defer r.Close()

	if err := r.EvalBatch(tokens, 0); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	for i := 0; i < 4; i++ {
		if err := r.Eval(tokens[0], len(tokens)+i); err != nil {
			t.Fatalf("warmup %d: %v", i, err)
		}
	}

	r.openWeightTraffic()
	base := len(tokens) + 4
	for i := 0; i < decodeSteps; i++ {
		if err := r.Eval(tokens[0], base+i); err != nil {
			t.Fatalf("decode %d: %v", i, err)
		}
	}
	ledger := r.weightTraffic()
	if ledger == nil {
		t.Fatal("no traffic ledger: openWeightTraffic did not take")
	}

	// A graph replay does not re-enter the Go launch path, so a kernel that
	// only ever runs from a captured graph would count zero however much it
	// reads. Every matvec below must appear, or the total is not a total.
	wantKernels := []string{
		"ds4_q8_0_matvec_q8_1",
		"ds4_q8_0_matvec_quantized",
		"ds4_q8_0_matvec_quantized_blockdiag",
		"ds4_iq2_xxs_pair_selected_q8_1",
		"ds4_f16_matvec_half_input",
		"ds4_f16_pair_ordered_chunks",
		"ds4_q2_k_selected_q8_1",
	}
	for _, k := range wantKernels {
		if ledger.bytes[k] == 0 {
			t.Errorf("kernel %s counted no bytes across %d decode steps: its launch site is uninstrumented, or it runs only from a captured graph", k, decodeSteps)
		}
	}

	names := make([]string, 0, len(ledger.bytes))
	for k := range ledger.bytes {
		names = append(names, k)
	}
	sort.Slice(names, func(i, j int) bool { return ledger.bytes[names[i]] > ledger.bytes[names[j]] })

	total := ledger.total()
	// Decimal GB, not GiB: the reference figure this is compared against is
	// decimal, and a 7.4% unit difference is larger than the gap being judged.
	const gb = 1e9
	t.Logf("%-40s %12s %10s %10s", "kernel", "GB/token", "launch/tok", "%bytes")
	for _, k := range names {
		b := ledger.bytes[k]
		t.Logf("%-40s %12.4f %10.1f %9.1f%%",
			k, float64(b)/decodeSteps/gb,
			float64(ledger.launches[k])/decodeSteps,
			100*float64(b)/float64(total))
	}
	t.Logf("weight traffic: %.4f GB/token across %d steps (reference streams 9.56 GB/token)",
		float64(total)/decodeSteps/gb, decodeSteps)
}
