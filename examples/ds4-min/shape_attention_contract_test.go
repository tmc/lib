package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestShapeAttentionSource(t *testing.T) {
	fixture, err := os.ReadFile("testdata/shape-attention-upstream.cu")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(fixture)) != "fc6438d875321e100ef7b0ce2688a61ce7832913f0353e8919f3007c7260d577" {
		t.Fatal("attention pinned fixture changed")
	}
	want := strings.Replace(string(fixture), "__global__ static void attention_decode_mixed_kernel(", "extern \"C\" __global__ void ds4_shape_attention_decode_mixed(", 1)
	// The pinned body places block barriers inside score_thread for the 4/8
	// lane paths. Move both to one uniform rendezvous without changing dots.
	branchBarrier := "            __syncthreads();\n            for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {\n                local_max = fmaxf(local_max, scores[i]);\n            }\n"
	if strings.Count(want, branchBarrier) != 2 {
		t.Fatal("pinned attention score barrier sites changed")
	}
	want = strings.ReplaceAll(want, branchBarrier, "")
	want = strings.Replace(want, "    if (score_thread) partial[threadIdx.x] = local_max;", "    // All block threads must rendezvous before the score subset reads shared\n    // scores. In particular block512 has 256 non-score threads.\n    __syncthreads();\n    if (score_thread) {\n        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {\n            local_max = fmaxf(local_max, scores[i]);\n        }\n    }\n    if (score_thread) partial[threadIdx.x] = local_max;", 1)
	path := "cuda_shape_attention.cu"
	if p := os.Getenv("DS4_SHAPE_ATTENTION_TEST_SOURCE"); p != "" {
		path = p
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(string(got), want) || !strings.Contains(string(got), "DS4_CUDA_ATTENTION_SCORE_CAP = 8192u") {
		t.Fatal("attention CUDA body differs from pinned source")
	}
}

func TestShapeAttentionUniformBarriers(t *testing.T) {
	path := "cuda_shape_attention.cu"
	if p := os.Getenv("DS4_SHAPE_ATTENTION_TEST_SOURCE"); p != "" {
		path = p
	}
	source, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	for offset := 0; ; {
		i := strings.Index(text[offset:], "if (score_thread) {")
		if i < 0 {
			break
		}
		start := offset + i + len("if (score_thread) {")
		depth, end := 1, start
		for ; end < len(text) && depth != 0; end++ {
			switch text[end] {
			case '{':
				depth++
			case '}':
				depth--
			}
		}
		if depth != 0 || strings.Contains(text[start:end], "__syncthreads") {
			t.Fatal("ATTENTION_DIVERGENT_BARRIER in score subset")
		}
		offset = end
	}
}
