package main

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

func TestShapeAttentionScoreContexts(t *testing.T) {
	for _, variant := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, err := dsv4shape.For(variant)
		if err != nil {
			t.Fatal(err)
		}
		for _, context := range []uint64{1, 3, 4, 2048, 32768, 131072} {
			stride, count, err := shapeAttentionScoreLayout(c, context)
			want := min(context, uint64(c.SlidingWindow)) + (context+3)/4
			if err != nil || uint64(stride) != want || count != want*uint64(c.Heads) {
				t.Fatalf("SCORE_CONTEXT_LAYOUT variant=%d context=%d stride=%d count=%d err=%v", variant, context, stride, count, err)
			}
			plan, _, err := planShapeConsumerBuffers(c, context)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, entry := range plan {
				if entry.name == "attention-scores" {
					found = true
					if uint64(entry.bytes) != count*4 {
						t.Fatal("SCORE_CONTEXT_BYTES")
					}
				}
			}
			if !found {
				t.Fatal("SCORE_CONTEXT_OWNER_MISSING")
			}
		}
		for _, context := range []uint64{0, uint64(math.MaxUint32) + 1, math.MaxUint64} {
			if _, _, err := shapeAttentionScoreLayout(c, context); err == nil {
				t.Fatalf("SCORE_CONTEXT_REFUSAL context=%d", context)
			}
		}
	}
}

func TestShapeAttentionExternalSource(t *testing.T) {
	base, err := os.ReadFile("cuda_shape_attention.cu")
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Replace(string(base), "enum { DS4_CUDA_ATTENTION_SCORE_CAP = 8192u };\n\n", "", 1)
	want = strings.Replace(want, "ds4_shape_attention_decode_mixed(", "ds4_shape_attention_decode_external(", 1)
	want = strings.Replace(want, "        uint32_t head_dim,\n        uint32_t score_lanes_single)", "        uint32_t head_dim,\n        float *score_scratch,\n        uint32_t score_stride,\n        uint32_t score_lanes_single)", 1)
	want = strings.Replace(want, "if (t >= n_tokens || h >= n_head) return;", "if (t >= n_tokens || h >= n_head || !score_scratch || score_stride < n_raw + n_comp) return;", 1)
	want = strings.Replace(want, "__shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];", "float *scores = score_scratch + ((uint64_t)t * n_head + h) * score_stride;", 1)
	actual, err := os.ReadFile("cuda_shape_attention_external.cu")
	if err != nil {
		t.Fatal(err)
	}
	if string(actual) != want {
		t.Fatal("EXTERNAL_SCORE_SOURCE differs beyond storage/bounds changes")
	}
}
