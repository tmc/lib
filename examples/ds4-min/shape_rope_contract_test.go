package main

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

func TestShapeRoPESource(t *testing.T) {
	b, err := os.ReadFile("testdata/shape-rope-upstream.cu")
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(b)
	if hex.EncodeToString(h[:]) != "8096163b919de6bb79570fe511ca3342111727712fce87ad64f99bf39bdd74cc" {
		t.Fatal("RoPE fixture pin mismatch")
	}
	want := strings.Replace(string(b), "__global__ static void rope_tail_kernel", "extern \"C\" __global__ void ds4_shape_rope_tail", 1)
	actual, err := os.ReadFile("cuda_shape_rope.cu")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(string(actual), want) {
		t.Fatal("RoPE body differs from pinned source")
	}
}

func TestShapeRoPELayerParameters(t *testing.T) {
	for _, v := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, err := dsv4shape.For(v)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < c.Layers; i++ {
			p, err := defaultShapeRope(c, i)
			if err != nil {
				t.Fatal(err)
			}
			compressed := v == dsv4shape.Pro || i >= 2
			if !compressed {
				if p.base != 10000 || p.scale != 1 || p.extension != 0 || p.originalContext != 0 || p.attentionFactor != 1 {
					t.Fatalf("ROPE_DENSE_PARAMETERS layer=%d: %+v", i, p)
				}
			} else {
				if p.base != 160000 || p.scale != 1.0/16 || p.extension != 1 || p.originalContext != 65536 {
					t.Fatalf("ROPE_COMPRESSED_PARAMETERS variant=%d layer=%d: %+v", v, i, p)
				}
				want := float32(1) / (1 + float32(0.1)*float32(math.Log(16)))
				if p.attentionFactor != want {
					t.Fatalf("ROPE_MAGNITUDE_FACTOR layer=%d got=%g want=%g", i, p.attentionFactor, want)
				}
			}
			if p.betaFast != 32 || p.betaSlow != 1 {
				t.Fatalf("ROPE_BETA layer=%d: %+v", i, p)
			}
		}
	}
}
