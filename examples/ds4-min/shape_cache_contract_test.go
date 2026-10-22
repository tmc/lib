package main

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

func TestShapeCacheAdmission(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	for _, context := range []uint64{0, 1 << 32} {
		if _, err := planShapeCache(c, context, math.MaxUint64, 0); err == nil {
			t.Fatalf("accepted context %d", context)
		}
	}
	for _, context := range []uint64{1, 4096, 32768, 1 << 20, math.MaxUint32} {
		p, err := planShapeCache(c, context, math.MaxUint64, 0)
		if err != nil {
			t.Fatal(err)
		}
		if p.layers != 61 {
			t.Fatal("Pro layer count")
		}
		var bytes uint64
		for _, v := range p.buffers {
			bytes += uint64(v.elements) * 4
			if strings.Contains(v.name, "score") && v.initial != math.Float32bits(-1e30) {
				t.Fatal("score initializer")
			}
		}
		if bytes != p.bytes {
			t.Fatalf("unaccounted bytes: %d vs %d", bytes, p.bytes)
		}
		if _, err := planShapeCache(c, context, p.bytes+1, 2); err == nil {
			t.Fatal("admitted insufficient capacity")
		}
	}
}

func TestShapeCacheUpstreamExtents(t *testing.T) {
	body, err := os.ReadFile("testdata/shape-cache-upstream.c")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(body)) != "357ccb83d6f0465e8dd62ad9d7236d0d97cfad921132efdc1004b7f0ae995f24" {
		t.Fatal("pinned cache fixture changed")
	}
	compiler, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal(err)
	}
	for _, variant := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, _ := dsv4shape.For(variant)
		p, err := planShapeCache(c, 4096, math.MaxUint64, 0)
		if err != nil {
			t.Fatal(err)
		}
		var source strings.Builder
		source.WriteString("#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#define DS4_VARIANT_FLASH 0\n#define DS4_VARIANT_PRO 1\n#define DS4_VARIANT_FLASH41 2\n#define DS4_NEG_INF (-1.0e30f)\n")
		fmt.Fprintf(&source, "#define DS4_MODEL_VARIANT %d\n#define DS4_N_LAYER %d\n#define DS4_N_HEAD_DIM %d\n#define DS4_N_INDEXER_HEAD_DIM %d\n", variant, c.Layers, c.HeadDim, c.IndexerHeadDim)
		source.WriteString("typedef struct {uint32_t cap_raw,compress_ratio,comp_cap;float *raw_kv,*attn_comp_kv,*attn_state_kv,*attn_state_score,*index_comp_kv,*index_state_kv,*index_state_score;} layer; typedef struct {uint32_t head_dim;layer layer[DS4_N_LAYER];} ds4_kv_cache; static void ds4_die(const char*s){(void)s;exit(2);} static uint32_t ds4_default_raw_cap(uint32_t c){return c;} static void*xmalloc_zeroed(size_t n,size_t s){printf(\"%zu \",n);return calloc(n,s);} static void*xmalloc(size_t n){printf(\"%zu \",n/sizeof(float));return malloc(n);} static uint32_t ds4_layer_compress_ratio(uint32_t i);\n")
		source.Write(body)
		fmt.Fprintf(&source, "\nstatic uint32_t ds4_layer_compress_ratio(uint32_t i){return ds4_expected_layer_compress_ratio(i);} int main(){ds4_kv_cache c;kv_cache_init(&c,4096,%d);}\n", c.SlidingWindow)
		d := t.TempDir()
		input, output := filepath.Join(d, "cache.c"), filepath.Join(d, "cache")
		if err := os.WriteFile(input, []byte(source.String()), 0600); err != nil {
			t.Fatal(err)
		}
		if b, err := exec.Command(compiler, input, "-o", output).CombinedOutput(); err != nil {
			t.Fatalf("compile: %v %s", err, b)
		}
		b, err := exec.Command(output).Output()
		if err != nil {
			t.Fatal(err)
		}
		var got, want []int
		for _, s := range strings.Fields(string(b)) {
			n, err := strconv.Atoi(s)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, n)
		}
		for _, v := range p.buffers {
			if !strings.Contains(v.name, "pooled") {
				want = append(want, v.elements)
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%v cache extents differ: C=%v Go=%v", variant, got, want)
		}
		t.Logf("variant=%v C persistent buffers=%d Go persistent+pooled=%d", variant, len(got), len(p.buffers))
	}
}
