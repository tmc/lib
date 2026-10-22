package dsv4shape

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestUpstreamQuantStrides(t *testing.T) {
	s := string(frozenOracle(t))
	start := strings.Index(s, "static const gguf_type_info gguf_types[] = {")
	if start < 0 {
		t.Fatal("type table missing")
	}
	end := strings.Index(s[start:], "\n};")
	if end < 0 {
		t.Fatal("type table unterminated")
	}
	table := s[start : start+end]
	for _, typ := range []uint32{0, 1, 2, 8, 10, 12, 13, 14, 16, 26, 39} {
		m := regexp.MustCompile(fmt.Sprintf(`\[%d\]\s*=\s*\{"[a-z0-9_]+",\s*([0-9]+),\s*([0-9]+)\}`, typ)).FindStringSubmatch(table)
		if len(m) != 3 {
			t.Fatalf("missing type %d", typ)
		}
		block, _ := strconv.ParseUint(m[1], 10, 64)
		bytes, _ := strconv.ParseUint(m[2], 10, 64)
		for _, dims := range [][]uint64{{7168}, {7168, 3072}, {7168, 3072, 384}, {3072, 7168, 384}} {
			tensor := gguf.Tensor{TypeID: typ, Dimensions: dims}
			got, err := tensor.Bytes()
			want := dims[0] / block * bytes
			for _, n := range dims[1:] {
				want *= n
			}
			if err != nil || got != want {
				t.Fatalf("type %d dims%v Cextent%d Go%d %v", typ, dims, want, got, err)
			}
		}
	}
}

// Execute the literal upstream validator with tensor-expect functions that
// print its dimension requests. This checks shape formulas and conditional
// bindings independently of Weights; it does not validate CUDA kernels.
func TestUpstreamWeightDimensions(t *testing.T) {
	s := string(frozenOracle(t))
	start := strings.Index(s, "static void weights_validate_layout(\n")
	if start < 0 {
		t.Fatal("upstream weight validator missing")
	}
	end := strings.Index(s[start:], "\nstatic void mtp_weights_validate_layout")
	if start < 0 || end < 0 {
		t.Fatal("upstream weight validator missing")
	}
	body := s[start : start+end]
	fields := regexp.MustCompile(`l->([a-z0-9_]+)`).FindAllStringSubmatch(body, -1)
	var names []string
	for _, m := range fields {
		if !slices.Contains(names, m[1]) {
			names = append(names, m[1])
		}
	}
	for _, variant := range []Variant{Flash, Pro} {
		c, _ := For(variant)
		var src strings.Builder
		src.WriteString("#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <stdbool.h>\n#define DS4_MODEL_FAMILY 0\n#define DS4_MODEL_FAMILY_GLM_DSA 2\n#define DS4_MODEL_FAMILY_DEEPSEEK41 1\n#define DS4_TENSOR_F32 0\n#define DS4_TENSOR_F16 1\n#define DS4_TENSOR_I32 26\n#define DS4_TENSOR_Q8_0 8\n")
		for k, v := range map[string]int{"DS4_N_LAYER": c.Layers, "DS4_N_EMBD": c.Embed, "DS4_N_HC": c.HC, "DS4_N_HEAD": c.Heads, "DS4_N_HEAD_DIM": c.HeadDim, "DS4_N_OUT_GROUP": c.OutGroups, "DS4_N_LORA_O": c.OutLoRA, "DS4_N_LORA_Q": c.QLoRA, "DS4_N_INDEXER_HEAD": c.IndexerHeads, "DS4_N_INDEXER_HEAD_DIM": c.IndexerHeadDim, "DS4_N_EXPERT": c.Experts, "DS4_N_EXPERT_USED": c.ExpertsUsed, "DS4_N_VOCAB": c.Vocab, "DS4_N_FF_EXP": c.ExpertFF, "DS4_N_HASH_LAYER": c.HashLayers} {
			fmt.Fprintf(&src, "#define %s %d\n", k, v)
		}
		src.WriteString("typedef struct {const char *name; uint32_t type;} ds4_tensor;\ntypedef struct {\n")
		for _, n := range names {
			fmt.Fprintf(&src, "ds4_tensor *%s;\n", n)
		}
		src.WriteString("} ds4_layer_weights;\ntypedef struct {ds4_tensor *token_embd,*output_hc_base,*output_hc_fn,*output_hc_scale,*output_norm,*output; ds4_layer_weights layer[DS4_N_LAYER];} ds4_weights;\n")
		src.WriteString("static void ds4_die(const char *s){fprintf(stderr,\"%s\\n\",s);exit(2);}\nstatic bool weights_have_output_head(const ds4_weights *w){(void)w;return true;}\nstatic bool weights_have_partial_output_head(const ds4_weights *w){(void)w;return false;}\nstatic bool weights_layer_has_required(const ds4_layer_weights *l,uint32_t il){(void)l;(void)il;return true;}\nstatic void weights_validate_glm_dsa_layout(const ds4_weights *w,uint32_t a,uint32_t b,bool x,bool y){(void)w;(void)a;(void)b;(void)x;(void)y;exit(3);}\nstatic bool ds41_kv_source(uint32_t i){(void)i;return false;}\nstatic bool ds41_index_source(uint32_t i){(void)i;return false;}\nstatic bool ds41_engram_layer(uint32_t i){(void)i;return false;}\n")
		fmt.Fprintf(&src, "static uint32_t ds4_layer_compress_ratio(uint32_t i){if(i<2)return %d;return i%%2?128:4;}\n", map[Variant]int{Flash: 0, Pro: 128}[variant])
		src.WriteString("static void tensor_expect_layout(const ds4_tensor*t,uint32_t ty,uint32_t n,uint64_t a,uint64_t b,uint64_t c){(void)ty;printf(\"%s %u %llu %llu %llu\\n\",t->name,n,(unsigned long long)a,(unsigned long long)b,(unsigned long long)c);}\n")
		for _, fn := range []string{"tensor_expect_dense_quant_layout", "tensor_expect_f16_or_q8_0_layout", "tensor_expect_routed_expert"} {
			fmt.Fprintf(&src, "static void %s(const ds4_tensor*t,uint32_t n,uint64_t a,uint64_t b,uint64_t c){tensor_expect_layout(t,0,n,a,b,c);}\n", fn)
		}
		src.WriteString("#define tensor_expect_optional tensor_expect_layout\n")
		src.WriteString(body)
		src.WriteString("\nint main(void){ds4_weights w={0};\n")
		for _, n := range []string{"token_embd", "output_hc_base", "output_hc_fn", "output_hc_scale", "output_norm", "output"} {
			fmt.Fprintf(&src, "w.%s=&(ds4_tensor){\"%s.weight\",0};\n", n, n)
		}
		for i := 0; i < c.Layers; i++ {
			for _, n := range names {
				name := n + ".weight"
				switch n {
				case "ffn_exp_probs_b":
					name = "exp_probs_b.bias"
				case "indexer_attn_q_b":
					name = "indexer.attn_q_b.weight"
				case "indexer_proj":
					name = "indexer.proj.weight"
				}
				fmt.Fprintf(&src, "w.layer[%d].%s=&(ds4_tensor){\"blk.%d.%s\",0};\n", i, n, i, name)
			}
		}
		src.WriteString("weights_validate_layout(&w,0,DS4_N_LAYER-1,true,true);return 0;}\n")
		d := t.TempDir()
		file := filepath.Join(d, "weights.c")
		bin := filepath.Join(d, "weights")
		if err := os.WriteFile(file, []byte(src.String()), 0600); err != nil {
			t.Fatal(err)
		}
		if out, err := exec.Command("cc", "-std=c11", "-Wall", "-Werror", file, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("compile %v: %s", err, out)
		}
		out, err := exec.Command(bin).Output()
		if err != nil {
			t.Fatal(err)
		}
		specs, _ := c.Weights()
		want := make(map[string][]uint64)
		for _, x := range specs {
			want[x.Name] = x.Dimensions
		}
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			var name string
			var n int
			var a, b, d uint64
			if _, err := fmt.Sscanf(line, "%s %d %d %d %d", &name, &n, &a, &b, &d); err != nil {
				t.Fatal(err)
			}
			if n < 1 || n > 3 || !slices.Equal([]uint64{a, b, d}[:n], want[name]) {
				t.Fatalf("C dimension request %s; Go %v", line, want[name])
			}
			delete(want, name)
		}
		if len(want) != 0 {
			t.Fatalf("%d Go weights not requested by C: %v", len(want), want)
		}
	}
}
