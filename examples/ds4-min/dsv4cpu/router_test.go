package dsv4cpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// oracleFunction extracts a definition, never a forward declaration.
func oracleFunction(t *testing.T, s, name string) string {
	t.Helper()
	pattern := regexp.MustCompile(`(?m)^static[^;{}]*\b` + regexp.QuoteMeta(name) + `\s*\(`)
	for _, match := range pattern.FindAllStringIndex(s, -1) {
		start := match[0]
		i := match[1] - 1
		brace := strings.Index(s[i:], "{")
		semi := strings.Index(s[i:], ";")
		if brace < 0 {
			break
		}
		if semi >= 0 && semi < brace {
			continue
		}
		brace += i
		depth := 0
		for end := brace; end < len(s); end++ {
			if s[end] == '{' {
				depth++
			}
			if s[end] == '}' {
				depth--
				if depth == 0 {
					return s[start : end+1]
				}
			}
		}
		break
	}
	t.Fatal("missing function definition", name)
	return ""
}

func TestPinnedRouter(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#define DS4_MAX_EXPERT 256
#define DS4_N_EXPERT 256
#define DS4_MAX_EXPERT_USED 6
#define DS4_N_EXPERT_USED 6
#define DS4_EXPERT_WEIGHT_SCALE 2.5f
typedef struct {float logits[256];} ds4_model;
typedef struct {const void *ffn_gate_inp; const float *ffn_exp_probs_b;} ds4_layer_weights;
static void matvec_any(float *out,const ds4_model *m,const void *w,const float *x){memcpy(out,m->logits,1024);}
static const void *tensor_data(const ds4_model *m,const float *p){return p;}
static void ds4_die(const char *s){exit(2);}
`
	for _, name := range []string{"softplus_stable", "layer_router_probs_one", "layer_hash_router_weights_from_probs", "topk_desc", "layer_topk_selected_experts_from_probs"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `int main(void){ds4_model m;float bias[256],p[256],w[6],x[4096]={0};int ids[6];ds4_layer_weights l={0,bias};for(int mode=0;mode<3;mode++){for(int i=0;i<256;i++){m.logits[i]=mode==0?(float)(i%53-26)/3:mode==1?0.0f:-30.0f;bias[i]=mode==0?(float)(i%11-5)/8:0;}layer_router_probs_one(p,&m,&l,x);layer_topk_selected_experts_from_probs(ids,w,&m,&l,p);fwrite(p,4,256,stdout);fwrite(ids,4,6,stdout);fwrite(w,4,6,stdout);int hash[6]={255,4,81,0,128,63};layer_hash_router_weights_from_probs(w,p,hash);fwrite(w,4,6,stdout);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "router.c"), filepath.Join(dir, "router")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", file, "-lm", "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	raw, err := exec.Command(bin).Output()
	if err != nil {
		t.Fatal(err)
	}
	const stride = (256 + 6 + 6 + 6) * 4
	if len(raw) != 3*stride {
		t.Fatal("short router oracle")
	}
	for mode := 0; mode < 3; mode++ {
		logits, bias, p := make([]float32, 256), make([]float32, 256), make([]float32, 256)
		for i := range logits {
			switch mode {
			case 0:
				logits[i] = float32(i%53-26) / 3
				bias[i] = float32(i%11-5) / 8
			case 2:
				logits[i] = -30
			}
		}
		if err := RouterProbabilities(p, logits); err != nil {
			t.Fatal(err)
		}
		ids, w := make([]int, 6), make([]float32, 6)
		if err := SelectExperts(ids, w, p, bias, 2.5); err != nil {
			t.Fatal(err)
		}
		data := raw[mode*stride:]
		check := func(name string, got, want float32) {
			t.Helper()
			if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Abs(float64(got-want)) > 1e-6+1e-6*math.Abs(float64(want)) {
				t.Fatalf("mode %d %s got %g want %g", mode, name, got, want)
			}
		}
		for i, v := range p {
			check(fmt.Sprintf("probability %d", i), v, math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:])))
		}
		for i, id := range ids {
			if id != int(int32(binary.LittleEndian.Uint32(data[(256+i)*4:]))) {
				t.Fatalf("mode %d selected %d got %d", mode, i, id)
			}
			check("top-k weight", w[i], math.Float32frombits(binary.LittleEndian.Uint32(data[(262+i)*4:])))
		}
		if err := ExpertWeights(w, p, []int{255, 4, 81, 0, 128, 63}, 2.5); err != nil {
			t.Fatal(err)
		}
		for i, v := range w {
			check("hash weight", v, math.Float32frombits(binary.LittleEndian.Uint32(data[(268+i)*4:])))
		}
	}
	t.Log("3 real 256-expert cases; 768 probabilities, 18 selected IDs, 36 weights match pinned scalar C")
}

func ExampleRouterProbabilities() {
	p := make([]float32, 2)
	_ = RouterProbabilities(p, []float32{0, 1})
	fmt.Printf("%.3f %.3f\n", p[0], p[1])
	// Output: 0.833 1.146
}

func TestRouterDomain(t *testing.T) {
	ids, w := make([]int, 1), make([]float32, 1)
	for _, v := range []float32{float32(math.Inf(1)), float32(math.NaN())} {
		if SelectExperts(ids, w, []float32{1}, []float32{v}, 1) == nil {
			t.Fatal("nonfinite bias accepted")
		}
		if SelectExperts(ids, w, []float32{1}, nil, v) == nil {
			t.Fatal("nonfinite scale accepted")
		}
		if ExpertWeights(w, []float32{1}, ids, v) == nil {
			t.Fatal("nonfinite weight scale accepted")
		}
	}
	if SelectExperts(ids, w, []float32{math.MaxFloat32}, []float32{math.MaxFloat32}, 1) == nil {
		t.Fatal("overflowing bias score accepted")
	}
	if ExpertWeights(w, []float32{1}, []int{1}, 1) == nil {
		t.Fatal("out-of-range expert accepted")
	}
}

func ExampleSelectExperts() {
	ids, w := make([]int, 2), make([]float32, 2)
	_ = SelectExperts(ids, w, []float32{1, 2, 3}, nil, 1)
	fmt.Println(ids, w)
	// Output: [2 1] [0.6 0.4]
}

func ExampleExpertWeights() {
	w := make([]float32, 2)
	_ = ExpertWeights(w, []float32{1, 2, 3}, []int{0, 2}, 1)
	fmt.Println(w)
	// Output: [0.25 0.75]
}
