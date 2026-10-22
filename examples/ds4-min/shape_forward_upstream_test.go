package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

func TestShapeForwardUpstreamScratch(t *testing.T) {
	body, err := os.ReadFile("testdata/shape-forward-upstream.c")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(body)) != "f1fc5a41645d7cbd0e948718c36e2ea146b7881b072d3f738e34c99c49b61ec9" {
		t.Fatal("pinned scratch fixture changed")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal(err)
	}
	fields := regexp.MustCompile(`scratch->([a-z0-9_]+) = xmalloc`).FindAllSubmatch(body, -1)
	for _, v := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		for _, ctx := range []uint64{1, 128, 2048, 32768} {
			c, _ := dsv4shape.For(v)
			p, err := planShapeForward(c, forwardTestBinding(t, c), ctx)
			if err != nil {
				t.Fatal(err)
			}
			var s strings.Builder
			s.WriteString("#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include <string.h>\n#define QK_K 256\ntypedef struct{unsigned char ignored[292];} block_q8_K;\n")
			for key, n := range map[string]int{"EMBD": c.Embed, "HC": c.HC, "HEAD": c.Heads, "HEAD_DIM": c.HeadDim, "LORA_Q": c.QLoRA, "OUT_GROUP": c.OutGroups, "LORA_O": c.OutLoRA, "INDEXER_HEAD_DIM": c.IndexerHeadDim, "INDEXER_HEAD": c.IndexerHeads, "FF_EXP": c.ExpertFF, "EXPERT_USED": c.ExpertsUsed} {
				fmt.Fprintf(&s, "#define DS4_N_%s %d\n", key, n)
			}
			s.WriteString("typedef struct{uint32_t ctx_size,comp_cap,attn_score_cap,q8_cap;")
			for _, f := range fields {
				fmt.Fprintf(&s, "void *%s;", f[1])
			}
			s.WriteString("}ds4_cpu_decode_scratch;\nstatic void ds4_die(const char*s){(void)s;exit(2);}\n")
			fmt.Fprintf(&s, "static uint32_t ds4_default_raw_cap(uint32_t c){return c<%d?c:%d;}\n", c.SlidingWindow, c.SlidingWindow)
			s.WriteString("static void*xmalloc(size_t n){size_t*p=malloc(n+sizeof(size_t));if(!p)exit(3);*p=n;return p+1;}\n")
			s.Write(body)
			fmt.Fprintf(&s, "int main(){ds4_cpu_decode_scratch x;cpu_decode_scratch_init(&x,%d);", ctx)
			for _, b := range p.buffers {
				fmt.Fprintf(&s, "printf(\"%%zu \",((size_t*)x.%s)[-1]/sizeof(float));", b.name)
			}
			s.WriteString("}\n")
			d := t.TempDir()
			src, out := filepath.Join(d, "scratch.c"), filepath.Join(d, "scratch")
			if err := os.WriteFile(src, []byte(s.String()), 0600); err != nil {
				t.Fatal(err)
			}
			if b, err := exec.Command(cc, src, "-o", out).CombinedOutput(); err != nil {
				t.Fatalf("compile: %v %s", err, b)
			}
			b, err := exec.Command(out).Output()
			if err != nil {
				t.Fatal(err)
			}
			got := strings.Fields(string(b))
			if len(got) != len(p.buffers) {
				t.Fatal("scratch census mismatch")
			}
			for i, x := range got {
				n, err := strconv.Atoi(x)
				if err != nil || n != p.buffers[i].elements {
					t.Fatalf("variant %d context %d %s C=%s Go=%d", v, ctx, p.buffers[i].name, x, p.buffers[i].elements)
				}
			}
			t.Logf("variant=%d context=%d exact float scratch fields=%d", v, ctx, len(got))
		}
	}
}
