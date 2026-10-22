package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

func TestShapeGroupingPlan(t *testing.T) {
	for _, variant := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, err := dsv4shape.For(variant)
		if err != nil {
			t.Fatal(err)
		}
		p, err := planShapeGrouping(c, 2048)
		if err != nil {
			t.Fatal(err)
		}
		if p.experts != c.Experts || p.pairs != 12288 || p.offsets != c.Experts+1 || p.active != c.Experts {
			t.Fatalf("%v: plan %+v", variant, p)
		}
		for _, n := range []int{-1, 0, 2731} {
			if _, err := planShapeGrouping(c, n); err == nil {
				t.Fatalf("accepted %d tokens", n)
			}
		}
	}
}

// This executes the actual integer kernel body on a CPU. It is not a CUDA
// compile or device qualification; those require a separate physical gate.
func TestShapeGroupingKernelHost(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	path := "cuda_shape_group.cu"
	if alternate := os.Getenv("DS4_GROUP_SOURCE"); alternate != "" {
		path = alternate
	}
	kernel, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, experts := range []int{256, 384} {
		ids := []int{383, 0, 256, 255, -1, 0, 384, 127, 255, 382, 1, 256}
		offsets := make([]int, experts+1)
		var pairs, active []int
		invalid := 0
		for _, id := range ids {
			if id < 0 || id >= experts {
				invalid++
			}
		}
		for e := 0; e < experts; e++ {
			offsets[e] = len(pairs)
			before := len(pairs)
			for p, id := range ids {
				if id == e {
					pairs = append(pairs, p)
				}
			}
			if len(pairs) != before {
				active = append(active, e)
			}
		}
		offsets[experts] = len(pairs)
		var source strings.Builder
		source.WriteString("#include <cstdio>\n#define __global__\nstruct Dim {unsigned x;}; Dim blockIdx={0}, threadIdx={0};\n")
		source.Write(kernel)
		fmt.Fprintf(&source, "\nint main(){int ids[]={")
		for _, id := range ids {
			fmt.Fprintf(&source, "%d,", id)
		}
		fmt.Fprintf(&source, "}; int offsets[%d]={},pairs[12]={},active[%d]={},count=-1,invalid=-1; ds4_shape_group_csr(ids,offsets,pairs,active,&count,&invalid,12,%d); printf(\"%%d %%d\\n\",count,invalid); for(int i=0;i<%d;i++)printf(\"%%d \",offsets[i]); for(int i=0;i<offsets[%d];i++)printf(\"%%d \",pairs[i]); for(int i=0;i<count;i++)printf(\"%%d \",active[i]);}\n", experts+1, experts, experts, experts+1, experts)
		dir := t.TempDir()
		input, output := filepath.Join(dir, "group.cc"), filepath.Join(dir, "group")
		if err := os.WriteFile(input, []byte(source.String()), 0600); err != nil {
			t.Fatal(err)
		}
		if b, err := exec.Command(compiler, "-std=c++11", input, "-o", output).CombinedOutput(); err != nil {
			t.Fatalf("compile: %v: %s", err, b)
		}
		b, err := exec.Command(output).Output()
		if err != nil {
			t.Fatal(err)
		}
		var got []int
		for _, f := range strings.Fields(string(b)) {
			var n int
			if _, err := fmt.Sscan(f, &n); err != nil {
				t.Fatal(err)
			}
			got = append(got, n)
		}
		want := []int{len(active), invalid}
		want = append(want, offsets...)
		want = append(want, pairs...)
		want = append(want, active...)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("experts=%d: got %v want %v", experts, got, want)
		}
	}
}

func TestShapeGroupingUpstreamPolicy(t *testing.T) {
	fixture, err := os.ReadFile("testdata/shape-group-upstream.cu")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(fixture)) != "ccc50abd6d4c3b632d65444f3e219f0988fb42fae0d4bc5a7ebd0ba2b3d7333d" {
		t.Fatal("upstream fixture hash changed")
	}
	body := string(fixture)
	path := os.Getenv("DS4_SHAPE_ORACLE")
	if path != "" {
		dir := filepath.Dir(path)
		const revision = "11e8cad4473716a41baf4d62e92ed9764a00fce6"
		b, err := exec.Command("git", "-C", dir, "rev-parse", revision+":ds4_cuda.cu").Output()
		if err != nil || strings.TrimSpace(string(b)) != "3543459df309d2b5bc4943b1c15a372fefe85e17" {
			t.Fatalf("frozen CUDA source pin: %v %s", err, b)
		}
		b, err = exec.Command("git", "-C", dir, "show", revision+":ds4_cuda.cu").Output()
		if err != nil {
			t.Fatal(err)
		}
		s := string(b)
		start, end := strings.Index(s, "__global__ static void moe_count_sorted_pairs_kernel("), strings.Index(s, "__global__ static void moe_build_expert_tile_offsets_kernel(")
		if start < 0 || end <= start {
			t.Fatal("missing upstream sorted-pair body")
		}
		if strings.TrimSpace(s[start:end]) != strings.TrimSpace(body) {
			t.Fatal("frozen fixture regeneration differs")
		}
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	source := "#include <cstdint>\n#define __global__\nstruct Dim {unsigned x;}; Dim blockIdx={0},blockDim={1},threadIdx={0}; uint32_t atomicAdd(uint32_t *p,uint32_t n){uint32_t old=*p;*p+=n;return old;}\n" + body + `\nint main(){int32_t ids[]={-1,384,383,0,256,383};uint32_t counts[384]={},offsets[385]={},cursors[384]={},pairs[6]={};for(unsigned p=0;p<6;p++){blockIdx.x=p;moe_count_sorted_pairs_kernel(counts,ids,6,384);}blockIdx.x=0;moe_prefix_sorted_pairs_kernel(offsets,cursors,counts,384);for(unsigned p=0;p<6;p++){blockIdx.x=p;moe_scatter_sorted_pairs_kernel(pairs,cursors,ids,6,384);}return !(offsets[384]==4 && counts[383]==2 && counts[256]==1 && pairs[0]==3 && pairs[1]==4 && pairs[2]==2 && pairs[3]==5);}`
	// This serial CPU execution qualifies the exclusion/count policy, not the
	// parallel scatter order on a GPU.
	source = strings.ReplaceAll(source, `\nint main`, "\nint main")
	d := t.TempDir()
	input, output := filepath.Join(d, "oracle.cc"), filepath.Join(d, "oracle")
	if err := os.WriteFile(input, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	if b, err := exec.Command(compiler, "-std=c++11", input, "-o", output).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, b)
	}
	if b, err := exec.Command(output).CombinedOutput(); err != nil {
		t.Fatalf("upstream policy: %v %s", err, b)
	}
}
