package cudafuse

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

const mhcHostShim = `
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#define __device__
#define __global__
struct Dim {unsigned int x,y,z;};
Dim blockIdx={0,0,0},threadIdx={0,0,0},blockDim={256,1,1},gridDim={1,1,1};
`
const mhcHostMain = `
int main() {
    int batch;
    if (std::fread(&batch,4,1,stdin)!=1) return 2;
    std::vector<float> logits(batch*16),x(batch*MHC_WIDTH),post(batch*4),residual(batch*4*MHC_WIDTH);
    for (auto *v : {&logits,&x,&post,&residual}) if (std::fread(v->data(),4,v->size(),stdin)!=v->size()) return 3;
    float poison;unsigned int bits=0x7fc01234;std::memcpy(&poison,&bits,4);
    std::vector<float> mix(batch*16+2,poison),out(batch*4*MHC_WIDTH+2,poison);
    for (int i=0;i<batch;++i) {blockIdx.x=i/256;threadIdx.x=i%256;mhc_sinkhorn_fp32(logits.data(),mix.data()+1,batch);}
    int count=batch*4*MHC_WIDTH;
    for (int i=0;i<count;++i) {blockIdx.x=i/256;threadIdx.x=i%256;mhc_post_fp32(x.data(),post.data(),residual.data(),mix.data()+1,out.data()+1,count);}
    std::fwrite(mix.data(),4,mix.size(),stdout);std::fwrite(out.data(),4,out.size(),stdout);
    return std::ferror(stdout)?4:0;
}
`

func mhcBuildHost(t *testing.T, s [2]FusedKernel) string {
	t.Helper()
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Fatal("mhc source emulation requires clang++")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "mhc.cc")
	bin := filepath.Join(dir, "mhc-host")
	if err := os.WriteFile(source, []byte(mhcHostShim+s[0].Source+s[1].Source+mhcHostMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-ffp-contract=off", source, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("mhc emulation build: %v (context: %v)\n%s", err, ctx.Err(), out)
	}
	return bin
}
func mhcRunHost(t *testing.T, bin string, f mhcFixture) mhcAnswer {
	t.Helper()
	var in bytes.Buffer
	for _, v := range []any{int32(f.config.Batch), f.logits, f.x, f.post, f.residual} {
		if err := binary.Write(&in, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin)
	cmd.Stdin = &in
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("mhc emulation run: %v (context: %v)", err, ctx.Err())
	}
	r := bytes.NewReader(out)
	a := mhcAnswer{}
	for i, n := range []int{f.config.Batch * 16, f.config.Batch * 4 * f.config.Width} {
		all := make([]float32, n+2)
		if err := binary.Read(r, binary.LittleEndian, all); err != nil {
			t.Fatal(err)
		}
		if math.Float32bits(all[0]) != 0x7fc01234 || math.Float32bits(all[n+1]) != 0x7fc01234 {
			t.Fatal("mhc guard corrupted")
		}
		if i == 0 {
			a.mix = all[1 : n+1]
		} else {
			a.out = all[1 : n+1]
		}
	}
	if r.Len() != 0 {
		t.Fatal(fmt.Sprintf("extra mhc emulation output: %d bytes", r.Len()))
	}
	return a
}
