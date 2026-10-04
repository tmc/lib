package cudafuse

import (
	"bytes"
	"context"
	"encoding/binary"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// Select the nearest member of the BF16 value set by FP64 distances, with
// explicit even parity at a midpoint. This oracle does not use the device's
// integer rounding-bias algorithm. The conceptual value beyond the largest
// finite BF16 is 2^128; rounding to that code publishes signed infinity.
func mhcBF16Reference(x float32) uint16 {
	sign := uint16(0)
	v := float64(x)
	if math.Signbit(v) {
		sign = 0x8000
		v = -v
	}
	if math.IsNaN(v) {
		return sign | 0x7fc0
	}
	if math.IsInf(v, 0) {
		return sign | 0x7f80
	}
	hi := sort.Search(0x7f80, func(i int) bool { return float64(math.Float32frombits(uint32(i)<<16)) >= v })
	if hi == 0 {
		return sign
	}
	lo := hi - 1
	upper := math.Ldexp(1, 128)
	if hi < 0x7f80 {
		upper = float64(math.Float32frombits(uint32(hi) << 16))
	}
	lower := float64(math.Float32frombits(uint32(lo) << 16))
	dl, du := v-lower, upper-v
	if dl < du || dl == du && lo%2 == 0 {
		return sign | uint16(lo)
	}
	return sign | uint16(hi)
}

func mhcBuildExtensionHost(t *testing.T, source, main string) string {
	t.Helper()
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "extension.cc")
	bin := filepath.Join(dir, "extension-host")
	if err := os.WriteFile(path, []byte(mhcHostShim+source+main), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-ffp-contract=off", path, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("extension build: %v context=%v\n%s", err, ctx.Err(), out)
	}
	return bin
}

func mhcRunExtensionHost(t *testing.T, bin string, input []byte) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin)
	cmd.Stdin = bytes.NewReader(input)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("extension run: %v context=%v", err, ctx.Err())
	}
	return out
}

const mhcConversionMain = `
int main() {
    unsigned int bits;
    while (std::fread(&bits,4,1,stdin)==1) {
        float x;std::memcpy(&x,&bits,4);unsigned short b=mhc_round_bf16(x);
        std::fwrite(&b,2,1,stdout);
        unsigned short raw=(unsigned short)bits;float expanded=mhc_expand_bf16(raw);
        unsigned int e;std::memcpy(&e,&expanded,4);std::fwrite(&e,4,1,stdout);
    }
    return std::ferror(stdout)?4:0;
}
`

func TestMHCBF16StorageConversion(t *testing.T) {
	sources, err := MHCBF16Source(MHCConfig{1, 1, 1, 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	var input bytes.Buffer
	var bits []uint32
	// Every 16-bit storage code is checked for exact expansion; sampled low
	// bits around every finite BF16 midpoint cover both parity and sign.
	for high := uint32(0); high < 65536; high++ {
		bits = append(bits, high)
		if high&0x7f80 != 0x7f80 {
			for _, low := range []uint32{0, 0x7fff, 0x8000, 0x8001, 0xffff} {
				bits = append(bits, high<<16|low)
			}
		}
	}
	for _, b := range bits {
		if err := binary.Write(&input, binary.LittleEndian, b); err != nil {
			t.Fatal(err)
		}
	}
	bin := mhcBuildExtensionHost(t, sources[1].Source, mhcConversionMain)
	r := bytes.NewReader(mhcRunExtensionHost(t, bin, input.Bytes()))
	for _, b := range bits {
		var round uint16
		var expanded uint32
		if err := binary.Read(r, binary.LittleEndian, &round); err != nil {
			t.Fatal(err)
		}
		if err := binary.Read(r, binary.LittleEndian, &expanded); err != nil {
			t.Fatal(err)
		}
		if expanded != uint32(uint16(b))<<16 {
			t.Fatalf("storage expansion changed bits %08x", b)
		}
		x := math.Float32frombits(b)
		if mhcFinite(x) && round != mhcBF16Reference(x) {
			t.Fatalf("RNE mismatch input=%08x got=%04x want=%04x", b, round, mhcBF16Reference(x))
		}
	}
	if r.Len() != 0 {
		t.Fatal("extra conversion output")
	}
	// Recompile a genuine truncation mutant and use a midpoint above odd code
	// 0x3f81. RNE must choose 0x3f82, while truncation stores 0x3f81.
	source := strings.Replace(sources[1].Source, "(v.u+0x7fffu+((v.u>>16)&1u))>>16", "v.u>>16", 1)
	if source == sources[1].Source {
		t.Fatal("conversion mutant did not match")
	}
	var in bytes.Buffer
	_ = binary.Write(&in, binary.LittleEndian, uint32(0x3f818000))
	raw := mhcRunExtensionHost(t, mhcBuildExtensionHost(t, source, mhcConversionMain), in.Bytes())
	if len(raw) != 6 || binary.LittleEndian.Uint16(raw[:2]) == mhcBF16Reference(math.Float32frombits(0x3f818000)) {
		t.Fatal("truncation mutant survived")
	}
}

func mhcBF16Fixture(f mhcFixture) (mhcFixture, []uint16, []uint16) {
	q := f
	q.x = make([]float32, len(f.x))
	q.residual = make([]float32, len(f.residual))
	x := make([]uint16, len(f.x))
	r := make([]uint16, len(f.residual))
	for i, v := range f.x {
		x[i] = mhcBF16Reference(v)
		q.x[i] = mhcBF16Expand(x[i])
	}
	for i, v := range f.residual {
		r[i] = mhcBF16Reference(v)
		q.residual[i] = mhcBF16Expand(r[i])
	}
	return q, x, r
}

const mhcBF16HostMain = `
int main() {
    int batch;if (std::fread(&batch,4,1,stdin)!=1) return 2;
    std::vector<float> logits(batch*16),post(batch*4);
    std::vector<unsigned short> x(batch*MHC_WIDTH),residual(batch*4*MHC_WIDTH);
    if (std::fread(logits.data(),4,logits.size(),stdin)!=logits.size()||std::fread(x.data(),2,x.size(),stdin)!=x.size()||std::fread(post.data(),4,post.size(),stdin)!=post.size()||std::fread(residual.data(),2,residual.size(),stdin)!=residual.size()) return 3;
    float poison;unsigned int p=0x7fc01234;std::memcpy(&poison,&p,4);
    std::vector<float> mix(batch*16+2,poison);std::vector<unsigned short> out(batch*4*MHC_WIDTH+2,0x7fc1);
    for(int i=0;i<batch;++i){blockIdx.x=i/256;threadIdx.x=i%256;mhc_sinkhorn_fp32(logits.data(),mix.data()+1,batch);}
    int count=batch*4*MHC_WIDTH;
    for(int i=0;i<count;++i){blockIdx.x=i/256;threadIdx.x=i%256;mhc_post_bf16(x.data(),post.data(),residual.data(),mix.data()+1,out.data()+1,count);}
    std::fwrite(mix.data(),4,mix.size(),stdout);std::fwrite(out.data(),2,out.size(),stdout);return std::ferror(stdout)?4:0;
}
`

func mhcCompareBF16(f mhcFixture, got mhcAnswer, source, target string) error {
	want := mhcOracle(f)
	if err := cudatest.CompareCUDAOutputFP32(got.mix, want.mix, mhcPolicy(f, source, target, len(got.mix))); err != nil {
		return err
	}
	p := mhcPolicy(f, source, target, len(got.out))
	p.MathMode = "FP32 accumulation --fmad=false; BF16 final RNE; separate conversion bound"
	p.Relative += 1.0 / 256
	return cudatest.CompareCUDAOutputFP32(got.out, want.out, p)
}

func TestMHCBF16TwoStageReference(t *testing.T) {
	for _, f := range mhcFixtures() {
		if f.config.Repeat != 10 {
			continue
		}
		t.Run(f.name, func(t *testing.T) {
			q, x, residual := mhcBF16Fixture(f)
			s, err := MHCBF16Source(q.config)
			if err != nil {
				t.Fatal(err)
			}
			var in bytes.Buffer
			for _, v := range []any{int32(q.config.Batch), q.logits, x, q.post, residual} {
				if err := binary.Write(&in, binary.LittleEndian, v); err != nil {
					t.Fatal(err)
				}
			}
			r := bytes.NewReader(mhcRunExtensionHost(t, mhcBuildExtensionHost(t, s[0].Source+s[1].Source, mhcBF16HostMain), in.Bytes()))
			mix := make([]float32, len(q.logits)+2)
			out := make([]uint16, len(q.residual)+2)
			if err := binary.Read(r, binary.LittleEndian, mix); err != nil {
				t.Fatal(err)
			}
			if err := binary.Read(r, binary.LittleEndian, out); err != nil {
				t.Fatal(err)
			}
			if math.Float32bits(mix[0]) != 0x7fc01234 || math.Float32bits(mix[len(mix)-1]) != 0x7fc01234 || out[0] != 0x7fc1 || out[len(out)-1] != 0x7fc1 || r.Len() != 0 {
				t.Fatal("BF16 guards/output extent invalid")
			}
			a := mhcAnswer{mix: mix[1 : len(mix)-1], out: make([]float32, len(q.residual))}
			for i, v := range out[1 : len(out)-1] {
				a.out[i] = mhcBF16Expand(v)
			}
			if err := mhcCompareBF16(q, a, s[0].Source+s[1].Source, "host-emulation"); err != nil {
				t.Fatal(err)
			}
		})
	}
}

const mhcFusedHostMain = `
int main(){
    int batch;if(std::fread(&batch,4,1,stdin)!=1)return 2;
    std::vector<float> logits(batch*16),x(batch*MHC_WIDTH),post(batch*4),residual(batch*4*MHC_WIDTH);
    for(auto *v:{&logits,&x,&post,&residual})if(std::fread(v->data(),4,v->size(),stdin)!=v->size())return 3;
    float poison;unsigned int p=0x7fc01234;std::memcpy(&poison,&p,4);std::vector<float> out(batch*4*MHC_WIDTH+2,poison);
    int count=batch*4*MHC_WIDTH;for(int i=0;i<count;++i){blockIdx.x=i/256;threadIdx.x=i%256;mhc_fused_fp32(logits.data(),x.data(),post.data(),residual.data(),out.data()+1,count);}
    std::fwrite(out.data(),4,out.size(),stdout);return std::ferror(stdout)?4:0;
}
`

func TestMHCFusedReference(t *testing.T) {
	for _, f := range mhcFixtures() {
		if f.config.Repeat != 10 {
			continue
		}
		t.Run(f.name, func(t *testing.T) {
			s, err := MHCFusedSource(f.config)
			if err != nil {
				t.Fatal(err)
			}
			var in bytes.Buffer
			for _, v := range []any{int32(f.config.Batch), f.logits, f.x, f.post, f.residual} {
				if err := binary.Write(&in, binary.LittleEndian, v); err != nil {
					t.Fatal(err)
				}
			}
			r := bytes.NewReader(mhcRunExtensionHost(t, mhcBuildExtensionHost(t, s.Source, mhcFusedHostMain), in.Bytes()))
			out := make([]float32, len(f.residual)+2)
			if err := binary.Read(r, binary.LittleEndian, out); err != nil {
				t.Fatal(err)
			}
			if math.Float32bits(out[0]) != 0x7fc01234 || math.Float32bits(out[len(out)-1]) != 0x7fc01234 || r.Len() != 0 {
				t.Fatal("fused output guards invalid")
			}
			want := mhcOracle(f)
			if err := cudatest.CompareCUDAOutputFP32(out[1:len(out)-1], want.out, mhcPolicy(f, s.Source, "host-emulation", len(want.out))); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestMHCBF16TailsAndAliasing(t *testing.T) {
	for _, width := range []int{1, 31, 32, 33, 129} {
		c := MHCConfig{4, width, 10, 1e-6}
		ranges := []mhcRange{{4100, uint64(c.Batch * 16 * 4), c.Batch * 16}, {8194, uint64(c.Batch * width * 2), c.Batch * width}, {16388, uint64(c.Batch * 4 * 4), c.Batch * 4}, {65538, uint64(c.Batch * 4 * width * 2), c.Batch * 4 * width}}
		if err := mhcValidateBF16Ranges(c, ranges); err != nil {
			t.Fatal(err)
		}
		for i := range ranges {
			for j := 0; j < i; j++ {
				bad := append([]mhcRange(nil), ranges...)
				bad[i].ptr = bad[j].ptr
				if mhcValidateBF16Ranges(c, bad) == nil {
					t.Fatal("BF16 overlapping view admitted")
				}
			}
		}
		for i := range ranges {
			bad := append([]mhcRange(nil), ranges...)
			bad[i].length--
			if mhcValidateBF16Ranges(c, bad) == nil {
				t.Fatal("BF16 short view admitted")
			}
		}
	}
}

func TestMHCFusedOrientationMutant(t *testing.T) {
	for _, f := range mhcFixtures() {
		if f.name != "nonsymmetric-w33-r10" {
			continue
		}
		s, err := MHCFusedSource(f.config)
		if err != nil {
			t.Fatal(err)
		}
		original := s.Source
		s.Source = strings.Replace(s.Source, "a[m*4+n]", "a[n*4+m]", 1)
		if original == s.Source {
			t.Fatal("fused mutation did not match")
		}
		var in bytes.Buffer
		for _, v := range []any{int32(f.config.Batch), f.logits, f.x, f.post, f.residual} {
			if err := binary.Write(&in, binary.LittleEndian, v); err != nil {
				t.Fatal(err)
			}
		}
		r := bytes.NewReader(mhcRunExtensionHost(t, mhcBuildExtensionHost(t, s.Source, mhcFusedHostMain), in.Bytes()))
		all := make([]float32, len(f.residual)+2)
		if err := binary.Read(r, binary.LittleEndian, all); err != nil {
			t.Fatal(err)
		}
		if math.Float32bits(all[0]) != 0x7fc01234 || math.Float32bits(all[len(all)-1]) != 0x7fc01234 || r.Len() != 0 {
			t.Fatal("fused mutant failed publication")
		}
		want := mhcOracle(f)
		if err := cudatest.CompareCUDAOutputFP32(all[1:len(all)-1], want.out, mhcPolicy(f, s.Source, "host-emulation", len(want.out))); err == nil {
			t.Fatal("fused orientation mutant survived")
		}
		return
	}
	t.Fatal("missing fused control fixture")
}
