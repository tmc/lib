package dsv4cpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestPinnedCompressorPool(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#define DS4_NEG_INF (-1.0e30f)\n" + oracleFunction(t, s, "compressor_pool_decode_state") + "\n"
	c += `int main(void){int widths[]={128,512};int ratios[]={4,128};for(int h=0;h<2;h++)for(int r=0;r<2;r++){int dim=widths[h],ratio=ratios[r],lanes=ratio==4?2:1,width=lanes*dim,n=lanes*ratio*width;float *kv=malloc(n*4),*score=malloc(n*4),*out=malloc(dim*4);for(int i=0;i<n;i++){kv[i]=(float)(i%97-48)/17;score[i]=(float)(i%29-14)/7;}compressor_pool_decode_state(out,kv,score,dim,ratio);fwrite(out,4,dim,stdout);free(out);free(score);free(kv);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "compressor.c"), filepath.Join(dir, "compressor")
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
	if len(raw) != 1280*4 {
		t.Fatal("short compressor oracle")
	}
	offset := 0
	var maxAbs float64
	for _, width := range []int{128, 512} {
		for _, ratio := range []int{4, 128} {
			state, err := NewCompressor(width, ratio)
			if err != nil {
				t.Fatal(err)
			}
			for i := range state.kv {
				state.kv[i] = float32(i%97-48) / 17
				state.score[i] = float32(i%29-14) / 7
			}
			out := make([]float32, width)
			state.pool(out)
			for i, v := range out {
				want := math.Float32frombits(binary.LittleEndian.Uint32(raw[(offset+i)*4:]))
				delta := math.Abs(float64(v - want))
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || delta > 1e-6+1e-6*math.Abs(float64(want)) {
					t.Fatalf("width %d ratio %d index %d got %g want %g", width, ratio, i, v, want)
				}
				if delta > maxAbs {
					maxAbs = delta
				}
			}
			offset += width
		}
	}
	t.Logf("1280 actual pinned-C pooled outputs; max abs %g", maxAbs)
}

func TestCompressorTimeline(t *testing.T) {
	for _, ratio := range []int{4, 128} {
		c, err := NewCompressor(512, ratio)
		if err != nil {
			t.Fatal(err)
		}
		kv, score, dst := make([]float32, c.projectionWidth), make([]float32, c.projectionWidth), make([]float32, 512)
		for i := range dst {
			dst[i] = 123
		}
		for pos := 0; pos < ratio*3; pos++ {
			for i := range kv {
				kv[i] = float32(pos + 1)
			}
			emitted, err := c.Push(dst, kv, score, pos)
			if err != nil {
				t.Fatal(err)
			}
			if emitted != ((pos+1)%ratio == 0) {
				t.Fatal("wrong emit position", ratio, pos)
			}
			if emitted {
				want := float32(pos+1) - float32(ratio-1)/2
				if ratio == 4 && pos >= ratio {
					want -= float32(ratio) / 2
				}
				for i, v := range dst {
					if v != want {
						t.Fatalf("ratio %d pos %d index %d got %g want %g", ratio, pos, i, v, want)
					}
				}
			} else if pos < ratio-1 && dst[0] != 123 {
				t.Fatal("early output changed")
			}
		}
		if err := c.Reset(); err != nil {
			t.Fatal(err)
		}
		for _, v := range c.kv {
			if v != 0 {
				t.Fatal("reset retained KV")
			}
		}
		for _, v := range c.score {
			if v != negativeInfinity {
				t.Fatal("reset retained score")
			}
		}
		if c.position != 0 {
			t.Fatal("reset retained position")
		}
	}
}

func TestCompressorDomain(t *testing.T) {
	for _, ratio := range []int{0, 1, 3, 127} {
		if _, err := NewCompressor(512, ratio); err == nil {
			t.Fatal("accepted ratio", ratio)
		}
	}
	c, _ := NewCompressor(512, 4)
	kv, score, out := make([]float32, 1024), make([]float32, 1024), make([]float32, 512)
	if _, err := c.Push(out, kv, score, 1); err == nil {
		t.Fatal("accepted skipped position")
	}
	score[0] = float32(math.NaN())
	if _, err := c.Push(out, kv, score, 0); err == nil || c.position != 0 {
		t.Fatal("nonfinite update changed timeline")
	}
	var zero Compressor
	if zero.Reset() == nil {
		t.Fatal("zero compressor accepted")
	}
}

func ExampleNewCompressor() {
	c, err := NewCompressor(512, 4)
	fmt.Println(c != nil, err)
	// Output: true <nil>
}

func ExampleCompressor() {
	c, _ := NewCompressor(512, 4)
	out := make([]float32, 512)
	row, score := make([]float32, 1024), make([]float32, 1024)
	for pos := 0; pos < 4; pos++ {
		emitted, _ := c.Push(out, row, score, pos)
		fmt.Println(emitted)
	}
	// Output:
	// false
	// false
	// false
	// true
}

func ExampleCompressor_Push() {
	c, _ := NewCompressor(512, 128)
	emitted, err := c.Push(make([]float32, 512), make([]float32, 512), make([]float32, 512), 0)
	fmt.Println(emitted, err)
	// Output: false <nil>
}

func ExampleCompressor_Reset() {
	c, _ := NewCompressor(128, 4)
	fmt.Println(c.Reset())
	// Output: <nil>
}
