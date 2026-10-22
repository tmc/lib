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

func TestPinnedRoPE(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := "#include <stdbool.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <math.h>\n"
	for _, name := range []string{"rope_yarn_ramp", "rope_yarn_corr_dim", "rope_yarn_corr_dims", "rope_tail_ext_inplace"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `int main(void){float x[64*512];int positions[]={0,1,64,2047,2048,2049};for(int mode=0;mode<2;mode++)for(int inv=0;inv<2;inv++)for(int p=0;p<6;p++){for(int i=0;i<64*512;i++)x[i]=(float)(i%97-48)/17;float scale=mode?0.0625f:1.0f;float factor=mode?1.0f/(1.0f+0.1f*logf(1.0f/scale)):1.0f;rope_tail_ext_inplace(x,64,512,64,positions[p],mode?65536:0,mode?160000.0f:10000.0f,scale,mode?1.0f:0.0f,factor,32.0f,1.0f,inv);fwrite(x,4,64*512,stdout);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "rope.c"), filepath.Join(dir, "rope")
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
	const width = 64 * 512
	if len(raw) != 24*width*4 {
		t.Fatal("short rotary oracle")
	}
	var maxAbs float64
	index := 0
	for mode := 0; mode < 2; mode++ {
		for inv := 0; inv < 2; inv++ {
			for _, pos := range []int{0, 1, 64, 2047, 2048, 2049} {
				x := make([]float32, width)
				for i := range x {
					x[i] = float32(i%97-48) / 17
				}
				r := RoPE{Base: 10000, Scale: 1}
				if mode == 1 {
					r = RoPE{Base: 160000, Scale: 1.0 / 16, OriginalContext: 65536, BetaFast: 32, BetaSlow: 1}
				}
				if err := r.Rotate(x, 512, 64, pos, inv != 0); err != nil {
					t.Fatal(err)
				}
				for i, v := range x {
					want := math.Float32frombits(binary.LittleEndian.Uint32(raw[(index*width+i)*4:]))
					delta := math.Abs(float64(v - want))
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || delta > 1e-6+1e-6*math.Abs(float64(want)) {
						t.Fatalf("mode %d inverse %d pos %d index %d got %g want %g delta %g", mode, inv, pos, i, v, want, delta)
					}
					if delta > maxAbs {
						maxAbs = delta
					}
				}
				index++
			}
		}
	}
	t.Logf("24 real 64x512 dense/compressed forward/inverse cases; max abs %g", maxAbs)
}

func ExampleRoPE() {
	r := RoPE{Base: 10000, Scale: 1}
	x := []float32{1, 0}
	_ = r.Rotate(x, 2, 2, 0, false)
	fmt.Println(x)
	// Output: [1 0]
}

func TestRoPEDomain(t *testing.T) {
	r := RoPE{Base: 10000, Scale: 1}
	for _, c := range []struct{ width, rotary, pos int }{{0, 2, 0}, {2, 0, 0}, {2, 3, 0}, {2, 1, 0}, {2, 2, -1}, {3, 2, 0}} {
		if r.Rotate([]float32{1, 0}, c.width, c.rotary, c.pos, false) == nil {
			t.Fatal("accepted shape", c)
		}
	}
	for _, bad := range []RoPE{{Base: 1, Scale: 1}, {Base: 10000, Scale: 0}, {Base: float32(math.Inf(1)), Scale: 1}, {Base: 160000, Scale: 1.0 / 16}} {
		if bad.Rotate([]float32{1, 0}, 2, 2, 0, false) == nil {
			t.Fatal("accepted config", bad)
		}
	}
	if r.Rotate([]float32{float32(math.NaN()), 0}, 2, 2, 0, false) == nil {
		t.Fatal("accepted NaN")
	}
}

func ExampleRoPE_Rotate() {
	r := RoPE{Base: 10000, Scale: 1}
	x := []float32{1, 0}
	_ = r.Rotate(x, 2, 2, 1, false)
	fmt.Printf("%.3f %.3f\n", x[0], x[1])
	// Output: 0.540 0.841
}
