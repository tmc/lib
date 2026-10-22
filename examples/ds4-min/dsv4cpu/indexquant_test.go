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

func TestPinnedIndexerQuantization(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\nstatic void ds4_die(const char *s){exit(2);}\n"
	for _, name := range []string{"dsv4_e2m1fn_value_cpu", "dsv4_e2m1fn_dequant_cpu", "dsv4_hadamard128_inplace_cpu", "dsv4_fp4_act_quantize_row_inplace_cpu", "dsv4_indexer_qat_row_inplace_cpu"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `int main(void){float x[128];for(int r=0;r<64;r++){for(int i=0;i<128;i++)x[i]=(float)((r*7+i)%97-48)/17;dsv4_indexer_qat_row_inplace_cpu(x,128);fwrite(x,4,128,stdout);}for(int i=0;i<7;i++){float a=dsv4_e2m1fn_value_cpu(i),b=dsv4_e2m1fn_value_cpu(i+1),v=dsv4_e2m1fn_dequant_cpu((a+b)/2);fwrite(&v,4,1,stdout);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "index.c"), filepath.Join(dir, "index")
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
	if len(raw) != (8192+7)*4 {
		t.Fatal("short indexer oracle")
	}
	x := make([]float32, 8192)
	for i := range x {
		x[i] = float32(((i/128)*7+i%128)%97-48) / 17
	}
	if err := QuantizeIndexer(x); err != nil {
		t.Fatal(err)
	}
	for i, v := range x {
		want := binary.LittleEndian.Uint32(raw[i*4:])
		if math.Float32bits(v) != want {
			t.Fatalf("index %d got %g want %g", i, v, math.Float32frombits(want))
		}
	}
	values := [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}
	for i := 0; i < 7; i++ {
		v := e2m1Round((values[i] + values[i+1]) / 2)
		if math.Float32bits(v) != binary.LittleEndian.Uint32(raw[(8192+i)*4:]) {
			t.Fatalf("tie %d", i)
		}
	}
	t.Log("64x128 indexer QAT values and 7 midpoint ties bit-exact")
}

func TestIndexerQuantizationDomain(t *testing.T) {
	if QuantizeIndexer(make([]float32, 127)) == nil {
		t.Fatal("short row accepted")
	}
	x := make([]float32, 128)
	x[0] = float32(math.NaN())
	if QuantizeIndexer(x) == nil {
		t.Fatal("NaN accepted")
	}
	for i := range x {
		x[i] = math.MaxFloat32
	}
	if QuantizeIndexer(x) == nil {
		t.Fatal("Hadamard overflow accepted")
	}
}

func TestIndexerHadamardNaNDomain(t *testing.T) {
	x := make([]float32, 128)
	state := uint32(1)
	for i := range x {
		state = state*1664525 + 1013904223
		x[i] = math.MaxFloat32
		if state>>31 != 0 {
			x[i] = -x[i]
		}
	}
	probe := append([]float32(nil), x...)
	for stride := 1; stride < 128; stride *= 2 {
		for base := 0; base < 128; base += 2 * stride {
			for i := 0; i < stride; i++ {
				a, b := probe[base+i], probe[base+stride+i]
				probe[base+i] = a + b
				probe[base+stride+i] = a - b
			}
		}
	}
	nans, infs := 0, 0
	for _, v := range probe {
		if math.IsNaN(float64(v)) {
			nans++
		}
		if math.IsInf(float64(v), 0) {
			infs++
		}
	}
	if nans == 0 || infs != 0 {
		t.Fatalf("counterexample did not isolate NaN: NaN=%d Inf=%d", nans, infs)
	}
	if err := QuantizeIndexer(x); err == nil {
		t.Fatal("accepted NaN-only Hadamard overflow")
	}
	t.Logf("finite inputs produce %d NaN and zero infinity outputs", nans)
}

func ExampleQuantizeIndexer() {
	x := make([]float32, 128)
	_ = QuantizeIndexer(x)
	fmt.Println(x[0])
	// Output: 0
}
