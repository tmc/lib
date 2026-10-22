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

func TestPinnedKVQuantization(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <math.h>\n"
	for _, name := range []string{"dsv4_e4m3fn_value_cpu", "dsv4_e4m3fn_dequant_cpu", "dsv4_fp8_kv_quantize_row_inplace_cpu"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `int main(void){float x[512];for(int mode=0;mode<3;mode++){for(int i=0;i<512;i++){x[i]=(float)(i%97-48)/17;if(mode==1)x[i]*=0.00001f;if(mode==2)x[i]*=10000.0f;}dsv4_fp8_kv_quantize_row_inplace_cpu(x,512,64);fwrite(x,4,512,stdout);}for(int i=0;i<126;i++){float a=dsv4_e4m3fn_value_cpu(i),b=dsv4_e4m3fn_value_cpu(i+1),v=dsv4_e4m3fn_dequant_cpu((a+b)/2);fwrite(&v,4,1,stdout);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "kv.c"), filepath.Join(dir, "kv")
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
	if len(raw) != (3*512+126)*4 {
		t.Fatal("short KV oracle")
	}
	for mode := 0; mode < 3; mode++ {
		x := make([]float32, 512)
		for i := range x {
			x[i] = float32(i%97-48) / 17
			if mode == 1 {
				x[i] *= 0.00001
			}
			if mode == 2 {
				x[i] *= 10000
			}
		}
		if err := QuantizeKV(x, 64); err != nil {
			t.Fatal(err)
		}
		for i, v := range x {
			want := binary.LittleEndian.Uint32(raw[(mode*512+i)*4:])
			if math.Float32bits(v) != want {
				t.Fatalf("mode %d index %d got %g want %g", mode, i, v, math.Float32frombits(want))
			}
		}
	}
	for i := 0; i < 126; i++ {
		v := e4m3Round((e4m3Value(i) + e4m3Value(i+1)) / 2)
		if math.Float32bits(v) != binary.LittleEndian.Uint32(raw[(3*512+i)*4:]) {
			t.Fatalf("tie %d", i)
		}
	}
	t.Log("1536 KV values including rotary tail and 126 E4M3 midpoint ties bit-exact")
}

func TestKVQuantizationDomain(t *testing.T) {
	for _, shape := range []struct{ width, rotary int }{{0, 0}, {512, -1}, {512, 513}, {512, 63}} {
		if QuantizeKV(make([]float32, shape.width), shape.rotary) == nil {
			t.Fatal("accepted shape", shape)
		}
	}
	x := make([]float32, 512)
	x[0] = float32(math.NaN())
	if QuantizeKV(x, 64) == nil {
		t.Fatal("accepted NaN")
	}
}

func ExampleQuantizeKV() {
	x := make([]float32, 512)
	x[0] = 1
	x[511] = 0.123
	_ = QuantizeKV(x, 64)
	fmt.Println(x[0], x[511])
	// Output: 1 0.123
}
