package dsv4cpu

import (
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// This characterizes the upstream scalar domain, not a value oracle. It does
// not assume native floating-point handling preserves subnormal inputs.
func TestPinnedSubnormalQuantizerDomain(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	start := strings.Index(s, "static void ds4_quantize_row_q8_K(")
	if start < 0 {
		t.Fatal("missing Q8_K quantizer")
	}
	end := strings.Index(s[start:], "\n}") + start + 2
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include <math.h>\n#include <stdlib.h>\n#define QK_K 256\ntypedef struct {float d;int8_t qs[256];int16_t bsums[16];} block_q8_K;\nstatic void ds4_die(const char *s){abort();}\n" + s[start:end] + `
int main(void){uint32_t bits;while(fread(&bits,4,1,stdin)==1){float x[256]={0},max=0,amax=0,id=0;memcpy(x,&bits,4);for(int j=0;j<256;j++){float a=fabsf(x[j]);if(a>amax){amax=a;max=x[j];}}if(amax!=0)id=-127/max;block_q8_K q;ds4_quantize_row_q8_K(x,&q,256);fwrite(&bits,4,1,stdout);fwrite(&max,4,1,stdout);fwrite(&amax,4,1,stdout);fwrite(&id,4,1,stdout);fwrite(&q.d,4,1,stdout);fwrite(q.qs,1,256,stdout);fwrite(q.bsums,2,16,stdout);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "domain.c"), filepath.Join(dir, "domain")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", file, "-lm", "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile %v %s", err, out)
	}
	bits := []uint32{1, 0x007fffff, 0x00800000, 0x02fdffff, 0x02fe0000, 0x02fe0001, 0x3f800000}
	input := make([]byte, len(bits)*4)
	for i, b := range bits {
		binary.LittleEndian.PutUint32(input[i*4:], b)
	}
	cmd := exec.Command(bin)
	cmd.Stdin = strings.NewReader(string(input))
	raw, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != len(bits)*308 {
		t.Fatal("domain probe output extent")
	}
	for i, b := range bits {
		r := raw[i*308:]
		if binary.LittleEndian.Uint32(r) != b {
			t.Fatal("input provenance")
		}
		x := make([]float32, 256)
		x[0] = math.Float32frombits(b)
		q := make([]Q8K, 1)
		goErr := QuantizeQ8K(q, x)
		t.Logf("bits=%08x Cmax=%08x Camax=%08x Cid=%08x Cscale=%08x Cq0=%d Csum0=%d Goerr=%v", b, binary.LittleEndian.Uint32(r[4:]), binary.LittleEndian.Uint32(r[8:]), binary.LittleEndian.Uint32(r[12:]), binary.LittleEndian.Uint32(r[16:]), int8(r[20]), int16(binary.LittleEndian.Uint16(r[276:])), goErr)
	}
}
