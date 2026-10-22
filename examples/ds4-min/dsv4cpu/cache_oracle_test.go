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

func TestPinnedHalfOracle(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	start := strings.Index(s, "static inline uint16_t f32_to_f16(")
	if start < 0 {
		t.Fatal("missing half helper")
	}
	end := strings.Index(s[start:], "\n}") + start + 2
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#undef __ARM_NEON\n" + s[start:end] + "\nint main(void){uint32_t b;while(fread(&b,4,1,stdin)==1){float f;memcpy(&f,&b,4);uint16_t h=f32_to_f16(f);fwrite(&h,2,1,stdout);}return 0;}"
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "half.c"), filepath.Join(dir, "half")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", file, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile %v %s", err, out)
	}
	bits := []uint32{0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0xffc00001, 0x477fe000, 0x477ff000, 0x33000000, 0x33800000, 0x387fc000, 0x38800000}
	for e := uint32(100); e < 145; e++ {
		for _, m := range []uint32{0, 1, 0xfff, 0x1000, 0x1001, 0x2000, 0x201000, 0x7fffff} {
			bits = append(bits, e<<23|m, e<<23|m|0x80000000)
		}
	}
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
	if len(raw) != len(bits)*2 {
		t.Fatal("short half oracle")
	}
	for i, b := range bits {
		want := binary.LittleEndian.Uint16(raw[i*2:])
		if got := toHalf(math.Float32frombits(b)); got != want {
			t.Fatalf("half bits=%08x got=%04x want=%04x", b, got, want)
		}
	}
	t.Logf("half extrema/ties cases=%d", len(bits))
}
