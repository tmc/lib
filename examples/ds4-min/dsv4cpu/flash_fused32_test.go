package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestFused32LiteralC(t *testing.T) {
	// The tiny third term selects opposite sides of an exact float32 midpoint.
	// A float64 FMA followed by a float32 conversion loses that distinction.
	a := float32(1 + math.Ldexp(1, -12))
	triples := [][3]float32{
		{a, a, float32(math.Ldexp(1, -80))},
		{a, a, -float32(math.Ldexp(1, -80))},
		{-a, a, float32(math.Ldexp(1, -80))},
		{-a, a, -float32(math.Ldexp(1, -80))},
		{math.SmallestNonzeroFloat32, .5, 0},
		{math.SmallestNonzeroFloat32, .5, math.SmallestNonzeroFloat32},
		{math.MaxFloat32, 2, -math.MaxFloat32},
	}
	seed := uint32(19)
	next := func() float32 {
		seed = seed*1664525 + 1013904223
		return math.Float32frombits((seed & 0x807fffff) | (((seed>>23)%250 + 2) << 23))
	}
	for i := 0; i < 2048; i++ {
		triples = append(triples, [3]float32{next(), next(), next()})
	}
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "fused.c"), filepath.Join(dir, "fused")
	c := `#include <math.h>
#include <stdio.h>
int main(void){float x[3];while(fread(x,4,3,stdin)==3){float y=fmaf(x[0],x[1],x[2]);if(fwrite(&y,4,1,stdout)!=1)return 2;}return ferror(stdin)?2:0;}`
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O3", "-ffp-contract=off", file, "-lm", "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile literal fmaf: %v %s", err, out)
	}
	var in bytes.Buffer
	if err := binary.Write(&in, binary.LittleEndian, triples); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin)
	cmd.Stdin = &in
	raw, err := cmd.Output()
	if err != nil || len(raw) != len(triples)*4 {
		t.Fatalf("literal fmaf: %v bytes=%d", err, len(raw))
	}
	for i, v := range triples {
		want := binary.LittleEndian.Uint32(raw[i*4:])
		got := math.Float32bits(fused32(v[0], v[1], v[2]))
		if got != want {
			t.Fatalf("fused32 case %d got=%08x want=%08x", i, got, want)
		}
	}
}
