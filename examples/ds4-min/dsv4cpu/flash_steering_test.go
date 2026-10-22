package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPinnedFlashSteering(t *testing.T) {
	source := string(pinnedCPUOracle(t))
	start := strings.LastIndex(source, "static bool cpu_directional_steering_enabled(")
	if start < 0 {
		t.Fatal("missing steering definition")
	}
	source = "\n" + source[start:]
	c := "#include <stdbool.h>\n#include <stdint.h>\n#include <stdio.h>\n#define DS4_N_EMBD 4096\n"
	c += oracleFunction(t, source, "cpu_directional_steering_enabled") + "\n"
	c += oracleFunction(t, source, "cpu_directional_steering_project_rows") + "\n"
	c += `int main(void){float x[4096],dirs[43*4096];for(int i=0;i<43*4096;i++)dirs[i]=(float)(i%23-11)/1024;for(int i=0;i<4096;i++)x[i]=(float)(i%97-48)/17;cpu_directional_steering_project_rows(x,dirs,7,1,0.75f);fwrite(x,4,4096,stdout);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "steer.c"), filepath.Join(dir, "steer")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", file, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	raw, err := exec.Command(bin).Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4096*4 {
		t.Fatal("short steering oracle")
	}
	s := codecFixture(t)
	defer s.close()
	if err := s.reset(); err != nil {
		t.Fatal(err)
	}
	if s.position != 0 {
		t.Fatal("steering fixture position is not empty")
	}
	for il, l := range s.layers {
		if l.raw.count != 0 || len(l.compressed) != 0 || len(l.indexRows) != 0 || (l.attention != nil && l.attention.position != 0) || (l.index != nil && l.index.position != 0) {
			t.Fatal("steering fixture caches not empty", il)
		}
	}
	dirs := make([]float32, 43*4096)
	for i := range dirs {
		dirs[i] = float32(i%23-11) / 1024
	}
	if err := s.setSteering(dirs, 0.75, 0.5); err != nil {
		t.Fatal(err)
	}
	dirs[7*4096] = 100 // Setter owns a copy.
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%97-48) / 17
	}
	if err := s.projectSteering(x, 7, s.attentionScale); err != nil {
		t.Fatal(err)
	}
	for i, v := range x {
		want := binary.LittleEndian.Uint32(raw[i*4:])
		if math.Float32bits(v) != want {
			t.Fatalf("steering index %d got %g want %g", i, v, math.Float32frombits(want))
		}
	}
	frame := saveFixture(t, s)
	dst, err := s.model.newState(s.context)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.close()
	if err := dst.load(bytes.NewReader(frame)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(frame, saveFixture(t, dst)) {
		t.Fatal("steering codec differs")
	}
	snap, err := s.snapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.close()
	s.steering[0]++
	if s.steering[0] == snap.state.steering[0] {
		t.Fatal("snapshot aliases directions")
	}
}
