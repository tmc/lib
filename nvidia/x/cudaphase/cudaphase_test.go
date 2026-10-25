package cudaphase

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestSummarize(t *testing.T) {
	// Three blocks of 4 slots: start, main end, end, SM. Block 1 never
	// stamped its main end; block 3 never ran.
	buf := []uint64{
		1000, 3000, 4000, 7,
		1256, 0, 5256, 9,
		1512, 4512, 4768, 7,
		0, 0, 0, 0,
	}
	phases := []Phase{{"main", 0, 1}, {"tail", 1, 2}, {"total", 0, 2}}
	got, err := Summarize("k", buf, 4, phases, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := Summary{
		Name: "k", Blocks: 3, SpanUS: 4.256, QuantumUS: 0.256, SMs: 2,
		Phases: []PhaseStat{
			{"main", 2, 2, 2, 3},
			{"tail", 2, 0.256, 0.256, 1},
			{"total", 3, 3.256, 3, 4},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Summarize =\n%+v\nwant\n%+v", got, want)
	}
}

func TestSummarizeErrors(t *testing.T) {
	for _, tt := range []struct {
		name   string
		buf    []uint64
		slots  int
		phases []Phase
		want   string
	}{
		{"ragged", make([]uint64, 5), 4, nil, "not a whole number of 4-slot rows"},
		{"no slots", nil, 0, nil, "not a whole number"},
		{"slot range", make([]uint64, 4), 4, []Phase{{"p", 0, 4}}, "phase p: slots 0, 4 out of range"},
	} {
		if _, err := Summarize("k", tt.buf, tt.slots, tt.phases, -1); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: error = %v, want %q", tt.name, err, tt.want)
		}
	}
}

// TestPythonMatchesGo checks cudaphase.py, which runs inside a judge,
// against Summarize on random buffers with unset stamps and idle blocks.
func TestPythonMatchesGo(t *testing.T) {
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3")
	}
	r := rand.New(rand.NewPCG(5, 6))
	phases := []Phase{{"a", 0, 1}, {"b", 1, 3}, {"c", 0, 3}, {"d", 2, 2}}
	var script strings.Builder
	script.WriteString(Python)
	var bufs [][]uint64
	for c := 0; c < 20; c++ {
		blocks := 1 + r.IntN(40)
		buf := make([]uint64, blocks*6)
		for b := 0; b < blocks; b++ {
			if r.IntN(8) == 0 {
				continue // never ran
			}
			t0 := 1_000_000 + uint64(r.IntN(5000))*256
			row := buf[b*6:]
			row[0] = t0
			row[1] = t0 + uint64(r.IntN(100))*256
			row[2] = row[1] + uint64(r.IntN(100))*256
			row[3] = row[2] + uint64(r.IntN(100))*256
			row[5] = uint64(r.IntN(9)) // 0: SM never stamped
			if r.IntN(5) == 0 {
				row[1+r.IntN(3)] = 0 // a stamp never written
			}
		}
		bufs = append(bufs, buf)
		fmt.Fprintf(&script, "print(cudaphase_summary(%q, %s, [(\"a\", 0, 1), (\"b\", 1, 3), (\"c\", 0, 3), (\"d\", 2, 2)], slots=6, sm_slot=5))\n",
			fmt.Sprint("case", c), strings.Join(strings.Fields(fmt.Sprint(buf)), ", "))
	}
	out, err := exec.Command(py, "-c", script.String()).CombinedOutput()
	if err != nil {
		t.Fatalf("python3: %v\n%s", err, out)
	}
	got, err := Parse(strings.NewReader(string(out)))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(bufs) {
		t.Fatalf("python printed %d summaries, want %d\n%s", len(got), len(bufs), out)
	}
	for c, buf := range bufs {
		want, err := Summarize(fmt.Sprint("case", c), buf, 6, phases, 5)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got[c], want) {
			t.Errorf("case %d: python\n%+v\ngo\n%+v", c, got[c], want)
		}
	}
}

// TestKernel marks phases from a kernel under cudahostemu, which compiles
// the header's host path, and summarizes the buffer.
func TestKernel(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "cudaphase.h"), []byte(Header), 0o644); err != nil {
		t.Fatal(err)
	}
	const src = `
#include "cudaphase.h"
__global__ void k(unsigned long long* trace, int skip) {
    CUDAPHASE_MARK(trace, 0, cudaphase_now());
    CUDAPHASE_MARK(trace, 4, threadIdx.x + 100);  // only thread 0 writes
    __syncthreads();
    CUDAPHASE_MARK(trace, 1, cudaphase_now());
    if ((int)(blockIdx.y * gridDim.x + blockIdx.x) == skip) return;
    CUDAPHASE_MARK(trace, 2, cudaphase_now());
    CUDAPHASE_MARK(trace, 3, cudaphase_smid());
}`
	const blocks = 6
	out, err := cudahostemu.Run(context.Background(), dir, cudahostemu.Program{
		Source:  src,
		Flags:   []string{"-I", dir},
		Launch:  "eg_emu_launch(dim3(3, 2), dim3(64), 0, k, trace, 4); eg_emu_launch(dim3(3, 2), dim3(64), 0, k, (unsigned long long*)nullptr, 0)",
		Buffers: []cudahostemu.Buffer{{Name: "trace", CType: "unsigned long long", Data: make([]byte, blocks*16*8)}},
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]uint64, blocks*16)
	for i := range buf {
		buf[i] = binary.LittleEndian.Uint64(out["trace"][8*i:])
	}
	s, err := Summarize("k", buf, 16, []Phase{{"sync", 0, 1}, {"rest", 1, 2}}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if s.Blocks != blocks || s.Phases[0].Blocks != blocks || s.Phases[1].Blocks != blocks-1 || s.SMs != 1 {
		t.Errorf("summary %+v: want %d blocks, %d with rest, 1 SM", s, blocks, blocks-1)
	}
	for b := 0; b < blocks; b++ {
		if v := buf[b*16+4]; v != 100 {
			t.Errorf("block %d slot 4 = %d, want 100 (thread 0's mark)", b, v)
		}
	}
	if s.SpanUS <= 0 {
		t.Errorf("span %v, want > 0", s.SpanUS)
	}
}

func TestParse(t *testing.T) {
	log := "noise\n  CUDAPHASE {\"name\":\"a\",\"blocks\":2,\"span_us\":1.5,\"quantum_us\":0.256,\"phases\":[]}\nTRACE other\nCUDAPHASE {\"name\":\"b\",\"blocks\":1,\"span_us\":0,\"quantum_us\":0,\"phases\":[],\"extra\":{\"regs\":96}}\n"
	got, err := Parse(strings.NewReader(log))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Name != "a" || got[0].SpanUS != 1.5 || got[1].Extra["regs"] != 96 {
		t.Errorf("Parse = %+v", got)
	}
	if _, err := Parse(strings.NewReader("x\nCUDAPHASE {bad\n")); err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Errorf("bad line: error = %v, want one naming line 2", err)
	}
}
