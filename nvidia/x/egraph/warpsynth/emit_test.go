package warpsynth

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestEmitPTX(t *testing.T) {
	s := schedules(t)["early"]
	ptx, err := s.EmitWarpSpecializedSkeletonPTX("ws_gemm")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		".target sm_100a",
		".visible .entry ws_gemm(.param .u32 num_tiles)",
		".maxnreg 128",
		"shr.u32 %r2, %r1, 5;",
		"setmaxnreg.dec.sync.aligned.u32 40;",
		"setmaxnreg.inc.sync.aligned.u32",
		"mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r6], %r5;",
		fmt.Sprintf("div.u32 %%r5, %%r3, %d;", s.PipelineStagesD),
		"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r6];",
		"mbarrier.arrive.expect_tx.shared::cta.b64",
	} {
		if !strings.Contains(ptx, want) {
			t.Errorf("PTX lacks %q", want)
		}
	}
	if strings.Contains(ptx, "%warpid") && !strings.Contains(ptx, "%warpid is not stable") {
		t.Error("PTX dispatches on %warpid")
	}
	if strings.Count(ptx, "{") != strings.Count(ptx, "}") {
		t.Error("unbalanced braces")
	}

	// The mutant protocols change the emitted text the way the model does.
	bad, _ := s.emitPTX("ws_gemm", protocol{parityByIter: true})
	if strings.Contains(bad, "div.u32 %r5") || !strings.Contains(bad, "and.b32 %r5, %r3, 1;") {
		t.Error("parity mutant not reflected in PTX")
	}
	bad, _ = s.emitPTX("ws_gemm", protocol{skipEmptyWait: true})
	if strings.Contains(bad, "$L_role0_empty") {
		t.Error("empty-wait mutant not reflected in PTX")
	}

	late := schedules(t)["late"]
	ptx, _ = late.EmitWarpSpecializedSkeletonPTX("ws_gemm")
	if strings.Contains(ptx, "tcgen05.commit") {
		t.Error("late release emits tcgen05.commit")
	}
	if _, err := s.EmitWarpSpecializedSkeletonPTX("1bad"); err == nil {
		t.Error("accepted an invalid kernel name")
	}
}

func f32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// runEmu runs the host-emulation skeleton of s for tiles tiles and reports
// whether every output element is right.
func runEmu(t *testing.T, s Schedule, tiles int, p protocol) (bool, error) {
	t.Helper()
	src, err := s.emitEmu("ws", tiles, p)
	if err != nil {
		t.Fatal(err)
	}
	n := tiles * HostEmuElems
	in := make([]byte, 4*n)
	for i := range n {
		binary.LittleEndian.PutUint32(in[4*i:], math.Float32bits(float32(i+1)))
	}
	res, err := cudahostemu.Run(context.Background(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  fmt.Sprintf("eg_emu_launch(dim3(1), dim3(%d), %d, ws, in, out)", 32*s.TotalWarps, s.PipelineStagesD*HostEmuElems*4),
		Buffers: []cudahostemu.Buffer{{Name: "in", CType: "float", Data: in}, {Name: "out", CType: "float", Data: cudahostemu.Poison(4 * n)}},
	})
	if err != nil {
		return false, err
	}
	for i, v := range f32s(res["out"]) {
		want := 2 * float32(i+1)
		if len(s.Roles) > 2 {
			want++
		}
		if v != want {
			return false, nil
		}
	}
	return true, nil
}

// TestHostEmu runs the C++ skeleton of each schedule for 16 tiles under
// cudahostemu's adversarial mbarrier model.
func TestHostEmu(t *testing.T) {
	if _, err := cudahostemu.Compiler(); err != nil {
		t.Skip(err)
	}
	if testing.Short() {
		t.Skip("host emulation compiles C++")
	}
	for name, s := range schedules(t) {
		t.Run(name, func(t *testing.T) {
			ok, err := runEmu(t, s, 16, protocol{})
			if err != nil || !ok {
				t.Fatalf("ok %v, err %v", ok, err)
			}
		})
	}
}

func TestHostEmuMutants(t *testing.T) {
	if _, err := cudahostemu.Compiler(); err != nil {
		t.Skip(err)
	}
	if testing.Short() {
		t.Skip("host emulation compiles C++")
	}
	s := schedules(t)["early"]
	for _, tt := range []struct {
		name string
		p    protocol
	}{
		{"parity k&1", protocol{parityByIter: true}},
		{"no empty wait", protocol{skipEmptyWait: true}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ok, err := runEmu(t, s, 16, tt.p)
			if ok {
				t.Fatal("mutant produced correct output")
			}
			if err != nil && strings.HasPrefix(err.Error(), "compile") {
				t.Fatalf("mutant did not compile: %v", err)
			}
			t.Logf("mutant failed as expected: err %v", err)
		})
	}
}
