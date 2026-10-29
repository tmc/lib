package hopper

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

// wsCase is a WarpSpecialized TileGEMM with inputs and its reference output.
type wsCase struct {
	g      TileGEMM
	m      int
	bufs   []cudahostemu.Buffer
	launch string
	want   []float32
}

func newWSCase(t *testing.T, g TileGEMM, m int) wsCase {
	t.Helper()
	r := rand.New(rand.NewPCG(7, uint64(g.K)))
	x := randBF16(r, m*g.K, 1)
	var w GEMMWeights
	bufs := []cudahostemu.Buffer{{Name: "x", CType: "uint16_t", Data: u16Bytes(x)}}
	args := []string{fmt.Sprint(m), "x"}
	switch g.Weight {
	case WeightNVFP4:
		w = randNVFP4(r, g.N, g.K)
		w.GlobalScal = 1.0 / 64
		q, sf, err := RepackNVFP4WGMMA(w.Packed, w.Scales, g.N, g.K, g.GLU)
		if err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs,
			cudahostemu.Buffer{Name: "w", CType: "uint32_t", Data: u32Bytes(q)},
			cudahostemu.Buffer{Name: "w_sf", CType: "uint8_t", Data: sf},
			cudahostemu.Buffer{Name: "w_gs", CType: "float", Data: f32Bytes([]float32{w.GlobalScal})})
		args = append(args, "w", "w_sf", "w_gs")
	default:
		w.BF16 = randBF16(r, g.N*g.K, 0.05)
		bufs = append(bufs, cudahostemu.Buffer{Name: "w", CType: "uint16_t", Data: u16Bytes(w.BF16)})
		args = append(args, "w")
	}
	bufs = append(bufs, cudahostemu.Buffer{Name: "out", CType: "float", Data: poison(4 * m * g.N)})
	args = append(args, "out")
	want, err := g.Reference(m, x, nil, nil, []GEMMWeights{w}, nil)
	if err != nil {
		t.Fatal(err)
	}
	grid, block := g.Grid(m, 1)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %s)",
		grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, strings.Join(args, ", "))
	return wsCase{g, m, bufs, launch, want}
}

// run runs src (g's kernel, possibly mutated) and reports whether its
// output matches the reference, or the run's error.
func (c wsCase) run(t *testing.T, src string, flags ...string) (bool, error) {
	t.Helper()
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source: Prelude + src, Launch: c.launch, Buffers: c.bufs, Flags: flags,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		return false, err
	}
	var ms float64
	for _, v := range c.want {
		ms += float64(v) * float64(v)
	}
	abs := max(1e-5, 1e-6*math.Sqrt(float64(c.g.K)*ms/float64(len(c.want))))
	_, ok := closeTo(bytesF32(res["out"]), c.want, 1e-5, abs)
	return ok, nil
}

// TestTileGEMMWarpSpecializedMutations breaks the producer/consumer protocol
// of the emitted kernels one way at a time. Under the emulator's adversarial
// mbarrier model every mutant must fail, by wrong output or a loud error, on
// every run; under EG_EMU_SYNC_MBAR, which lands copies at issue and returns
// waits at once, the test only logs what each mutant does.
func TestTileGEMMWarpSpecializedMutations(t *testing.T) {
	mutants := []struct {
		name, old, new string
	}{
		{"no full wait", "eg_mbar_wait(&full[s], (kt / STAGES) & 1);", ""},
		{"wrong full parity", "eg_mbar_wait(&full[s], (kt / STAGES) & 1);", "eg_mbar_wait(&full[s], ((kt / STAGES) & 1) ^ 1);"},
		{"no empty arrive", "if (lane == 0) eg_mbar_arrive(&empty[(kt - 1) % STAGES]);", ""},
		{"no empty wait", "if (kt >= STAGES) eg_mbar_wait(&empty[s], (kt / STAGES - 1) & 1);", ""},
		{"expect_tx short", "eg_mbar_arrive_expect_tx(&full[s], TX);", "eg_mbar_arrive_expect_tx(&full[s], TX - 16);"},
		{"expect_tx long", "eg_mbar_arrive_expect_tx(&full[s], TX);", "eg_mbar_arrive_expect_tx(&full[s], TX + 16);"},
	}
	for _, tt := range []struct {
		name string
		g    TileGEMM
		m    int
	}{
		{"bf16", TileGEMM{Name: "ws_mut_bf16", N: 128, K: 384, Weight: WeightBF16, Out: F32, BN: 64, Stages: 2}, 130},
		{"nvfp4", TileGEMM{Name: "ws_mut_nvfp4", N: 128, K: 384, Weight: WeightNVFP4, Out: F32, BM: 64, Repacked: true, Stages: 2}, 70},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.WGMMA, g.Swizzle128B, g.WarpSpecialized = true, true, true
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			c := newWSCase(t, g, tt.m)
			for _, flags := range [][]string{nil, {"-DEG_EMU_SYNC_MBAR"}} {
				if ok, err := c.run(t, src, flags...); !ok || err != nil {
					t.Fatalf("flags %v: unmutated kernel: ok %v, err %v", flags, ok, err)
				}
			}
			for _, mut := range mutants {
				if !strings.Contains(src, mut.old) {
					t.Fatalf("%s: %q not in source", mut.name, mut.old)
				}
				msrc := strings.Replace(src, mut.old, mut.new, 1)
				for range 2 {
					ok, err := c.run(t, msrc)
					if ok {
						t.Errorf("%s: mutant passed", mut.name)
					}
					t.Logf("%s: adversarial: %s", mut.name, outcome(ok, err))
				}
				ok, err := c.run(t, msrc, "-DEG_EMU_SYNC_MBAR")
				t.Logf("%s: EG_EMU_SYNC_MBAR: %s", mut.name, outcome(ok, err))
			}
		})
	}
}

func outcome(ok bool, err error) string {
	switch {
	case err != nil:
		msg := err.Error()
		if i := strings.Index(msg, "cudahostemu: "); i >= 0 {
			msg = msg[i:]
		}
		if i := strings.IndexByte(msg, '\n'); i >= 0 {
			msg = msg[:i]
		}
		return "error: " + msg
	case ok:
		return "correct output"
	}
	return "wrong output"
}
