package qualify

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

const hopperAttnGroup = 16

type hopperTyped struct {
	dtype hopper.DType
	vals  []float64
}

func (b hopperTyped) bytes() []byte {
	switch b.dtype {
	case hopper.BF16:
		v := make([]uint16, len(b.vals))
		for i, f := range b.vals {
			v[i] = hopper.BF16Bits(float32(f))
		}
		return u16Bytes(v)
	case hopper.E4M3:
		v := make([]byte, len(b.vals))
		for i, f := range b.vals {
			v[i] = hopper.E4M3Bits(float32(f))
		}
		return v
	case hopper.U8:
		v := make([]byte, len(b.vals))
		for i, f := range b.vals {
			v[i] = byte(f)
		}
		return v
	case hopper.I64:
		v := make([]int64, len(b.vals))
		for i, f := range b.vals {
			v[i] = int64(f)
		}
		return i64Bytes(v)
	case hopper.I32:
		v := make([]int32, len(b.vals))
		for i, f := range b.vals {
			v[i] = int32(f)
		}
		return i32Bytes(v)
	}
	v := make([]float32, len(b.vals))
	for i, f := range b.vals {
		v[i] = float32(f)
	}
	return f32Bytes(v)
}

func hopperDecode(d hopper.DType, raw []byte) []float64 {
	var out []float64
	switch d {
	case hopper.BF16:
		for _, b := range bytesU16(raw) {
			out = append(out, float64(hopper.FromBF16Bits(b)))
		}
	case hopper.E4M3:
		for _, b := range raw {
			out = append(out, float64(hopper.FromE4M3Bits(b)))
		}
	default:
		for _, f := range bytesF32(raw) {
			out = append(out, float64(f))
		}
	}
	return out
}

func f64s(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, f := range v {
		out[i] = float64(f)
	}
	return out
}

func hopperSize(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func randVals(r *rand.Rand, n int, scale float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = r.NormFloat64() * scale
	}
	return v
}

func agree(t *testing.T, name string, d hopper.DType, got, want []float64) {
	t.Helper()
	rel := map[hopper.DType]float64{hopper.F32: 1e-4, hopper.BF16: 1.0 / 64, hopper.E4M3: 1.0 / 4}[d]
	bad := 0
	for i := range want {
		if diff := math.Abs(got[i] - want[i]); diff > rel*math.Abs(want[i])+1e-6 || math.IsNaN(got[i]) {
			if bad < 3 {
				t.Errorf("%s[%d] = %g, want %g", name, i, got[i], want[i])
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d of %d values differ", name, bad, len(want))
	}
}

type wsCase struct {
	g      hopper.TileGEMM
	m      int
	bufs   []cudahostemu.Buffer
	launch string
	want   []float32
}

func newWSCase(t *testing.T, g hopper.TileGEMM, m int) wsCase {
	t.Helper()
	r := rand.New(rand.NewPCG(7, uint64(g.K)))
	x := randBF16(r, m*g.K, 1)
	var w hopper.GEMMWeights
	bufs := []cudahostemu.Buffer{{Name: "x", CType: "uint16_t", Data: u16Bytes(x)}}
	args := []string{fmt.Sprint(m), "x"}
	switch g.Weight {
	case hopper.WeightNVFP4:
		w = randNVFP4(r, g.N, g.K)
		w.GlobalScal = 1.0 / 64
		q, sf, err := hopper.RepackNVFP4WGMMA(w.Packed, w.Scales, g.N, g.K, g.GLU)
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
	want, err := g.Reference(m, x, nil, nil, []hopper.GEMMWeights{w}, nil)
	if err != nil {
		t.Fatal(err)
	}
	grid, block := g.Grid(m, 1)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %s)",
		grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, strings.Join(args, ", "))
	return wsCase{g, m, bufs, launch, want}
}

func (c wsCase) run(t *testing.T, src string, flags ...string) (bool, error) {
	t.Helper()
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source: hopper.Prelude + src, Launch: c.launch, Buffers: c.bufs, Flags: flags,
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

type tileInputs struct {
	m      int
	x      []uint16
	rows   []int32
	w      hopper.GEMMWeights
	extras [][]float32
}

func newTileInputs(r *rand.Rand, g hopper.TileGEMM, m, tokens int, exact bool) tileInputs {
	in := tileInputs{m: m}
	xrows := m
	if g.Gather {
		xrows = tokens
		for range m {
			in.rows = append(in.rows, int32(r.IntN(tokens)))
		}
	}
	small := func() float32 { return float32(r.IntN(9) - 4) }
	in.x = make([]uint16, xrows*g.K)
	for i := range in.x {
		switch {
		case g.Weight == hopper.WeightE4M3 && exact:
			in.x[i] = uint16(hopper.E4M3Bits(small()))
		case g.Weight == hopper.WeightE4M3:
			in.x[i] = uint16(hopper.E4M3Bits(float32(r.NormFloat64() * 0.5)))
		case exact:
			in.x[i] = hopper.BF16Bits(small())
		default:
			in.x[i] = hopper.BF16Bits(float32(r.NormFloat64()))
		}
	}
	switch g.Weight {
	case hopper.WeightNVFP4:
		in.w = randNVFP4(r, g.N, g.K)
		in.w.GlobalScal = 1.0 / 64
	case hopper.WeightE4M3:
		in.w.E4M3 = make([]uint8, g.N*g.K)
		for i := range in.w.E4M3 {
			if exact {
				in.w.E4M3[i] = hopper.E4M3Bits(small())
			} else {
				in.w.E4M3[i] = hopper.E4M3Bits(float32(r.NormFloat64() * 0.125))
			}
		}
	default:
		in.w.BF16 = make([]uint16, g.N*g.K)
		for i := range in.w.BF16 {
			if exact {
				in.w.BF16[i] = hopper.BF16Bits(small())
			} else {
				in.w.BF16[i] = hopper.BF16Bits(float32(r.NormFloat64() * 0.05))
			}
		}
	}
	nout := g.N
	if g.GLU {
		nout /= 2
	}
	for _, e := range g.Extras {
		v := make([]float32, map[hopper.Broadcast]int{hopper.PerCol: nout, hopper.PerRow: m, hopper.Full: m * nout}[e.Broadcast])
		for i := range v {
			v[i] = e.DType.Round(float32(r.NormFloat64()))
		}
		in.extras = append(in.extras, v)
	}
	return in
}

func emuTileGEMM(t *testing.T, g hopper.TileGEMM, src string, in tileInputs, launches int, reverse bool) map[string][]byte {
	t.Helper()
	var bufs []hopperEmuBuf
	if g.Weight == hopper.WeightE4M3 {
		xb := make([]byte, len(in.x))
		for i, v := range in.x {
			xb[i] = uint8(v)
		}
		bufs = append(bufs, hopperEmuBuf{"x", "uint8_t", xb})
	} else {
		bufs = append(bufs, hopperEmuBuf{"x", "uint16_t", u16Bytes(in.x)})
	}
	args := []string{fmt.Sprint(in.m), "x"}
	if g.Gather {
		bufs = append(bufs, hopperEmuBuf{"rows", "int32_t", i32Bytes(in.rows)})
		args = append(args, "rows")
	}
	switch {
	case g.Weight == hopper.WeightNVFP4 && g.Repacked:
		repack := hopper.RepackNVFP4WGMMA
		if g.DequantWords == 2 {
			repack = hopper.RepackNVFP4Dequant2Word
		}
		q, sf, err := repack(in.w.Packed, in.w.Scales, g.N, g.K, g.GLU)
		if err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, hopperEmuBuf{"w", "uint32_t", u32Bytes(q)}, hopperEmuBuf{"w_sf", "uint8_t", sf},
			hopperEmuBuf{"w_gs", "float", f32Bytes([]float32{in.w.GlobalScal})})
		args = append(args, "w", "w_sf", "w_gs")
	case g.Weight == hopper.WeightNVFP4:
		bufs = append(bufs, hopperEmuBuf{"w", "uint8_t", in.w.Packed}, hopperEmuBuf{"w_sf", "uint8_t", in.w.Scales},
			hopperEmuBuf{"w_gs", "float", f32Bytes([]float32{in.w.GlobalScal})})
		args = append(args, "w", "w_sf", "w_gs")
	case g.Weight == hopper.WeightE4M3:
		bufs = append(bufs, hopperEmuBuf{"w", "uint8_t", in.w.E4M3})
		args = append(args, "w")
	default:
		bufs = append(bufs, hopperEmuBuf{"w", "uint16_t", u16Bytes(in.w.BF16)})
		args = append(args, "w")
	}
	for i, e := range g.Extras {
		bufs = append(bufs, hopperEmuBuf{e.Name, e.DType.CType(), hopperTyped{e.DType, f64s(in.extras[i])}.bytes()})
		args = append(args, e.Name)
	}
	nout := g.N
	if g.GLU {
		nout /= 2
	}
	bufs = append(bufs, hopperEmuBuf{"out", g.Out.CType(), poison(in.m * nout * g.Out.Bits() / 8)})
	args = append(args, "out")
	if g.SumSq {
		bufs = append(bufs, hopperEmuBuf{"ss", "float", poison(4 * in.m * nout / 32)})
		args = append(args, "ss")
	}
	if g.StreamK > 0 {
		ws := poison(g.WorkspaceBytes())
		clear(ws[len(ws)-4*g.StreamK:])
		bufs = append(bufs, hopperEmuBuf{"ws", "float", ws})
		args = append(args, "ws")
	}
	grid, block := g.Grid(in.m, 1)
	kernel := g.Name
	if reverse {
		kernel = fmt.Sprintf("[](auto... a) { blockIdx.x = gridDim.x - 1 - blockIdx.x; %s(a...); }", g.Name)
	}
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %s)",
		grid[0], grid[1], grid[2], block[0], g.SharedBytes(), kernel, strings.Join(args, ", "))
	return runHopperEmu(t, src, strings.Repeat(launch+"; ", launches), bufs...)
}

func firstDiff(d hopper.DType, a, b []byte) string {
	if len(a) != len(b) {
		return fmt.Sprintf("length %d vs %d", len(a), len(b))
	}
	if d != hopper.F32 && d != hopper.BF16 {
		d = hopper.F32
	}
	va, vb := hopperDecode(d, a), hopperDecode(d, b)
	for i := range va {
		if !bytes.Equal(a[i*d.Bits()/8:(i+1)*d.Bits()/8], b[i*d.Bits()/8:(i+1)*d.Bits()/8]) {
			return fmt.Sprintf("element %d: %g vs %g", i, va[i], vb[i])
		}
	}
	return "no element"
}

func pagedCase(r *rand.Rand, a hopper.PagedAttention, seqlens []int32, kvScale float32) ([]float32, hopper.PagedCache) {
	maxPages := 0
	for _, s := range seqlens {
		maxPages = max(maxPages, (int(s)+a.Page-1)/a.Page)
	}
	total := len(seqlens) * maxPages
	c := hopper.PagedCache{
		PageTable: make([]int32, len(seqlens)*maxPages),
		MaxPages:  maxPages,
		SeqLens:   seqlens,
		KScale:    kvScale,
		VScale:    kvScale,
		K:         make([]uint8, total*a.Page*a.Hkv*hopper.HeadDim),
		V:         make([]uint8, total*a.Page*a.Hkv*hopper.HeadDim),
	}
	perm := r.Perm(total)
	for i := range c.PageTable {
		c.PageTable[i] = int32(perm[i])
	}
	for b, s := range seqlens {
		for p := (int(s) + a.Page - 1) / a.Page; p < maxPages; p++ {
			c.PageTable[b*maxPages+p] = -1
		}
	}
	live := make([]bool, total*a.Page)
	for b, s := range seqlens {
		for t := range int(s) {
			live[int(c.PageTable[b*maxPages+t/a.Page])*a.Page+t%a.Page] = true
		}
	}
	for i := range c.K {
		c.K[i], c.V[i] = 0x7f, 0x7f
		if live[i/(a.Hkv*hopper.HeadDim)] {
			c.K[i] = hopper.E4M3Bits(float32(r.NormFloat64()) / kvScale)
			c.V[i] = hopper.E4M3Bits(float32(r.NormFloat64()) / kvScale)
		}
	}
	q := make([]float32, len(seqlens)*a.QLen*a.Hq*hopper.HeadDim)
	for i := range q {
		q[i] = a.Q.Round(float32(r.NormFloat64()))
	}
	return q, c
}

func attnCounterBytes(a hopper.PagedAttention, batch, splits int) int {
	if a.ClusterCombine {
		splits /= hopper.ClusterSize
	}
	ng := (splits + hopperAttnGroup - 1) / hopperAttnGroup
	return 4 * batch * a.Hkv / max(a.KVHeadsPerBlock, 1) * (1 + ng)
}

func attnWorkspace(a hopper.PagedAttention, batch, maxKeys int) []byte {
	ws := poison(a.WorkspaceBytes(batch, maxKeys))
	clear(ws[:attnCounterBytes(a, batch, a.Splits(maxKeys))])
	return ws
}

func runPaged(t *testing.T, a hopper.PagedAttention, q []float32, c hopper.PagedCache, maxKeys, launches, slots int) ([]float64, []byte, [3]int) {
	t.Helper()
	src, err := a.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	l, err := a.Launcher()
	if err != nil {
		t.Fatal(err)
	}
	src += "\n" + l
	batch := len(c.SeqLens)
	args := "q, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, out, ws"
	bufs := []hopperEmuBuf{
		{"q", a.Q.CType(), hopperTyped{a.Q, f64s(q)}.bytes()},
		{"k_cache", "uint8_t", c.K},
		{"v_cache", "uint8_t", c.V},
		{"page_table", "int32_t", i32Bytes(c.PageTable)},
		{"seqlens", "int32_t", i32Bytes(c.SeqLens)},
		{"k_scale", "float", f32Bytes([]float32{c.KScale})},
		{"v_scale", "float", f32Bytes([]float32{c.VScale})},
	}
	if a.Gate {
		bufs = append(bufs, hopperEmuBuf{"gate", "uint16_t", hopperTyped{hopper.BF16, f64s(c.Gate)}.bytes()})
		args = "q, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, gate, out, ws"
	}
	bufs = append(bufs,
		hopperEmuBuf{"out", a.Out.CType(), poison(len(q) * a.Out.Bits() / 8)},
		hopperEmuBuf{"ws", "unsigned char", attnWorkspace(a, batch, maxKeys)},
		hopperEmuBuf{"grid", "uint32_t", poison(12)})
	launch := fmt.Sprintf("eg_emu_sms = %d;\n    eg_emu_blocks_per_sm = 1", slots)
	one := fmt.Sprintf("%s_launch(%d, %d, "+args+", nullptr)", a.Name, batch, maxKeys, c.MaxPages)
	launch += ";\n    " + one
	for i := 1; i < launches; i++ {
		launch += ";\n    " + one
	}
	launch += ";\n    grid[0] = eg_emu_last_grid.x;\n    grid[1] = eg_emu_last_grid.y;\n    grid[2] = eg_emu_last_grid.z"
	res := runHopperEmu(t, src, launch, bufs...)
	g := i32s(res["grid"])
	return hopperDecode(a.Out, res["out"]), res["ws"], [3]int{int(g[0]), int(g[1]), int(g[2])}
}

func runMegakernelEpochs(t *testing.T, m hopper.Megakernel, envs []map[string]hopper.Value, flags ...string) ([][]float64, []uint64, error) {
	t.Helper()
	src, err := m.Source()
	if err != nil {
		t.Fatal(err)
	}
	var cbufs []cudahostemu.Buffer
	var launches []string

	for _, tmp := range m.Temps {
		raw := poison(hopperSize(tmp.Shape) * tmp.DType.Bits() / 8)
		cbufs = append(cbufs, cudahostemu.Buffer{Name: tmp.Name, CType: tmp.DType.CType(), Data: raw})
	}
	cbufs = append(cbufs, cudahostemu.Buffer{
		Name:  "ready",
		CType: "unsigned long long",
		Data:  make([]byte, m.ReadyBytes()),
	})

	for epIdx, env := range envs {
		epoch := epIdx + 1
		var args []string
		for _, in := range m.Inputs {
			bname := fmt.Sprintf("%s_ep%d", in.Name, epoch)
			cbufs = append(cbufs, cudahostemu.Buffer{
				Name:  bname,
				CType: in.DType.CType(),
				Data:  hopperTyped{in.DType, env[in.Name].Data}.bytes(),
			})
			args = append(args, bname)
		}
		for _, tmp := range m.Temps {
			args = append(args, tmp.Name)
		}
		outName := fmt.Sprintf("%s_ep%d", m.Output.Name, epoch)
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  outName,
			CType: m.Output.DType.CType(),
			Data:  poison(hopperSize(m.Output.Shape) * m.Output.DType.Bits() / 8),
		})
		args = append(args, outName, "ready", fmt.Sprintf("%du", epoch), "nullptr")
		launches = append(launches, fmt.Sprintf("%s_run(%s)", m.Name, strings.Join(args, ", ")))
	}

	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  strings.Join(launches, ";\n    "),
		Buffers: cbufs,
		Flags:   flags,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		return nil, nil, err
	}
	outs := make([][]float64, len(envs))
	for epIdx := range envs {
		outName := fmt.Sprintf("%s_ep%d", m.Output.Name, epIdx+1)
		outs[epIdx] = hopperDecode(m.Output.DType, res[outName])
	}
	return outs, bytesU64(res["ready"]), nil
}

func runPlan(t *testing.T, p *hopper.Plan, env map[string]hopper.Value) []float64 {
	t.Helper()
	var src strings.Builder
	var launches []string
	wsZero := make(map[string]int)
	for _, k := range p.Kernels {
		if k.GEMM != nil {
			s, err := k.GEMM.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			grid, block, _ := k.GEMM.Grid()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)",
				grid[0], grid[1], block[0], k.GEMM.Name, k.M, strings.Join(k.Args, ", ")))
			continue
		}
		if k.Tile != nil {
			s, err := k.Tile.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			if k.Tile.Grouped {
				grid, block := k.Tile.Grid(k.MaxRows, k.Experts)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], grid[2], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			} else {
				grid, block := k.Tile.Grid(k.M, 1)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			}
			continue
		}
		if k.UMMA != nil || k.Quant != nil {
			var s, l, name string
			var err error
			if k.UMMA != nil {
				name = k.UMMA.Name
				if s, err = k.UMMA.CUDA(); err == nil {
					l, err = k.UMMA.Launcher()
				}
			} else {
				name = k.Quant.Name
				if s, err = k.Quant.CUDA(); err == nil {
					l, err = k.Quant.Launcher()
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s + "\n" + l)
			launches = append(launches, fmt.Sprintf("%s_launch(%d, %s, nullptr)", name, k.M, strings.Join(k.Args, ", ")))
			continue
		}
		if k.Attention != nil {
			s, err := k.Attention.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			l, err := k.Attention.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s + "\n" + l)
			args := append([]string(nil), k.Args[:4]...)
			args = append(args, fmt.Sprint(k.MaxPages))
			args = append(args, k.Args[4:]...)
			launches = append(launches, fmt.Sprintf("eg_emu_sms = 16; eg_emu_blocks_per_sm = 1; %s_launch(%d, %d, %s, nullptr)",
				k.Attention.Name, k.Batch, k.MaxKeys, strings.Join(args, ", ")))
			wsName := k.Args[len(k.Args)-1]
			wsZero[wsName] = attnCounterBytes(*k.Attention, k.Batch, k.Attention.Splits(k.MaxKeys))
			continue
		}
		if k.TopK != nil {
			s, err := k.TopK.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			grid, block := k.TopK.Grid(k.Rows)
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
				grid[0], block[0], k.TopK.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		if k.Align != nil {
			s, err := k.Align.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			al, _ := k.Align.Norm()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(1), dim3(%d), 0, %s, %d, %s)",
				32*al.Warps, k.Align.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		if k.ExpertSum != nil {
			s, err := k.ExpertSum.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			grid, block := k.ExpertSum.Grid(k.Rows)
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
				grid[0], block[0], k.ExpertSum.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		s, err := k.Row.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		src.WriteString(s)
		if k.Row.Cluster {
			l, err := k.Row.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString("\n" + l)
			launches = append(launches, fmt.Sprintf("%s_launch(%d, %s, nullptr)",
				k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		grid, block := k.Row.Grid(k.Rows)
		launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
			grid[0], block[0], k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
	}
	var bufs []hopperEmuBuf
	for _, in := range p.Inputs {
		bufs = append(bufs, hopperEmuBuf{in.Name, in.DType.CType(), hopperTyped{in.DType, env[in.Name].Data}.bytes()})
	}
	for _, tmp := range append(p.Temps, p.Output) {
		raw := poison(hopperSize(tmp.Shape) * tmp.DType.Bits() / 8)
		if z, ok := wsZero[tmp.Name]; ok && z <= len(raw) {
			clear(raw[:z])
		}
		if tmp.DType == hopper.I32 && strings.HasSuffix(tmp.Name, "_count") {
			clear(raw)
		}
		bufs = append(bufs, hopperEmuBuf{tmp.Name, tmp.DType.CType(), raw})
	}
	res := runHopperEmu(t, src.String(), strings.Join(launches, ";\n    "), bufs...)
	return hopperDecode(p.Output.DType, res[p.Output.Name])
}

func randEnv(r *rand.Rand, ts ...hopper.Tensor) map[string]hopper.Value {
	env := make(map[string]hopper.Value)
	for _, t := range ts {
		v := randVals(r, hopperSize(t.Shape), 1)
		for i := range v {
			v[i] = float64(t.DType.Round(float32(v[i])))
		}
		env[t.Name] = hopper.Value{Shape: t.Shape, Data: v}
	}
	return env
}

func TestPGORefineClassifyBottlenecks(t *testing.T) {
	tests := []struct {
		name string
		obs  KernelObservation
		want []BottleneckKind
	}{
		{
			name: "healthy",
			obs: KernelObservation{
				Name:               "healthy_k",
				DurationNs:         20000,
				LaunchGapNs:        1000,
				SharedBankConflict: 0.02,
				Registers:          96,
				AchievedOccupancy:  0.50,
				GridBlocks:         264,
				DeviceSMs:          132,
			},
			want: nil,
		},
		{
			name: "bank_conflict",
			obs:  KernelObservation{SharedBankConflict: 0.12},
			want: []BottleneckKind{BottleneckBankConflict},
		},
		{
			name: "wgmma_serialized",
			obs:  KernelObservation{WGMMASerialized: true},
			want: []BottleneckKind{BottleneckWGMMASerial},
		},
		{
			name: "register_spill_store",
			obs:  KernelObservation{SpillStoreBytes: 16},
			want: []BottleneckKind{BottleneckRegisterSpill},
		},
		{
			name: "register_spill_load",
			obs:  KernelObservation{SpillLoadBytes: 32},
			want: []BottleneckKind{BottleneckRegisterSpill},
		},
		{
			name: "low_occupancy_high_regs",
			obs:  KernelObservation{AchievedOccupancy: 0.18, Registers: 168},
			want: []BottleneckKind{BottleneckLowOccupancy},
		},
		{
			name: "memory_bound_kv_explicit",
			obs:  KernelObservation{MemoryBoundKV: true},
			want: []BottleneckKind{BottleneckMemoryBoundKV},
		},
		{
			name: "memory_bound_kv_low_occ_low_regs",
			obs:  KernelObservation{AchievedOccupancy: 0.15, Registers: 64},
			want: []BottleneckKind{BottleneckMemoryBoundKV},
		},
		{
			name: "wave_tail_quantization",
			obs:  KernelObservation{GridBlocks: 160, DeviceSMs: 132}, // 1.212 waves
			want: []BottleneckKind{BottleneckWaveTail},
		},
		{
			name: "wave_tail_boundary_035",
			obs:  KernelObservation{GridBlocks: 27, DeviceSMs: 20}, // 1.35 waves
			want: []BottleneckKind{BottleneckWaveTail},
		},
		{
			name: "wave_full_no_tail",
			obs:  KernelObservation{GridBlocks: 264, DeviceSMs: 132}, // 2.0 waves
			want: nil,
		},
		{
			name: "launch_overhead",
			obs:  KernelObservation{DurationNs: 10000, LaunchGapNs: 2000}, // 0.20 >= 0.15
			want: []BottleneckKind{BottleneckLaunchOverhead},
		},
		{
			name: "multiple_combined",
			obs: KernelObservation{
				SharedBankConflict: 0.25,
				WGMMASerialized:    true,
				SpillStoreBytes:    8,
				AchievedOccupancy:  0.12,
				Registers:          192,
				GridBlocks:         150,
				DeviceSMs:          132,
				DurationNs:         5000,
				LaunchGapNs:        1500,
			},
			want: []BottleneckKind{
				BottleneckBankConflict,
				BottleneckWGMMASerial,
				BottleneckRegisterSpill,
				BottleneckLowOccupancy,
				BottleneckWaveTail,
				BottleneckLaunchOverhead,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyBottlenecks(tt.obs)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("ClassifyBottlenecks(%+v) = %v, want %v", tt.obs, got, tt.want)
			}
		})
	}
}

func TestPGORefineObservationFromPTXAS(t *testing.T) {
	fn := ptxasinfo.Function{
		Name:        "_Z12tile_gemm_k0iPKtS0_Pf",
		Arch:        "sm_90a",
		Entry:       true,
		SpillStores: 24,
		SpillLoads:  32,
		Registers:   184,
		Diagnostics: []ptxasinfo.Diagnostic{
			{
				Code: "C7519",
				Text: "(C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function '_Z12tile_gemm_k0iPKtS0_Pf'",
			},
		},
	}
	obs := ObservationFromPTXAS(fn, 160, 132)
	if obs.Name != fn.Name {
		t.Fatalf("Name = %q, want %q", obs.Name, fn.Name)
	}
	if !obs.WGMMASerialized {
		t.Fatal("expected WGMMASerialized = true from C7519 diagnostic")
	}
	if obs.SpillStoreBytes != 24 || obs.SpillLoadBytes != 32 || obs.Registers != 184 {
		t.Fatalf("unexpected resource counts: %+v", obs)
	}
	bns := ClassifyBottlenecks(obs)
	for _, want := range []BottleneckKind{BottleneckWGMMASerial, BottleneckRegisterSpill, BottleneckWaveTail} {
		if !slices.Contains(bns, want) {
			t.Fatalf("ClassifyBottlenecks(%+v) = %v, missing %v", obs, bns, want)
		}
	}
}

func TestPGORefineTileGEMM(t *testing.T) {
	t.Run("bank_conflict_and_wgmma_serial_to_warpspec", func(t *testing.T) {
		base := hopper.TileGEMM{
			Name:   "pgo_tile_ws",
			N:      128,
			K:      128,
			Weight: hopper.WeightBF16,
			Out:    hopper.F32,
			BM:     128,
			BN:     64,
			Stages: 2,
			WGMMA:  true,
		}
		step := Step{Tile: &base, M: 130}
		obs := KernelObservation{
			Name:               base.Name,
			SharedBankConflict: 0.22,
			WGMMASerialized:    true,
		}
		refined, reasons := RefineStep(step, obs)
		if refined.Tile == nil {
			t.Fatal("refined.Tile is nil")
		}
		if !refined.Tile.Swizzle128B || !refined.Tile.WarpSpecialized {
			t.Fatalf("refined TileGEMM = %+v, want Swizzle128B=true and WarpSpecialized=true (reasons: %v)", *refined.Tile, reasons)
		}
		if len(reasons) < 2 {
			t.Fatalf("expected >= 2 rewrite rationales, got %v", reasons)
		}

		src, err := refined.Tile.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		wc := newWSCase(t, *refined.Tile, step.M)
		if ok, err := wc.run(t, src); !ok || err != nil {
			t.Fatalf("emulated refined WarpSpecialized TileGEMM failed: ok=%v err=%v", ok, err)
		}
	})

	t.Run("register_spill_stepdowns", func(t *testing.T) {
		g2w := hopper.TileGEMM{
			Name:         "pgo_tile_2w",
			N:            128,
			K:            128,
			Weight:       hopper.WeightNVFP4,
			Out:          hopper.F32,
			BM:           128,
			BN:           128,
			Stages:       2,
			WGMMA:        true,
			Repacked:     true,
			Swizzle128B:  true,
			DequantWords: 2,
		}
		s1, r1 := RefineStep(Step{Tile: &g2w, M: 64}, KernelObservation{SpillStoreBytes: 16})
		if s1.Tile.DequantWords != 1 || len(r1) == 0 {
			t.Fatalf("expected DequantWords=1, got %d (reasons: %v)", s1.Tile.DequantWords, r1)
		}

		gBN := hopper.TileGEMM{
			Name:        "pgo_tile_bn",
			N:           128,
			K:           128,
			Weight:      hopper.WeightBF16,
			Out:         hopper.F32,
			BM:          128,
			BN:          128,
			Stages:      3,
			WGMMA:       true,
			Swizzle128B: true,
		}
		s2, r2 := RefineStep(Step{Tile: &gBN, M: 64}, KernelObservation{AchievedOccupancy: 0.15, Registers: 196})
		if s2.Tile.BN != 64 || len(r2) == 0 {
			t.Fatalf("expected BN=64, got %d (reasons: %v)", s2.Tile.BN, r2)
		}

		s3, r3 := RefineStep(s2, KernelObservation{SpillLoadBytes: 8})
		if s3.Tile.Stages != 2 || len(r3) == 0 {
			t.Fatalf("expected Stages=2, got %d (reasons: %v)", s3.Tile.Stages, r3)
		}

		r := rand.New(rand.NewPCG(701, 702))
		exact := newTileInputs(r, *s3.Tile, 64, 0, true)
		baseSrc, err := gBN.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		refSrc, err := s3.Tile.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		wantOut := emuTileGEMM(t, gBN, baseSrc, exact, 1, false)["out"]
		gotOut := emuTileGEMM(t, *s3.Tile, refSrc, exact, 1, false)["out"]
		if !bytes.Equal(gotOut, wantOut) {
			t.Fatalf("refined TileGEMM output differs from baseline at %s", firstDiff(hopper.F32, gotOut, wantOut))
		}
	})

	t.Run("wave_tail_to_streamk", func(t *testing.T) {
		base := hopper.TileGEMM{
			Name:        "pgo_tile_sk",
			N:           192,
			K:           192,
			Weight:      hopper.WeightBF16,
			Out:         hopper.F32,
			BM:          128,
			BN:          64,
			Stages:      2,
			WGMMA:       true,
			Swizzle128B: true,
		}
		obs := KernelObservation{
			Name:       base.Name,
			GridBlocks: 6,
			DeviceSMs:  5,
		}
		refined, reasons := RefineStep(Step{Tile: &base, M: 150}, obs)
		if refined.Tile.StreamK != 5 || len(reasons) == 0 {
			t.Fatalf("expected StreamK=5, got %d (reasons: %v)", refined.Tile.StreamK, reasons)
		}

		r := rand.New(rand.NewPCG(703, 704))
		exact := newTileInputs(r, *refined.Tile, 150, 0, true)
		baseSrc, err := base.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		skSrc, err := refined.Tile.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		wantOut := emuTileGEMM(t, base, baseSrc, exact, 1, false)["out"]
		gotOut := emuTileGEMM(t, *refined.Tile, skSrc, exact, 2, false)["out"]
		if !bytes.Equal(gotOut, wantOut) {
			t.Fatalf("StreamK refined TileGEMM differs from baseline at %s", firstDiff(hopper.F32, gotOut, wantOut))
		}
	})
}

func TestPGORefineSkinnyGEMM(t *testing.T) {
	base := hopper.SkinnyGEMM{
		Name:         "pgo_skinny_nvfp4",
		N:            4,
		K:            2048,
		MaxM:         2,
		Weight:       hopper.WeightNVFP4,
		Out:          hopper.F32,
		RowsPerBlock: 2,
		Warps:        1,
	}
	step := Step{GEMM: &base, M: 2}
	obs := KernelObservation{
		Name:              base.Name,
		AchievedOccupancy: 0.14,
		Registers:         64,
		MemoryBoundKV:     true,
	}
	refined, reasons := RefineStep(step, obs)
	if refined.GEMM.Prefetch <= 0 || !refined.GEMM.BlackwellFP4 {
		t.Fatalf("expected Prefetch > 0 and BlackwellFP4=true, got %+v (reasons: %v)", *refined.GEMM, reasons)
	}

	spilled, spillReasons := RefineStep(refined, KernelObservation{SpillStoreBytes: 16, SpillLoadBytes: 16})
	if spilled.GEMM.Prefetch != 0 || spilled.GEMM.RowsPerBlock != 1 || len(spillReasons) == 0 {
		t.Fatalf("expected Prefetch=0 and RowsPerBlock=1 after spill refinement, got %+v (reasons: %v)", *spilled.GEMM, spillReasons)
	}

	r := rand.New(rand.NewPCG(801, 802))
	m := 2
	x := randBF16(r, m*base.K, 1)
	w := randNVFP4(r, base.N, base.K)
	want, err := base.Reference(m, x, w, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		g    hopper.SkinnyGEMM
	}{
		{"base", base},
		{"refined_prefetch_blackwell", *refined.GEMM},
		{"refined_spill_stepdown", *spilled.GEMM},
	} {
		src, err := tc.g.CUDA()
		if err != nil {
			t.Fatalf("%s CUDA: %v", tc.name, err)
		}
		grid, block, err := tc.g.Grid()
		if err != nil {
			t.Fatalf("%s Grid: %v", tc.name, err)
		}
		launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, x, w, w_sf, w_gs, out)",
			grid[0], grid[1], block[0], tc.g.Name, m)
		res := runHopperEmu(t, src, launch,
			hopperEmuBuf{"x", "uint16_t", u16Bytes(x)},
			hopperEmuBuf{"w", "uint8_t", w.Packed},
			hopperEmuBuf{"w_sf", "uint8_t", w.Scales},
			hopperEmuBuf{"w_gs", "float", f32Bytes([]float32{w.GlobalScal})},
			hopperEmuBuf{"out", "float", poison(4 * m * base.N)},
		)
		got := bytesF32(res["out"])
		if idx, ok := closeTo(got, want, 1e-5, 1e-6); !ok {
			t.Fatalf("%s out[%d] = %g, want %g", tc.name, idx, got[idx], want[idx])
		}
	}
}

func TestPGORefinePagedAttention(t *testing.T) {
	base := hopper.PagedAttention{
		Name:       "pgo_attn",
		Hq:         16,
		Hkv:        2,
		QLen:       1,
		Page:       16,
		Q:          hopper.E4M3,
		Out:        hopper.F32,
		Chunk:      256,
		Stages:     3,
		TableAhead: 1,
	}
	step := Step{
		Attention: &base,
		Batch:     2,
		MaxKeys:   512,
	}
	obs := KernelObservation{
		Name:               base.Name,
		SharedBankConflict: 0.18,
		MemoryBoundKV:      true,
		GridBlocks:         160,
		DeviceSMs:          132,
	}
	refined, reasons := RefineStep(step, obs)
	got := *refined.Attention
	if !got.Swizzle {
		t.Fatalf("expected Swizzle=true, got %+v (reasons: %v)", got, reasons)
	}
	if got.TableAhead < 2 {
		t.Fatalf("expected TableAhead >= 2, got %d (reasons: %v)", got.TableAhead, reasons)
	}
	if !got.WGMMA || !got.QKFP8 {
		t.Fatalf("expected WGMMA=true and QKFP8=true for G=8 E4M3, got %+v", got)
	}
	if got.Chunk != 128 && got.Chunk != 64 {
		t.Fatalf("expected Chunk stepped down to 128 or 64, got %d", got.Chunk)
	}

	r := rand.New(rand.NewPCG(901, 902))
	seqlens := []int32{300, 512}
	q, cache := pagedCase(r, got, seqlens, 0.25)
	want, err := got.Reference(q, cache)
	if err != nil {
		t.Fatal(err)
	}
	baseOut, _, _ := runPaged(t, base, q, cache, step.MaxKeys, 1, 16)
	refOut, _, _ := runPaged(t, got, q, cache, step.MaxKeys, 1, 16)
	for i := range want {
		if diff := absDiff(refOut[i], float64(want[i])); diff > 5e-4 {
			t.Fatalf("refined PagedAttention out[%d] = %g, want %g (diff %g)", i, refOut[i], want[i], diff)
		}
		if diff := absDiff(refOut[i], baseOut[i]); diff > 5e-4 {
			t.Fatalf("refined vs base PagedAttention out[%d] = %g vs %g (diff %g)", i, refOut[i], baseOut[i], diff)
		}
	}
}

func TestPGORefinePlanMegakernel(t *testing.T) {
	const (
		tokens = 2
		kDim   = 256
		nOut   = 256
	)
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.BF16, tokens, kDim)
	w := b.Input("w", hopper.BF16, nOut, kDim)
	qkv := b.MatMul(x, w)
	q := b.Reshape(b.Slice(qkv, 0, 128), tokens, 2, 64)
	ss := b.ReduceSum(b.Map(hopper.MustLambda("(mul $0 $0)"), q))
	norm := b.Map(hopper.MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), q, ss)
	gate := b.Reshape(b.Slice(qkv, 128, 128), tokens, 2, 64)
	root := b.Map(hopper.MustLambda("(mul $0 (sigmoid $1))"), norm, gate)

	expr, err := hopper.Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := hopper.Compile(expr, "pgo_qkv_norm", hopper.BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 2 || plan.Kernels[0].GEMM == nil || plan.Kernels[1].Row == nil {
		t.Fatalf("unexpected plan kernels: %+v", plan.Kernels)
	}

	receipt := ProfileReceipt{
		Device: "NVIDIA H200",
		Observations: []KernelObservation{
			{
				Name:              plan.Kernels[0].GEMM.Name,
				DurationNs:        12000,
				LaunchGapNs:       500,
				AchievedOccupancy: 0.16,
				Registers:         64,
				MemoryBoundKV:     true,
			},
			{
				Name:        plan.Kernels[1].Row.Name,
				DurationNs:  6000,
				LaunchGapNs: 2100,
			},
		},
	}

	refinedPlan, mega, actions, err := RefineFromProfile(plan, receipt)
	if err != nil {
		t.Fatal(err)
	}
	if mega == nil {
		t.Fatalf("expected non-nil MegakernelSource when consumer has BottleneckLaunchOverhead (actions: %v)", actions)
	}
	if mega.Name != "pgo_qkv_norm_mega" {
		t.Fatalf("mega.Name = %q, want %q", mega.Name, "pgo_qkv_norm_mega")
	}
	if refinedPlan.Kernels[0].GEMM.Prefetch <= 0 {
		t.Fatalf("expected refined step 0 SkinnyGEMM Prefetch > 0, got %d", refinedPlan.Kernels[0].GEMM.Prefetch)
	}
	if plan.Kernels[0].GEMM.Prefetch != 0 {
		t.Fatal("original plan was mutated in place")
	}
	if len(actions) < 2 {
		t.Fatalf("expected >= 2 action log entries, got %v", actions)
	}

	r := rand.New(rand.NewPCG(1001, 1002))
	envs := []map[string]hopper.Value{
		randEnv(r, hopper.Tensor{Name: "x", DType: hopper.BF16, Shape: []int{tokens, kDim}}, hopper.Tensor{Name: "w", DType: hopper.BF16, Shape: []int{nOut, kDim}}),
		randEnv(r, hopper.Tensor{Name: "x", DType: hopper.BF16, Shape: []int{tokens, kDim}}, hopper.Tensor{Name: "w", DType: hopper.BF16, Shape: []int{nOut, kDim}}),
	}
	megaOuts, _, err := runMegakernelEpochs(t, *mega, envs)
	if err != nil {
		t.Fatal(err)
	}
	for ep, env := range envs {
		unrefinedOut := runPlan(t, plan, env)
		refinedPlanOut := runPlan(t, refinedPlan, env)
		agree(t, fmt.Sprintf("epoch_%d_refined_plan", ep+1), hopper.BF16, refinedPlanOut, unrefinedOut)
		agree(t, fmt.Sprintf("epoch_%d_megakernel", ep+1), hopper.BF16, megaOuts[ep], unrefinedOut)
	}
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}

func TestPGORefineProposeStepCandidates(t *testing.T) {
	baseTile := hopper.TileGEMM{
		Name:   "pgo_tile_cand",
		N:      128,
		K:      128,
		Weight: hopper.WeightBF16,
		Out:    hopper.F32,
		BM:     128,
		BN:     128,
		Stages: 3,
		WGMMA:  true,
	}
	step := Step{Tile: &baseTile, M: 128}

	fn := ptxasinfo.Function{
		Name:      baseTile.Name,
		Registers: 128,
	}
	causal := ObservationFromCountersAndPTXAS(fn, CounterMetrics{
		DurationNs:  15000,
		LaunchGapNs: 500,
	}, 264, 132)
	if causal.SharedBankConflict != 0 {
		t.Fatalf("expected unmeasured SharedBankConflict == 0, got %v", causal.SharedBankConflict)
	}
	if bns := ClassifyBottlenecks(causal); len(bns) != 0 {
		t.Fatalf("expected no causal bottlenecks, got %v", bns)
	}

	stalls := StallSample{
		TotalSamples: 100,
		MIOThrottle:  35, // 0.35 -> BottleneckBankConflict
		WaitTMEM:     20, // 0.20 -> BottleneckLowOccupancy
	}
	proposals := ProposeStepCandidates(step, causal, stalls)
	if causal.SharedBankConflict != 0 {
		t.Fatalf("ProposeStepCandidates mutated causal observation: %+v", causal)
	}
	if len(proposals) != 2 {
		t.Fatalf("expected 2 stall-hypothesis proposals, got %d: %+v", len(proposals), proposals)
	}
	if proposals[0].Causal || proposals[0].Hypothesis != BottleneckBankConflict || !proposals[0].Step.Tile.Swizzle128B {
		t.Fatalf("unexpected first proposal: %+v", proposals[0])
	}
	if proposals[1].Causal || proposals[1].Hypothesis != BottleneckLowOccupancy || proposals[1].Step.Tile.BN != 64 {
		t.Fatalf("unexpected second proposal: %+v", proposals[1])
	}

	bank := 0.25
	causalWithBank := ObservationFromCountersAndPTXAS(fn, CounterMetrics{
		SharedBankConflict: &bank,
	}, 264, 132)
	proposalsWithCausal := ProposeStepCandidates(step, causalWithBank, stalls)
	if len(proposalsWithCausal) != 2 {
		t.Fatalf("expected 1 causal + 1 distinct stall proposal, got %d: %+v", len(proposalsWithCausal), proposalsWithCausal)
	}
	if !proposalsWithCausal[0].Causal || !proposalsWithCausal[0].Step.Tile.Swizzle128B {
		t.Fatalf("expected first proposal to be Causal: true Swizzle128B, got %+v", proposalsWithCausal[0])
	}
	if proposalsWithCausal[1].Causal || proposalsWithCausal[1].Hypothesis != BottleneckLowOccupancy {
		t.Fatalf("expected second proposal to be non-causal BottleneckLowOccupancy, got %+v", proposalsWithCausal[1])
	}
}

func ExampleRefinePlan() {
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.BF16, 2, 256)
	w := b.Input("w", hopper.BF16, 64, 256)
	y := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(hopper.MustLambda("(mul $0 $0)"), y))
	root := b.Map(hopper.MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), y, ss)
	expr, _ := hopper.Optimize(b.G, root)
	plan, _ := hopper.Compile(expr, "proj_norm", hopper.BF16)

	receipt := ProfileReceipt{
		Device: "NVIDIA H200",
		Observations: []KernelObservation{
			{Name: plan.Kernels[0].GEMM.Name, MemoryBoundKV: true},
			{Name: plan.Kernels[1].Row.Name, DurationNs: 5000, LaunchGapNs: 1500},
		},
	}
	refined, mega, actions, _ := RefinePlan(plan, receipt)
	fmt.Println(refined.Kernels[0].GEMM.Prefetch > 0, mega != nil, len(actions), strings.HasSuffix(mega.Name, "_mega"))
	// Output:
	// true true 2 true
}
