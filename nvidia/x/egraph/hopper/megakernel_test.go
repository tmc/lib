package hopper

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func u64Bytes(v []uint64) []byte {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], x)
	}
	return b
}

func bytesU64(b []byte) []uint64 {
	v := make([]uint64, len(b)/8)
	for i := range v {
		v[i] = binary.LittleEndian.Uint64(b[8*i:])
	}
	return v
}

// TestReadyPrimitivesEmulated tests PackReadyWord, UnpackReadyWord, and the
// ReadyPrelude device helpers (eg_ready_signal, eg_ready_wait) across
// consecutive epochs 1, 2, 3 without resetting the ready counter array.
func TestReadyPrimitivesEmulated(t *testing.T) {
	for _, tc := range []struct {
		epoch, count uint32
	}{
		{1, 1},
		{2, 64},
		{3, 255},
		{1000, 4096},
	} {
		w := PackReadyWord(tc.epoch, tc.count)
		gotEpoch, gotCount := UnpackReadyWord(w)
		if gotEpoch != tc.epoch || gotCount != tc.count {
			t.Fatalf("UnpackReadyWord(PackReadyWord(%d, %d)) = (%d, %d)", tc.epoch, tc.count, gotEpoch, gotCount)
		}
	}

	const src = ReadyPrelude + `
__global__ void ready_pipe(const float* in, float* mid, float* out, unsigned long long* ready, unsigned int epoch) {
    const int bid = blockIdx.x, tid = threadIdx.x;
    if (bid < 4) {
        mid[bid * 32 + tid] = in[bid * 32 + tid] * (float)epoch + 1.0f;
        __syncthreads();
        if (tid == 0) {
            EG_READY_STAGE(&ready[0], epoch, 4, mid, 4 * 32 * sizeof(float));
            eg_ready_signal(&ready[0], epoch, 1);
        }
        return;
    }
    if (tid == 0) {
        eg_ready_wait(&ready[0], epoch, 4);
    }
    __syncthreads();
    const int cid = bid - 4;
    out[cid * 64 + tid] = mid[cid * 64 + tid] + 10.0f;
    out[cid * 64 + 32 + tid] = mid[cid * 64 + 32 + tid] + 10.0f;
}
`
	in := make([]float32, 128)
	for i := range in {
		in[i] = float32(i + 1)
	}
	// Run 3 epochs back to back in one host program without zeroing ready between calls.
	launch := strings.Join([]string{
		"eg_emu_launch(dim3(6), dim3(32), 0, ready_pipe, in1, mid, out1, ready, 1u)",
		"eg_emu_launch(dim3(6), dim3(32), 0, ready_pipe, in2, mid, out2, ready, 2u)",
		"eg_emu_launch(dim3(6), dim3(32), 0, ready_pipe, in3, mid, out3, ready, 3u)",
	}, ";\n    ")
	in2 := make([]float32, 128)
	in3 := make([]float32, 128)
	for i := range in {
		in2[i] = float32(2*i + 3)
		in3[i] = float32(3*i - 5)
	}
	res := runEmu(t, src, launch,
		emuBuf{"in1", "float", f32Bytes(in)},
		emuBuf{"in2", "float", f32Bytes(in2)},
		emuBuf{"in3", "float", f32Bytes(in3)},
		emuBuf{"mid", "float", poison(4 * 128)},
		emuBuf{"out1", "float", poison(4 * 128)},
		emuBuf{"out2", "float", poison(4 * 128)},
		emuBuf{"out3", "float", poison(4 * 128)},
		emuBuf{"ready", "unsigned long long", make([]byte, 8)},
	)
	for ep, tc := range []struct {
		name string
		in   []float32
	}{
		{"out1", in},
		{"out2", in2},
		{"out3", in3},
	} {
		epoch := float32(ep + 1)
		got := bytesF32(res[tc.name])
		for i, v := range got {
			want := tc.in[i]*epoch + 1.0 + 10.0
			if v != want {
				t.Fatalf("epoch %d %s[%d] = %g, want %g", ep+1, tc.name, i, v, want)
			}
		}
	}
	gotReady := bytesU64(res["ready"])
	if want := PackReadyWord(3, 4); gotReady[0] != want {
		t.Fatalf("ready[0] after 3 epochs = %#x, want %#x", gotReady[0], want)
	}
}

// runMegakernelEpochs runs m across consecutive epochs (1..len(envs)) in a
// single host program, reusing the same ready counter buffer without zeroing it
// between epochs, and returns each epoch's output and the final ready words.
func runMegakernelEpochs(t *testing.T, m Megakernel, envs []map[string]Value, flags ...string) ([][]float64, []uint64, error) {
	t.Helper()
	src, err := m.Source()
	if err != nil {
		t.Fatal(err)
	}
	return runMegakernelSourceEpochs(t, m, src, envs, flags...)
}

func runMegakernelSourceEpochs(t *testing.T, m Megakernel, src string, envs []map[string]Value, flags ...string) ([][]float64, []uint64, error) {
	t.Helper()
	var cbufs []cudahostemu.Buffer
	var launches []string

	// Shared temps and ready buffer (allocated once, never reset between epochs).
	for _, tmp := range m.Temps {
		raw := poison(size(tmp.Shape) * tmp.DType.Bits() / 8)
		cbufs = append(cbufs, cudahostemu.Buffer{Name: tmp.Name, CType: tmp.DType.cType(), Data: raw})
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
				CType: in.DType.cType(),
				Data:  typed{in.DType, env[in.Name].Data}.bytes(),
			})
			args = append(args, bname)
		}
		for _, tmp := range m.Temps {
			args = append(args, tmp.Name)
		}
		outName := fmt.Sprintf("%s_ep%d", m.Output.Name, epoch)
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  outName,
			CType: m.Output.DType.cType(),
			Data:  poison(size(m.Output.Shape) * m.Output.DType.Bits() / 8),
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
	if err != nil {
		return nil, nil, err
	}
	outs := make([][]float64, len(envs))
	for epIdx := range envs {
		outName := fmt.Sprintf("%s_ep%d", m.Output.Name, epIdx+1)
		outs[epIdx] = decode(m.Output.DType, res[outName])
	}
	return outs, bytesU64(res["ready"]), nil
}

// TestMegakernelSkinnyGEMMThenRow tests a fused persistent megakernel for
// SkinnyGEMM -> RowKernel (QKV projection followed by per-head RMSNorm +
// sigmoid gate) across 3 consecutive epochs reusing the same ready buffer
// without zeroing between calls.
func TestMegakernelSkinnyGEMMThenRow(t *testing.T) {
	const (
		tokens = 2
		kDim   = 256
		nOut   = 256 // 2 heads of 64 q + 2 heads of 64 gate
	)
	b := NewBuilder()
	x := b.Input("x", BF16, tokens, kDim)
	w := b.Input("w", BF16, nOut, kDim)
	qkv := b.MatMul(x, w)
	q := b.Reshape(b.Slice(qkv, 0, 128), tokens, 2, 64)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), q))
	norm := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), q, ss)
	gate := b.Reshape(b.Slice(qkv, 128, 128), tokens, 2, 64)
	root := b.Map(MustLambda("(mul $0 (sigmoid $1))"), norm, gate)

	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "qkv_norm", BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 2 || plan.Kernels[0].GEMM == nil || plan.Kernels[1].Row == nil {
		t.Fatalf("plan has %d kernels, want [SkinnyGEMM, RowKernel]", len(plan.Kernels))
	}

	mega, err := plan.Megakernel("qkv_norm_mega")
	if err != nil {
		t.Fatal(err)
	}
	if len(mega.Stages) != 2 {
		t.Fatalf("megakernel has %d stages, want 2", len(mega.Stages))
	}
	if mega.ReadyWords() != 1 {
		t.Fatalf("ReadyWords = %d, want 1", mega.ReadyWords())
	}

	r := rand.New(rand.NewPCG(21, 22))
	envs := make([]map[string]Value, 3)
	for i := range envs {
		envs[i] = randEnv(r,
			Tensor{"x", BF16, []int{tokens, kDim}},
			Tensor{"w", BF16, []int{nOut, kDim}},
		)
	}

	outs, readyWords, err := runMegakernelEpochs(t, mega, envs)
	if err != nil {
		t.Fatal(err)
	}
	for ep, env := range envs {
		want, err := Eval(expr, env)
		if err != nil {
			t.Fatal(err)
		}
		agree(t, fmt.Sprintf("epoch_%d_eval", ep+1), BF16, outs[ep], roundAll(BF16, want.Data))
		planOut := runPlan(t, plan, env)
		agree(t, fmt.Sprintf("epoch_%d_plan", ep+1), BF16, outs[ep], planOut)
	}
	wantWord := PackReadyWord(3, uint32(mega.Stages[0].Blocks))
	if readyWords[0] != wantWord {
		t.Fatalf("ready[0] = %#x, want %#x", readyWords[0], wantWord)
	}
}

// TestMegakernelRowThenSkinnyGEMM tests a fused persistent megakernel for
// RowKernel (pre-norm with gain) -> SkinnyGEMM across 3 consecutive epochs
// reusing the same ready buffer without zeroing between calls.
func TestMegakernelRowThenSkinnyGEMM(t *testing.T) {
	const (
		tokens = 3
		dModel = 256
		nOut   = 32
	)
	b := NewBuilder()
	x := b.Input("x", F32, tokens, dModel)
	gain := b.Input("gain", BF16, dModel)
	w := b.Input("w", BF16, nOut, dModel)
	bias := b.Input("bias", F32, nOut)

	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), x))
	normed := b.Map(MustLambda("(mul (mul $0 (rsqrt (add (div $1 256) 1e-5))) (add 1 $2))"), x, ss, gain)
	normedBF16 := b.Cast(BF16, normed)
	mm := b.MatMul(normedBF16, w)
	root := b.Map(MustLambda("(silu (add $0 $1))"), mm, bias)

	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "prenorm_proj", F32)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 2 || plan.Kernels[0].Row == nil || plan.Kernels[1].GEMM == nil {
		t.Fatalf("plan has %d kernels, want [RowKernel, SkinnyGEMM]", len(plan.Kernels))
	}

	mega, err := plan.Megakernel("prenorm_proj_mega")
	if err != nil {
		t.Fatal(err)
	}
	if len(mega.Stages) != 2 {
		t.Fatalf("megakernel has %d stages, want 2", len(mega.Stages))
	}

	r := rand.New(rand.NewPCG(31, 32))
	envs := make([]map[string]Value, 3)
	for i := range envs {
		envs[i] = randEnv(r,
			Tensor{"x", F32, []int{tokens, dModel}},
			Tensor{"gain", BF16, []int{dModel}},
			Tensor{"w", BF16, []int{nOut, dModel}},
			Tensor{"bias", F32, []int{nOut}},
		)
	}

	outs, readyWords, err := runMegakernelEpochs(t, mega, envs)
	if err != nil {
		t.Fatal(err)
	}
	for ep, env := range envs {
		want, err := Eval(expr, env)
		if err != nil {
			t.Fatal(err)
		}
		agree(t, fmt.Sprintf("epoch_%d_eval", ep+1), F32, outs[ep], want.Data)
		planOut := runPlan(t, plan, env)
		agree(t, fmt.Sprintf("epoch_%d_plan", ep+1), F32, outs[ep], planOut)
	}
	wantWord := PackReadyWord(3, uint32(mega.Stages[0].Blocks))
	if readyWords[0] != wantWord {
		t.Fatalf("ready[0] = %#x, want %#x", readyWords[0], wantWord)
	}
}

// TestMegakernelTileGEMMSumSqThenExpertSum tests fusing TileGEMM(SumSq) ->
// ExpertSum into a persistent megakernel across 3 consecutive epochs reusing
// the same ready buffer without zeroing between calls.
func TestMegakernelTileGEMMSumSqThenExpertSum(t *testing.T) {
	const (
		tokens, local  = 3, 2
		width, dExpert = 64, 64
		kRoutes        = 2
		routedRows     = tokens * kRoutes
	)
	down := TileGEMM{
		Name:     "moe_down",
		N:        width,
		K:        dExpert,
		Weight:   WeightBF16,
		Out:      BF16,
		BM:       64,
		BN:       64,
		Grouped:  true,
		SumSq:    true,
		Epilogue: MustLambda("(mul $0 (add 1 $1))"),
		Extras:   []Operand{{"post_w", BF16, PerCol}},
	}
	esum := ExpertSum{
		Name:  "moe_esum",
		Width: width,
		K:     kRoutes,
		Y:     BF16,
		Out:   F32,
		Eps:   1e-8,
	}
	plan := &Plan{
		Name: "moe_down_sum",
		Inputs: []Tensor{
			{"inter", BF16, []int{routedRows, dExpert}},
			{"w2", BF16, []int{local, width, dExpert}},
			{"offsets", I32, []int{local + 1}},
			{"post_w", BF16, []int{width}},
			{"slots", I32, []int{routedRows}},
			{"weights", F32, []int{tokens, kRoutes}},
		},
		Temps: []Tensor{
			{"y", BF16, []int{routedRows, width}},
			{"ss", F32, []int{routedRows, width / 32}},
		},
		Output: Tensor{"out", F32, []int{tokens, width}},
		Kernels: []PlanKernel{
			{
				Tile:    &down,
				M:       routedRows,
				MaxRows: routedRows,
				Experts: local,
				Args:    []string{"inter", "w2", "offsets", "post_w", "y", "ss"},
			},
			{
				ExpertSum: &esum,
				Rows:      tokens,
				Args:      []string{"y", "ss", "slots", "weights", "out"},
			},
		},
	}

	mega, err := plan.Megakernel("moe_down_sum_mega")
	if err != nil {
		t.Fatal(err)
	}
	if mega.SharedBytes() != down.SharedBytes() {
		t.Fatalf("SharedBytes = %d, want %d", mega.SharedBytes(), down.SharedBytes())
	}

	r := rand.New(rand.NewPCG(41, 42))
	envs := make([]map[string]Value, 3)
	for i := range envs {
		env := randEnv(r,
			Tensor{"inter", BF16, []int{routedRows, dExpert}},
			Tensor{"w2", BF16, []int{local, width, dExpert}},
			Tensor{"post_w", BF16, []int{width}},
		)
		env["offsets"] = Value{Shape: []int{local + 1}, Data: []float64{0, 3, 6}}
		env["slots"] = Value{Shape: []int{routedRows}, Data: []float64{0, 3, -1, 1, 4, 2}}
		env["weights"] = Value{Shape: []int{tokens, kRoutes}, Data: []float64{0.6, 0.4, 0.0, 1.0, 0.25, 0.75}}
		envs[i] = env
	}

	outs, readyWords, err := runMegakernelEpochs(t, mega, envs)
	if err != nil {
		t.Fatal(err)
	}
	for ep, env := range envs {
		planOut := runPlan(t, plan, env)
		agree(t, fmt.Sprintf("moe_epoch_%d", ep+1), F32, outs[ep], planOut)
	}
	wantWord := PackReadyWord(3, uint32(mega.Stages[0].Blocks))
	if readyWords[0] != wantWord {
		t.Fatalf("ready[0] = %#x, want %#x", readyWords[0], wantWord)
	}
}

// TestMegakernelMutations verifies that skipping eg_ready_wait
// (-DEG_NO_READY_WAIT) or corrupting the expected producer count or epoch in
// eg_ready_wait causes the emulator test to fail (reading NaN-poisoned
// intermediate rows or failing the ready wait check).
func TestMegakernelMutations(t *testing.T) {
	b := NewBuilder()
	x := b.Input("x", BF16, 2, 256)
	w := b.Input("w", BF16, 64, 256)
	y := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), y))
	root := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), y, ss)

	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "mut", F32)
	if err != nil {
		t.Fatal(err)
	}
	mega, err := plan.Megakernel("mut_mega")
	if err != nil {
		t.Fatal(err)
	}
	baseSrc, err := mega.Source()
	if err != nil {
		t.Fatal(err)
	}
	need := mega.Stages[0].Blocks

	r := rand.New(rand.NewPCG(51, 52))
	envs := []map[string]Value{
		randEnv(r, Tensor{"x", BF16, []int{2, 256}}, Tensor{"w", BF16, []int{64, 256}}),
		randEnv(r, Tensor{"x", BF16, []int{2, 256}}, Tensor{"w", BF16, []int{64, 256}}),
	}

	// Control: unmodified megakernel succeeds and produces no NaNs.
	ctrlOuts, _, err := runMegakernelSourceEpochs(t, mega, baseSrc, envs)
	if err != nil {
		t.Fatalf("control megakernel failed: %v", err)
	}
	for ep, out := range ctrlOuts {
		for i, v := range out {
			if math.IsNaN(v) {
				t.Fatalf("control epoch %d out[%d] is NaN", ep+1, i)
			}
		}
	}

	// Mutation 1: -DEG_NO_READY_WAIT skips eg_ready_wait, so consumer reads
	// NaN-poisoned intermediate rows from mut_t1.
	t.Run("EG_NO_READY_WAIT", func(t *testing.T) {
		outs, _, err := runMegakernelSourceEpochs(t, mega, baseSrc, envs, "-DEG_NO_READY_WAIT")
		if err != nil {
			return // loud failure is also acceptable
		}
		nans := 0
		for _, out := range outs {
			for _, v := range out {
				if math.IsNaN(v) {
					nans++
				}
			}
		}
		if nans == 0 {
			t.Fatal("EG_NO_READY_WAIT produced no NaNs; expected poisoned intermediate rows")
		}
	})

	// Mutation 2: corrupted need count (need + 1) exceeds producer signals.
	t.Run("corrupted_need_too_high", func(t *testing.T) {
		orig := fmt.Sprintf("eg_ready_wait(&ready[0], epoch, %d);", need)
		mut := fmt.Sprintf("eg_ready_wait(&ready[0], epoch, %d);", need+1)
		if !strings.Contains(baseSrc, orig) {
			t.Fatalf("missing %q in source", orig)
		}
		msrc := strings.Replace(baseSrc, orig, mut, 1)
		_, _, err := runMegakernelSourceEpochs(t, mega, msrc, envs[:1])
		if err == nil {
			t.Fatal("corrupted need+1 succeeded; expected timeout/wait error")
		}
	})

	// Mutation 3: corrupted need count (need - 1) waits for fewer producers than stage 0 launches.
	t.Run("corrupted_need_too_low", func(t *testing.T) {
		orig := fmt.Sprintf("eg_ready_wait(&ready[0], epoch, %d);", need)
		mut := fmt.Sprintf("eg_ready_wait(&ready[0], epoch, %d);", need-1)
		msrc := strings.Replace(baseSrc, orig, mut, 1)
		_, _, err := runMegakernelSourceEpochs(t, mega, msrc, envs[:1])
		if err == nil {
			t.Fatal("corrupted need-1 succeeded; expected need < expected error")
		}
	})

	// Mutation 4: waiting on stale epoch 1 during epoch 2 leaves epoch 2's
	// intermediate buffer poisoned with NaN.
	t.Run("stale_epoch_wait", func(t *testing.T) {
		orig := fmt.Sprintf("eg_ready_wait(&ready[0], epoch, %d);", need)
		mut := fmt.Sprintf("eg_ready_wait(&ready[0], 1u, %d);", need)
		msrc := strings.Replace(baseSrc, orig, mut, 1)
		outs, _, err := runMegakernelSourceEpochs(t, mega, msrc, envs)
		if err != nil {
			return
		}
		// Epoch 2 must read NaN because it waited on stale epoch 1 instead of epoch 2.
		nans := 0
		for _, v := range outs[1] {
			if math.IsNaN(v) {
				nans++
			}
		}
		if nans == 0 {
			t.Fatal("stale epoch wait in epoch 2 produced no NaNs")
		}
	})
}

func TestMegakernelErrors(t *testing.T) {
	if _, err := (*Plan)(nil).Megakernel("x"); err == nil {
		t.Error("nil plan: expected error")
	}
	if _, err := (&Plan{Name: "p"}).Megakernel("x"); err == nil {
		t.Error("empty plan: expected error")
	}
	validRow := RowKernel{
		Name:    "rk",
		D:       64,
		Inputs:  []RowTensor{{Name: "x", DType: F32, Kind: RowData}},
		Values:  []RowValue{{Name: "v", Body: Identity, Args: []string{"x"}}},
		Outputs: []RowTensor{{Name: "y", DType: F32, Kind: RowData, Value: "v"}},
	}
	p := &Plan{
		Name:    "p",
		Inputs:  []Tensor{{"x", F32, []int{2, 64}}},
		Output:  Tensor{"y", F32, []int{2, 64}},
		Kernels: []PlanKernel{{Row: &validRow, Rows: 2, Args: []string{"x", "y"}}},
	}
	if _, err := p.Megakernel("eg_reserved"); err == nil {
		t.Error("bad megakernel name: expected error")
	}
	clusterRow := validRow
	clusterRow.D = 2048
	clusterRow.Cluster = true
	pCluster := &Plan{
		Name:    "pc",
		Kernels: []PlanKernel{{Row: &clusterRow, Rows: 2, Args: []string{"x", "y"}}},
	}
	if _, err := pCluster.Megakernel("pc_mega"); err == nil {
		t.Error("cluster row stage: expected error")
	}
	pAttn := &Plan{
		Name:    "pa",
		Kernels: []PlanKernel{{Attention: &PagedAttention{Hq: 16, Hkv: 2}}},
	}
	if _, err := pAttn.Megakernel("pa_mega"); err == nil {
		t.Error("attention stage: expected error")
	}
}

func ExamplePlan_Megakernel() {
	b := NewBuilder()
	x := b.Input("x", BF16, 2, 256)
	w := b.Input("w", BF16, 64, 256)
	y := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), y))
	root := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), y, ss)
	expr, _ := Optimize(b.G, root)
	plan, _ := Compile(expr, "proj_norm", BF16)
	mega, _ := plan.Megakernel("proj_norm_mega")
	grid, block := mega.Grid()
	fmt.Println(len(mega.Stages), grid, block, mega.ReadyWords())
	// Output:
	// 2 [65 1 1] [256 1 1] 1
}
