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

func runPDLPlanChain(t *testing.T, pdl *PDLPlanSource, env map[string]Value) ([]float64, error) {
	t.Helper()
	var args []cudahostemu.Arg
	for _, in := range pdl.Inputs {
		args = append(args, cudahostemu.BufferArg(
			in.Name,
			in.DType.cType(),
			typed{in.DType, env[in.Name].Data}.bytes(),
		))
	}
	for _, tmp := range pdl.Temps {
		nbytes := size(tmp.Shape) * tmp.DType.Bits() / 8
		args = append(args, cudahostemu.BufferArg(
			tmp.Name,
			tmp.DType.cType(),
			poison(nbytes),
		))
	}
	outBytes := poison(size(pdl.Output.Shape) * pdl.Output.DType.Bits() / 8)
	args = append(args, cudahostemu.BufferArg(
		pdl.Output.Name,
		pdl.Output.DType.cType(),
		outBytes,
	))

	stages := make([]cudahostemu.PDLStage, len(pdl.Stages))
	for i, st := range pdl.Stages {
		var bufNames []string
		for _, out := range st.Outputs {
			bufNames = append(bufNames, out.Name)
		}
		sid := st.TriggerStage
		if sid == 0 {
			sid = st.WaitStage
		}
		stages[i] = cudahostemu.PDLStage{
			Kernel:      st.Name,
			Grid:        cudahostemu.Dim3{X: st.Grid[0], Y: st.Grid[1], Z: st.Grid[2]},
			Block:       cudahostemu.Dim3{X: st.Block[0], Y: st.Block[1], Z: st.Block[2]},
			SharedBytes: st.SharedBytes,
			StageID:     sid,
			Buffers:     bufNames,
			Args:        st.Args,
		}
	}

	err := cudahostemu.RunPDLChain(t.Context(), pdl.Source, stages, args...)
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated PDL chain not run")
	}
	return decode(pdl.Output.DType, outBytes), err
}

func runPDLPlanLauncher(t *testing.T, pdl *PDLPlanSource, env map[string]Value) []float64 {
	t.Helper()
	var cbufs []cudahostemu.Buffer
	var callArgs []string
	for _, in := range pdl.Inputs {
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  in.Name,
			CType: in.DType.cType(),
			Data:  typed{in.DType, env[in.Name].Data}.bytes(),
		})
		callArgs = append(callArgs, in.Name)
	}
	for _, tmp := range pdl.Temps {
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: tmp.DType.cType(),
			Data:  poison(size(tmp.Shape) * tmp.DType.Bits() / 8),
		})
		callArgs = append(callArgs, tmp.Name)
	}
	cbufs = append(cbufs, cudahostemu.Buffer{
		Name:  pdl.Output.Name,
		CType: pdl.Output.DType.cType(),
		Data:  poison(size(pdl.Output.Shape) * pdl.Output.DType.Bits() / 8),
	})
	callArgs = append(callArgs, pdl.Output.Name, "nullptr")

	launch := fmt.Sprintf("launch_%s_pdl(%s); if (eg_emu_pdl_violation_count() != 0) std::_Exit(4)",
		pdl.Name, strings.Join(callArgs, ", "))
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  cudahostemu.PDLHeader() + "\n" + pdl.Source,
		Launch:  launch,
		Buffers: cbufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated PDL launcher not run")
	}
	if err != nil {
		t.Fatalf("emulated launch_%s_pdl: %v", pdl.Name, err)
	}
	return decode(pdl.Output.DType, res[pdl.Output.Name])
}

func TestPDLSkinnyGEMMThenRMSNorm(t *testing.T) {
	const (
		m    = 16
		kDim = 256
		nOut = 64
	)
	b := NewBuilder()
	x := b.Input("x", BF16, m, kDim)
	w := b.Input("w", BF16, nOut, kDim)
	gain := b.Input("gain", BF16, nOut)
	proj := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), proj))
	root := b.Map(MustLambda("(mul (mul $0 (rsqrt (add (div $1 64) 1e-5))) (add 1 $2))"), proj, ss, gain)

	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "skinny_rmsnorm", BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Steps()) != 2 || plan.Steps()[0].GEMM == nil || plan.Steps()[1].Row == nil {
		t.Fatalf("plan has %d steps, want [SkinnyGEMM, RowKernel]", len(plan.Steps()))
	}

	pdl, err := plan.EmitPDL("skinny_rmsnorm")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"griddepcontrol.launch_dependents",
		"griddepcontrol.wait",
		"cudaLaunchAttributeProgrammaticStreamSerialization",
		"launch_skinny_rmsnorm_pdl",
		"eg_pdl_trigger(1u);",
		"eg_pdl_wait(1u);",
	} {
		if !strings.Contains(pdl.Source, want) {
			t.Fatalf("EmitPDL source missing %q", want)
		}
	}

	r := rand.New(rand.NewPCG(101, 102))
	env := randEnv(r,
		Tensor{"x", BF16, []int{m, kDim}},
		Tensor{"w", BF16, []int{nOut, kDim}},
		Tensor{"gain", BF16, []int{nOut}},
	)

	seqOut := runPlan(t, plan, env)
	pdlChainOut, err := runPDLPlanChain(t, pdl, env)
	if err != nil {
		t.Fatalf("RunPDLChain failed: %v", err)
	}
	agree(t, "skinny_rmsnorm_pdl_chain_vs_seq", BF16, pdlChainOut, seqOut)

	pdlLauncherOut := runPDLPlanLauncher(t, pdl, env)
	agree(t, "skinny_rmsnorm_pdl_launcher_vs_seq", BF16, pdlLauncherOut, seqOut)

	// Adversarial check: omitting eg_pdl_wait(1u) must poison output and fail RunPDLChain.
	t.Run("adversarial_missing_wait", func(t *testing.T) {
		mut := *pdl
		mut.Source = strings.Replace(pdl.Source, "eg_pdl_wait(1u);", "(void)0;", 1)
		out, err := runPDLPlanChain(t, &mut, env)
		if err == nil {
			t.Fatal("expected PDL ordering violation when eg_pdl_wait(1u) is removed")
		}
		for i, v := range out {
			if !math.IsNaN(v) {
				t.Fatalf("out[%d] = %g, want poisoned NaN", i, v)
			}
		}
	})
}

func TestPDLTileGEMMThenSwiGLU(t *testing.T) {
	const (
		m      = 16
		kDim   = 64
		nTotal = 128 // 64 gate + 64 up
		dOut   = 64
	)
	tile := TileGEMM{
		Name:   "gate_up_tile",
		N:      nTotal,
		K:      kDim,
		Weight: WeightBF16,
		Out:    F32,
		BM:     64,
		BN:     64,
		Stages: 2,
	}
	swigluRow := RowKernel{
		Name: "swiglu_row",
		D:    dOut,
		Inputs: []RowTensor{
			{Name: "gate", Buffer: "gate_up", DType: F32, Kind: RowData, Offset: 0, Group: 1, GroupStride: nTotal, RowStride: nTotal},
			{Name: "up", Buffer: "gate_up", DType: F32, Kind: RowData, Offset: dOut, Group: 1, GroupStride: nTotal, RowStride: nTotal},
		},
		Values: []RowValue{
			{Name: "act", Body: MustLambda("(mul (silu $0) $1)"), Args: []string{"gate", "up"}},
		},
		Outputs: []RowTensor{
			{Name: "out", DType: BF16, Kind: RowData, Value: "act"},
		},
	}
	plan := &Plan{
		Name: "tile_swiglu",
		Inputs: []Tensor{
			{"x", BF16, []int{m, kDim}},
			{"w", BF16, []int{nTotal, kDim}},
		},
		Temps: []Tensor{
			{"gate_up", F32, []int{m, nTotal}},
		},
		Output: Tensor{"out", BF16, []int{m, dOut}},
		Kernels: []PlanKernel{
			{
				Tile: &tile,
				M:    m,
				Args: []string{"x", "w", "gate_up"},
			},
			{
				Row:  &swigluRow,
				Rows: m,
				Args: []string{"gate_up", "out"},
			},
		},
	}

	pdl, err := plan.EmitPDL("tile_swiglu")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"griddepcontrol.launch_dependents",
		"griddepcontrol.wait",
		"cudaLaunchAttributeProgrammaticStreamSerialization",
		"launch_tile_swiglu_pdl",
		"eg_pdl_trigger(1u);",
		"eg_pdl_wait(1u);",
	} {
		if !strings.Contains(pdl.Source, want) {
			t.Fatalf("EmitPDL source missing %q", want)
		}
	}

	r := rand.New(rand.NewPCG(201, 202))
	env := randEnv(r,
		Tensor{"x", BF16, []int{m, kDim}},
		Tensor{"w", BF16, []int{nTotal, kDim}},
	)

	seqOut := runPlan(t, plan, env)
	pdlChainOut, err := runPDLPlanChain(t, pdl, env)
	if err != nil {
		t.Fatalf("RunPDLChain failed: %v", err)
	}
	agree(t, "tile_swiglu_pdl_chain_vs_seq", BF16, pdlChainOut, seqOut)

	pdlLauncherOut := runPDLPlanLauncher(t, pdl, env)
	agree(t, "tile_swiglu_pdl_launcher_vs_seq", BF16, pdlLauncherOut, seqOut)

	// Adversarial check: omitting eg_pdl_trigger(1u) must poison output and fail RunPDLChain.
	t.Run("adversarial_missing_trigger", func(t *testing.T) {
		mut := *pdl
		mut.Source = strings.Replace(pdl.Source, "eg_pdl_trigger(1u);", "(void)0;", 1)
		out, err := runPDLPlanChain(t, &mut, env)
		if err == nil {
			t.Fatal("expected PDL ordering violation when eg_pdl_trigger(1u) is removed")
		}
		for i, v := range out {
			if !math.IsNaN(v) {
				t.Fatalf("out[%d] = %g, want poisoned NaN", i, v)
			}
		}
	})
}

func TestPDLErrors(t *testing.T) {
	if _, err := (*Plan)(nil).EmitPDL("x"); err == nil {
		t.Error("nil plan: expected error")
	}
	if _, err := (&Plan{Name: "p"}).EmitPDL("x"); err == nil {
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
		Inputs:  []Tensor{{"x", F32, []int{16, 64}}},
		Output:  Tensor{"y", F32, []int{16, 64}},
		Kernels: []PlanKernel{{Row: &validRow, Rows: 16, Args: []string{"x", "y"}}},
	}
	if _, err := p.EmitPDL("eg_reserved"); err == nil {
		t.Error("bad PDL plan name: expected error")
	}
}

func ExamplePlan_EmitPDL() {
	b := NewBuilder()
	x := b.Input("x", BF16, 16, 256)
	w := b.Input("w", BF16, 64, 256)
	y := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), y))
	root := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), y, ss)
	expr, _ := Optimize(b.G, root)
	plan, _ := Compile(expr, "proj_norm", BF16)
	pdl, _ := plan.EmitPDL("proj_norm")
	fmt.Println(len(pdl.Stages), pdl.Stages[0].TriggerStage, pdl.Stages[1].WaitStage)
	// Output:
	// 2 1 1
}
