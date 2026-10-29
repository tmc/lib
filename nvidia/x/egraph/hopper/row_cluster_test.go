package hopper

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"regexp"
	"strings"
	"testing"
)

// clusterPreNorm builds a whole-row RMSNorm * weight kernel over D columns.
func clusterPreNorm(d int, outDType DType) RowKernel {
	div := fmt.Sprintf("(mul $0 (rsqrt (add (div $1 %d) 1e-5)))", d)
	return RowKernel{
		Name:    "row_prenorm",
		D:       d,
		Cluster: true,
		Inputs: []RowTensor{
			{Name: "x", DType: BF16, Kind: RowData},
			{Name: "w", DType: BF16, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "ss", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"x"}},
			{Name: "n", Body: MustLambda(div), Args: []string{"x", "ss"}},
			{Name: "y", Body: MustLambda("(mul $0 $1)"), Args: []string{"n", "w"}},
		},
		Outputs: []RowTensor{{Name: "out", DType: outDType, Kind: RowData, Value: "y"}},
	}
}

// clusterBoundary builds a two-reduction residual + pre-norm boundary kernel
// over 8192 columns, optionally zero-padding a narrower branch of branchCols.
func clusterBoundary(branchCols int, outDType DType) RowKernel {
	return RowKernel{
		Name:    "row_boundary",
		D:       8192,
		Cluster: true,
		Inputs: []RowTensor{
			{Name: "resid", DType: F32, Kind: RowData},
			{Name: "branch", DType: BF16, Kind: RowData, Padded: branchCols},
			{Name: "post_w", DType: BF16, Kind: ColVector},
			{Name: "alpha", DType: F32, Kind: ColVector},
			{Name: "beta", DType: F32, Kind: ColVector},
			{Name: "pre_w", DType: BF16, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "b", Body: MustLambda("(bf16 (mul $0 $1))"), Args: []string{"branch", "post_w"}},
			{Name: "bsq", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"b"}},
			{Name: "bn", Body: MustLambda("(bf16 (mul $0 (rsqrt (add (div $1 8192) 1e-8))))"), Args: []string{"b", "bsq"}},
			{Name: "s1", Body: MustLambda("(add (mul $0 $1) (mul $2 $3))"), Args: []string{"alpha", "resid", "beta", "bn"}},
			{Name: "s1sq", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"s1"}},
			{Name: "h", Body: MustLambda("(mul (mul $0 (rsqrt (add (div $1 8192) 1e-5))) $2)"), Args: []string{"s1", "s1sq", "pre_w"}},
		},
		Outputs: []RowTensor{
			{Name: "resid_out", DType: F32, Kind: RowData, Value: "s1"},
			{Name: "normed", DType: outDType, Kind: RowData, Value: "h"},
		},
	}
}

// clusterSlotSum builds a multi-slot RMSNorm and weighted-sum kernel over D
// columns and k slots per row.
func clusterSlotSum(d, k int, outDType DType) RowKernel {
	inv := fmt.Sprintf("(mul (rsqrt (add (div $0 %d) 1e-8)) $1)", d)
	return RowKernel{
		Name:    "row_slot_sum",
		D:       d,
		Cluster: true,
		Slots:   k,
		SlotIDs: "ids",
		Inputs: []RowTensor{
			{Name: "y", DType: BF16, Kind: RowData},
			{Name: "w", DType: BF16, Kind: ColVector},
			{Name: "ids", DType: I32, Kind: RowScalar},
			{Name: "weights", DType: F32, Kind: RowScalar},
		},
		Values: []RowValue{
			{Name: "s", Body: MustLambda("(mul $0 (add 1 $1))"), Args: []string{"y", "w"}},
			{Name: "ss", Reduce: "sum", Body: MustLambda("(mul $0 $0)"), Args: []string{"s"}},
			{Name: "rw", Body: MustLambda(inv), Args: []string{"ss", "weights"}},
			{Name: "n", Body: MustLambda("(mul $0 $1)"), Args: []string{"s", "rw"}},
		},
		Outputs: []RowTensor{{Name: "out", DType: outDType, Kind: RowData, Value: "n"}},
	}
}

// spikeRows sets one element per row of vals (row width cols) to 4096 so its
// square (2^24) makes floating-point summation order-sensitive.
func spikeRows(r *rand.Rand, vals []float64, cols int) {
	rows := len(vals) / cols
	for row := range rows {
		vals[row*cols+r.IntN(cols)] = 4096
	}
}

func runRowVariant(t *testing.T, k RowKernel, src string, rows int, variant string, bufs map[string]typed) map[string][]byte {
	t.Helper()
	var eb []emuBuf
	var names []string
	for _, p := range k.params()[1:] {
		f := strings.Fields(p)
		name := f[len(f)-1]
		names = append(names, name)
		b := bufs[name]
		eb = append(eb, emuBuf{name, b.dtype.cType(), b.bytes()})
	}
	var launch string
	switch variant {
	case "cluster":
		grid, block := k.ClusterGrid(rows)
		launch = fmt.Sprintf("eg_emu_launch_cluster(dim3(%d, %d), dim3(%d), 0, %d, %s_cluster, %d, %s)",
			grid[0], grid[1], block[0], ClusterSize, k.Name, rows, strings.Join(names, ", "))
	case "wide":
		grid, block := k.WideGrid(rows)
		launch = fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s_wide, %d, %s)",
			grid[0], block[0], k.Name, rows, strings.Join(names, ", "))
	case "launch":
		launch = fmt.Sprintf("%s_launch(%d, %s, nullptr)", k.Name, rows, strings.Join(names, ", "))
	default:
		grid, block := k.Grid(rows)
		launch = fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
			grid[0], block[0], k.Name, rows, strings.Join(names, ", "))
	}
	return runEmu(t, src, launch, eb...)
}

// TestRowClusterBitwiseAndReference checks that the 8-CTA cluster variant and
// the 1-block wide variant are bitwise equal on order-sensitive spiked inputs
// with FP32 outputs, and that the base, cluster and wide variants all agree
// with Reference.
func TestRowClusterBitwiseAndReference(t *testing.T) {
	r := rand.New(rand.NewPCG(42, 99))

	t.Run("prenorm_4096_f32", func(t *testing.T) {
		const rows, d = 4, 4096
		k := clusterPreNorm(d, F32)
		src, err := k.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		x := randVals(r, rows*d, 2)
		spikeRows(r, x, d)
		bufs := map[string]typed{
			"x":   {BF16, x},
			"w":   {BF16, randVals(r, d, 0.2)},
			"out": {F32, make([]float64, rows*d)},
		}
		cRes := runRowVariant(t, k, src, rows, "cluster", bufs)
		wRes := runRowVariant(t, k, src, rows, "wide", bufs)
		if !bytes.Equal(cRes["out"], wRes["out"]) {
			t.Fatalf("cluster and wide outputs differ on spiked FP32 prenorm")
		}
		got, want := runRowKernel(t, k, rows, bufs)
		agree(t, "base", F32, got["out"], want["out"])
		agree(t, "cluster", F32, decode(F32, cRes["out"]), want["out"])
	})

	t.Run("boundary_8192_and_padded_slice1024", func(t *testing.T) {
		for _, sliceCols := range []int{0, 1024} {
			name := "full"
			bw := 8192
			if sliceCols > 0 {
				name = "slice1024"
				bw = sliceCols
			}
			t.Run(name, func(t *testing.T) {
				const rows, d = 3, 8192
				k := clusterBoundary(sliceCols, F32)
				src, err := k.CUDA()
				if err != nil {
					t.Fatal(err)
				}
				resid := randVals(r, rows*d, 1)
				branch := randVals(r, rows*bw, 3)
				spikeRows(r, resid, d)
				spikeRows(r, branch, bw)
				bufs := map[string]typed{
					"resid":     {F32, resid},
					"branch":    {BF16, branch},
					"post_w":    {BF16, randVals(r, d, 0.5)},
					"alpha":     {F32, randVals(r, d, 0.5)},
					"beta":      {F32, randVals(r, d, 0.5)},
					"pre_w":     {BF16, randVals(r, d, 0.5)},
					"resid_out": {F32, make([]float64, rows*d)},
					"normed":    {F32, make([]float64, rows*d)},
				}
				cRes := runRowVariant(t, k, src, rows, "cluster", bufs)
				wRes := runRowVariant(t, k, src, rows, "wide", bufs)
				for _, out := range []string{"resid_out", "normed"} {
					if !bytes.Equal(cRes[out], wRes[out]) {
						t.Fatalf("%s: cluster and wide differ on spiked FP32 %s", name, out)
					}
				}
				got, want := runRowKernel(t, k, rows, bufs)
				for _, out := range []string{"resid_out", "normed"} {
					agree(t, out+"/base", F32, got[out], want[out])
					agree(t, out+"/cluster", F32, decode(F32, cRes[out]), want[out])
				}
			})
		}
	})

	t.Run("slot_sum_4096_k8_f32", func(t *testing.T) {
		const rows, d, slots = 4, 4096, 8
		k := clusterSlotSum(d, slots, F32)
		src, err := k.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		y := randVals(r, rows*slots*d, 2)
		spikeRows(r, y, d)
		ids := make([]float64, rows*slots)
		weights := make([]float64, rows*slots)
		for i := range ids {
			if i < slots || r.IntN(3) == 0 {
				ids[i] = -1 // row 0 has no active slots; ~1/3 skipped elsewhere
			} else {
				ids[i] = float64(i % 16)
			}
			weights[i] = r.Float64()
		}
		bufs := map[string]typed{
			"y":       {BF16, y},
			"w":       {BF16, randVals(r, d, 0.1)},
			"ids":     {I32, ids},
			"weights": {F32, weights},
			"out":     {F32, make([]float64, rows*d)},
		}
		cRes := runRowVariant(t, k, src, rows, "cluster", bufs)
		wRes := runRowVariant(t, k, src, rows, "wide", bufs)
		if !bytes.Equal(cRes["out"], wRes["out"]) {
			t.Fatalf("slot_sum: cluster and wide differ on spiked FP32")
		}
		got, want := runRowKernel(t, k, rows, bufs)
		agree(t, "base", F32, got["out"], want["out"])
		agree(t, "cluster", F32, decode(F32, cRes["out"]), want["out"])
		for c := range d {
			if got["out"][c] != 0 {
				t.Fatalf("row 0 (all slots skipped) out[%d] = %g, want 0", c, got["out"][c])
			}
		}
	})
}

// TestRowClusterLauncherDispatch verifies that <Name>_launch dispatches to the
// 8-CTA cluster kernel at rows <= ClusterMaxRows and the single-block wide
// kernel at rows > ClusterMaxRows.
func TestRowClusterLauncherDispatch(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 11))
	k := clusterPreNorm(2048, BF16)
	k.ClusterMaxRows = 4
	cuda, err := k.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	l, err := k.Launcher()
	if err != nil {
		t.Fatal(err)
	}
	src := cuda + "\n" + l

	for _, rows := range []int{3, 4, 5} {
		bufs := map[string]typed{
			"x":   {BF16, randVals(r, rows*k.D, 1)},
			"w":   {BF16, randVals(r, k.D, 0.2)},
			"out": {BF16, make([]float64, rows*k.D)},
		}
		wantVariant := "cluster"
		if rows > k.ClusterMaxRows {
			wantVariant = "wide"
		}
		got := runRowVariant(t, k, src, rows, "launch", bufs)
		want := runRowVariant(t, k, cuda, rows, wantVariant, bufs)
		if !bytes.Equal(got["out"], want["out"]) {
			t.Errorf("rows=%d: launcher output differs from %s variant", rows, wantVariant)
		}
	}
}

// TestRowClusterMutations checks that order and boundary mutations in the
// cluster/wide kernels are caught by the spiked FP32 tests.
func TestRowClusterMutations(t *testing.T) {
	r := rand.New(rand.NewPCG(19, 23))

	// Mutation 1: changing block_sum8's group reduction order breaks bitwise
	// equality with cluster_sum on spiked FP32 prenorm.
	{
		const rows, d = 4, 4096
		k := clusterPreNorm(d, F32)
		src, err := k.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		const target = "for (int m = 4; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);"
		const mutant = "for (int m = 1; m <= 4; m <<= 1) s += __shfl_xor_sync(0xffffffffu, s, m);"
		// Replace only the second occurrence (inside eg_block_sum8).
		idx := strings.LastIndex(src, target)
		if idx < 0 {
			t.Fatal("target not found in ClusterPrelude")
		}
		mutSrc := src[:idx] + mutant + src[idx+len(target):]
		x := randVals(r, rows*d, 2)
		spikeRows(r, x, d)
		bufs := map[string]typed{
			"x":   {BF16, x},
			"w":   {BF16, randVals(r, d, 0.2)},
			"out": {F32, make([]float64, rows*d)},
		}
		cRes := runRowVariant(t, k, mutSrc, rows, "cluster", bufs)
		wRes := runRowVariant(t, k, mutSrc, rows, "wide", bufs)
		if bytes.Equal(cRes["out"], wRes["out"]) {
			t.Errorf("butterfly order mutant in eg_block_sum8 survived")
		}
	}

	// Mutation 2: reversing rank summation order in eg_block_sum8_slots breaks
	// bitwise equality with eg_cluster_sum_slots on spiked FP32 slot sum.
	{
		const rows, d, slots = 4, 4096, 8
		k := clusterSlotSum(d, slots, F32)
		src, err := k.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		const target = "for (int r = 0; r < 8; ++r) {\n            float s = part[r * NW][threadIdx.x];"
		const mutant = "for (int r = 7; r >= 0; --r) {\n            float s = part[r * NW][threadIdx.x];"
		if !strings.Contains(src, target) {
			t.Fatal("slot target not found in ClusterPrelude")
		}
		mutSrc := strings.Replace(src, target, mutant, 1)
		y := randVals(r, rows*slots*d, 2)
		spikeRows(r, y, d)
		ids := make([]float64, rows*slots)
		weights := make([]float64, rows*slots)
		for i := range ids {
			ids[i] = float64(i % 8)
			weights[i] = 0.3 + 0.1*float64(i%5)
		}
		bufs := map[string]typed{
			"y":       {BF16, y},
			"w":       {BF16, randVals(r, d, 0.1)},
			"ids":     {I32, ids},
			"weights": {F32, weights},
			"out":     {F32, make([]float64, rows*d)},
		}
		cRes := runRowVariant(t, k, mutSrc, rows, "cluster", bufs)
		wRes := runRowVariant(t, k, mutSrc, rows, "wide", bufs)
		if bytes.Equal(cRes["out"], wRes["out"]) {
			t.Errorf("rank order mutant in eg_block_sum8_slots survived")
		}
	}

	// Mutation 3: reading one vector chunk past Padded width fails.
	{
		const rows, d, sliceCols = 2, 8192, 1024
		k := clusterBoundary(sliceCols, F32)
		src, err := k.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		const target = "if (EG_COL(j) < 1024)"
		const mutant = "if (EG_COL(j) < 1032)"
		if !strings.Contains(src, target) {
			t.Fatal("padded guard not found")
		}
		mutSrc := strings.ReplaceAll(src, target, mutant)
		bufs := map[string]typed{
			"resid":     {F32, randVals(r, rows*d, 1)},
			"branch":    {BF16, randVals(r, rows*sliceCols+64, 5)},
			"post_w":    {BF16, randVals(r, d, 0.5)},
			"alpha":     {F32, randVals(r, d, 0.5)},
			"beta":      {F32, randVals(r, d, 0.5)},
			"pre_w":     {BF16, randVals(r, d, 0.5)},
			"resid_out": {F32, make([]float64, rows*d)},
			"normed":    {F32, make([]float64, rows*d)},
		}
		good := runRowVariant(t, k, src, rows, "cluster", bufs)
		bad := runRowVariant(t, k, mutSrc, rows, "cluster", bufs)
		if bytes.Equal(good["resid_out"], bad["resid_out"]) {
			t.Errorf("padded out-of-bounds mutant survived")
		}
	}
}

// TestRowClusterStructure verifies the emitted source rules for cluster and
// wide kernels:
//   - __cluster_dims__(8, 1, 1) and __launch_bounds__(threads, 1) on cluster kernels
//   - all global loads (eg_load and scalar array loads) occur before the first
//     cluster reduction / barrier
//   - 16-byte write-back vectorized stores (eg_store_wb / st.global.wb.v4.u32) are used
func TestRowClusterStructure(t *testing.T) {
	kernels := []struct {
		k          RowKernel
		ctaThreads int
	}{
		{clusterPreNorm(4096, BF16), 64},
		{clusterBoundary(0, BF16), 128},
		{clusterBoundary(1024, BF16), 128},
		{clusterSlotSum(4096, 8, BF16), 64},
	}
	loadRe := regexp.MustCompile(`\beg_load<|\b_slot\[k\]\s*=\s*\(`)
	barrierRe := regexp.MustCompile(`\beg_cluster_sum|\beg_cluster_max|\beg_cluster_sync|\b__syncthreads`)

	for _, tc := range kernels {
		t.Run(tc.k.Name, func(t *testing.T) {
			csrc, err := tc.k.ClusterCUDA()
			if err != nil {
				t.Fatal(err)
			}
			wsrc, err := tc.k.WideCUDA()
			if err != nil {
				t.Fatal(err)
			}
			wantHeader := fmt.Sprintf("__global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(%d, 1) %s_cluster(",
				tc.ctaThreads, tc.k.Name)
			if !strings.Contains(csrc, wantHeader) {
				t.Errorf("cluster kernel missing header %q in:\n%s", wantHeader, csrc)
			}
			wantWide := fmt.Sprintf("__global__ void __launch_bounds__(%d) %s_wide(",
				ClusterSize*tc.ctaThreads, tc.k.Name)
			if !strings.Contains(wsrc, wantWide) {
				t.Errorf("wide kernel missing header %q in:\n%s", wantWide, wsrc)
			}
			if !strings.Contains(csrc, "eg_store_wb<VEC>") || !strings.Contains(wsrc, "eg_store_wb<VEC>") {
				t.Errorf("missing vectorized eg_store_wb<VEC>")
			}
			if !strings.Contains(ClusterPrelude, "st.global.wb.v4.u32") {
				t.Errorf("ClusterPrelude missing st.global.wb.v4.u32")
			}

			// Check that every global load in the cluster kernel appears before
			// the first cluster reduction / barrier.
			firstBar := barrierRe.FindStringIndex(csrc)
			if firstBar == nil {
				t.Fatalf("no cluster reduction/barrier found in %s_cluster", tc.k.Name)
			}
			afterBar := csrc[firstBar[0]:]
			if m := loadRe.FindString(afterBar); m != "" {
				t.Errorf("%s_cluster has global load %q after first barrier:\n%s", tc.k.Name, m, afterBar)
			}
		})
	}
}
