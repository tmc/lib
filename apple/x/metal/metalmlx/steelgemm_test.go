package metalmlx

import (
	"math"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestMLXMetallibSteelGEMM runs MLX's steel GEMM out of mlx.metallib with the
// generated [GEMMParams], which is what that struct is measured for: the
// kernel reads the block by value, so a wrong offset gives wrong numbers
// rather than an error. It follows steel_matmul_regular_axpby in
// mlx/backend/metal/matmul.cpp: same kernel name, same function constants,
// same buffer indices, same grid.
func TestMLXMetallibSteelGEMM(t *testing.T) {
	d := testDevice(t)
	if got := d.Architecture(); len(got) == 0 || (got[len(got)-1] != 's' && got[len(got)-1] != 'd') {
		// GEMM_TPARAM_MACRO picks other tile sizes, and so another kernel,
		// on the small device classes.
		t.Skipf("tile sizes below are for the max and ultra classes, not %s", got)
	}
	lib, err := d.LibraryFromPath("mlx", metallibPath(t))
	if err != nil {
		t.Fatal(err)
	}

	// Tile parameters for float32, no transpose, on a large device.
	const bm, bn, bk, wm, wn = 64, 64, 16, 2, 2
	const name = "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2"

	const m, n, k = 128, 64, 96
	hasBatch, useOutSource, doAxpby := false, false, false
	alignM, alignN, alignK := m%bm == 0, n%bn == 0, k%bk == 0
	consts := []FunctionConstant{
		{Value: unsafe.Pointer(&hasBatch), Type: metal.MTLDataTypeBool, Index: 10},
		{Value: unsafe.Pointer(&useOutSource), Type: metal.MTLDataTypeBool, Index: 100},
		{Value: unsafe.Pointer(&doAxpby), Type: metal.MTLDataTypeBool, Index: 110},
		{Value: unsafe.Pointer(&alignM), Type: metal.MTLDataTypeBool, Index: 200},
		{Value: unsafe.Pointer(&alignN), Type: metal.MTLDataTypeBool, Index: 201},
		{Value: unsafe.Pointer(&alignK), Type: metal.MTLDataTypeBool, Index: 202},
	}
	hash := name + "_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t"
	kernel, err := d.KernelSpecialized(lib, name, hash, consts)
	if err != nil {
		t.Fatal(err)
	}

	a := d.Allocator()
	bufA, err := a.Malloc(m * k * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free(bufA)
	bufB, err := a.Malloc(k * n * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free(bufB)
	bufOut, err := a.Malloc(m * n * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free(bufOut)

	hostA := unsafe.Slice((*float32)(bufA.Contents()), m*k)
	hostB := unsafe.Slice((*float32)(bufB.Contents()), k*n)
	hostOut := unsafe.Slice((*float32)(bufOut.Contents()), m*n)
	for i := range hostA {
		hostA[i] = float32(i%7) - 3
	}
	for i := range hostB {
		hostB[i] = float32(i%5) * 0.5
	}
	for i := range hostOut {
		hostOut[i] = math.Float32frombits(0xdeadbeef)
	}

	tn, tm := (n+bn-1)/bn, (m+bm-1)/bm
	params := GEMMParams{
		M: m, N: n, K: k,
		Lda: k, Ldb: n, Ldd: n,
		TilesN: int32(tn), TilesM: int32(tm),
		BatchStrideA: m * k, BatchStrideB: k * n, BatchStrideD: m * n,
		SwizzleLog:             0,
		GemmKIterationsAligned: k / bk,
		BatchNdim:              1,
	}

	e, err := NewCommandEncoder(d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	e.SetComputePipelineState(kernel)
	e.SetInputBuffer(bufA, 0, 0)
	e.SetInputBuffer(bufB, 1, 0)
	e.SetOutputBuffer(bufOut, 3, 0)
	e.SetBytes(mustValueBytes(&params), 4)
	e.DispatchThreadgroups(Size{Width: uint(tn), Height: uint(tm), Depth: 1},
		Size{Width: 32, Height: wn, Depth: wm})
	if err := e.Synchronize(); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var want float32
			for p := 0; p < k; p++ {
				want += hostA[i*k+p] * hostB[p*n+j]
			}
			if got := hostOut[i*n+j]; math.Abs(float64(got-want)) > 1e-3 {
				t.Fatalf("out[%d,%d] = %v, want %v", i, j, got, want)
			}
		}
	}
}
