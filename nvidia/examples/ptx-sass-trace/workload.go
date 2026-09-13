package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"runtime/trace"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// BlockQ8_0 represents a quantized block of 32 int8 values with a float16 scale.
type BlockQ8_0 struct {
	D  uint16
	QS [32]int8
}

type workloadResult struct {
	Intervals map[string][]int64
	Verified  int
	MaxError  float64
}

func runWorkload(ctx context.Context, dev *cuda.Device, mod *cuda.Module, prof *cudaprof.Session, code *cudacapture.Code, cfg config) (res workloadResult, err error) {
	cols := 256
	numExperts := 256
	expertRows := 16
	q8Rows := 256
	normRows := 1
	n := cfg.elements
	if n >= numExperts*expertRows*cols {
		expertRows = n / (numExperts * cols)
	} else if n < 256*32 {
		cols = 32
		expertRows = max(1, n/(numExperts*cols))
	}
	iqCount := max(1, cols/16)

	// Host allocations and initialization
	xHost := make([]float32, normRows*cols)
	gammaHost := make([]float32, cols)
	for i := range xHost {
		xHost[i] = float32(i%128)*0.01 + 0.1
	}
	for i := range gammaHost {
		gammaHost[i] = 1.0 + 0.01*float32(i%8)
	}

	blocksPerRow := cols / 32
	wQ8Host := make([]BlockQ8_0, q8Rows*blocksPerRow)
	for b := range wQ8Host {
		wQ8Host[b].D = float32ToFloat16(0.05 * float32((b%7)+1))
		for i := 0; i < 32; i++ {
			wQ8Host[b].QS[i] = int8((i % 7) - 3)
		}
	}

	upHost := make([]float32, cols)
	for i := range upHost {
		upHost[i] = 0.5 + 0.005*float32(i%32)
	}

	expertIDs := [6]int32{3, 42, 89, 128, 201, 250}
	expertWeightCount := numExperts * expertRows * cols
	expertWeightsHost := make([]float32, expertWeightCount)
	for i := range expertWeightsHost {
		expertWeightsHost[i] = 0.001 * float32((i%31)-15)
	}

	iqQHost := make([]uint32, iqCount)
	iqAHost := make([]int8, iqCount*16)
	for i := range iqQHost {
		iqQHost[i] = uint32(0x1b2e3d4c ^ (i * 0x55555555))
	}
	for i := range iqAHost {
		iqAHost[i] = int8((i % 9) - 4)
	}
	scaleIQ := float32(0.125)

	smemAHost := make([]float32, expertRows*cols)
	for i := range smemAHost {
		smemAHost[i] = 0.002 * float32((i%17)-8)
	}

	nVec := min(n, 1024)
	aVecHost, bVecHost := make([]float32, nVec), make([]float32, nVec)
	for i := range aVecHost {
		aVecHost[i] = float32(i % 1024)
		bVecHost[i] = float32(2*(i%1024) + 1)
	}

	// Compute host references
	wantRMSNorm := refRMSNorm(xHost, gammaHost, normRows, cols, 1e-5)
	wantQ8 := refQ8_0Matvec(wQ8Host, wantRMSNorm[:cols], q8Rows, cols)
	wantSwiGLU := refSwiGLU(wantQ8[:cols], upHost)
	wantMoE := make([][]float32, 6)
	for slot := 0; slot < 6; slot++ {
		wantMoE[slot] = refMoEDispatch(expertWeightsHost, wantSwiGLU, expertIDs[:], slot, expertRows, cols)
	}
	wantIQ2 := refIQ2(iqQHost, iqAHost, scaleIQ)
	wantSmem := refSmemStageMatvec(smemAHost, wantSwiGLU, expertRows, cols)
	wantCompute := reference(cfg.rounds)

	// Device allocations
	dX, err := cudamem.Alloc[float32](dev, normRows*cols)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dX.Free()) }()

	dGamma, err := cudamem.Alloc[float32](dev, cols)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dGamma.Free()) }()

	dXNorm, err := cudamem.Alloc[float32](dev, normRows*cols)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dXNorm.Free()) }()

	dWQ8, err := cudamem.Alloc[BlockQ8_0](dev, len(wQ8Host))
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dWQ8.Free()) }()

	dH, err := cudamem.Alloc[float32](dev, q8Rows)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dH.Free()) }()

	dUp, err := cudamem.Alloc[float32](dev, cols)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dUp.Free()) }()

	dSwiGLU, err := cudamem.Alloc[float32](dev, cols)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dSwiGLU.Free()) }()

	dExpertWeights, err := cudamem.Alloc[float32](dev, len(expertWeightsHost))
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dExpertWeights.Free()) }()

	dExpertIDs, err := cudamem.Alloc[int32](dev, len(expertIDs))
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dExpertIDs.Free()) }()

	dMoEOut, err := cudamem.Alloc[float32](dev, 6*expertRows)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dMoEOut.Free()) }()

	dIQQ, err := cudamem.Alloc[uint32](dev, len(iqQHost))
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dIQQ.Free()) }()

	dIQA, err := cudamem.Alloc[int8](dev, len(iqAHost))
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dIQA.Free()) }()

	dIQOutSimd, err := cudamem.Alloc[float32](dev, iqCount)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dIQOutSimd.Free()) }()

	dIQOutScalar, err := cudamem.Alloc[float32](dev, iqCount)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dIQOutScalar.Free()) }()

	dSmemA, err := cudamem.Alloc[float32](dev, len(smemAHost))
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dSmemA.Free()) }()

	dSmemOut, err := cudamem.Alloc[float32](dev, expertRows)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dSmemOut.Free()) }()

	daVec, err := cudamem.Alloc[float32](dev, nVec)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, daVec.Free()) }()

	dbVec, err := cudamem.Alloc[float32](dev, nVec)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dbVec.Free()) }()

	dcVec, err := cudamem.Alloc[float32](dev, nVec)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dcVec.Free()) }()

	// Copy input data to device
	if err := dX.CopyFromHost(xHost); err != nil {
		return res, err
	}
	if err := dGamma.CopyFromHost(gammaHost); err != nil {
		return res, err
	}
	if err := dWQ8.CopyFromHost(wQ8Host); err != nil {
		return res, err
	}
	if err := dUp.CopyFromHost(upHost); err != nil {
		return res, err
	}
	if err := dExpertWeights.CopyFromHost(expertWeightsHost); err != nil {
		return res, err
	}
	if err := dExpertIDs.CopyFromHost(expertIDs[:]); err != nil {
		return res, err
	}
	if err := dIQQ.CopyFromHost(iqQHost); err != nil {
		return res, err
	}
	if err := dIQA.CopyFromHost(iqAHost); err != nil {
		return res, err
	}
	if err := dSmemA.CopyFromHost(smemAHost); err != nil {
		return res, err
	}
	if err := daVec.CopyFromHost(aVecHost); err != nil {
		return res, err
	}
	if err := dbVec.CopyFromHost(bVecHost); err != nil {
		return res, err
	}

	stream, err := dev.NewStream()
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, stream.Destroy()) }()

	start, err := dev.NewEvent()
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, start.Destroy()) }()

	end, err := dev.NewEvent()
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, end.Destroy()) }()

	res.Intervals = make(map[string][]int64)

	// Helper to lookup kernel & function
	getK := func(name string) (*cudacapture.Kernel, *cuda.Kernel, error) {
		k := code.Kernel(name)
		if k == nil {
			return nil, nil, fmt.Errorf("missing compiled kernel %s", name)
		}
		fn, err := mod.Function(name)
		if err != nil {
			return nil, nil, err
		}
		return k, fn, nil
	}

	kNorm, fnNorm, err := getK("ds4_rmsnorm")
	if err != nil {
		return res, err
	}
	kQ8, fnQ8, err := getK("ds4_q8_0_matvec")
	if err != nil {
		return res, err
	}
	kSwiGLU, fnSwiGLU, err := getK("ds4_swiglu")
	if err != nil {
		return res, err
	}
	kMoE, fnMoE, err := getK("ds4_moe_expert_dispatch")
	if err != nil {
		return res, err
	}
	kIQ2Simd, fnIQ2Simd, err := getK("ds4_iq2_simd")
	if err != nil {
		return res, err
	}
	kIQ2Scalar, fnIQ2Scalar, err := getK("ds4_iq2_scalar")
	if err != nil {
		return res, err
	}
	kSmem, fnSmem, err := getK("ds4_smem_stage_matvec")
	if err != nil {
		return res, err
	}
	kVecAdd, fnVecAdd, _ := getK("vector_add")
	kCompute, fnCompute, _ := getK("compute")

	normBuf := make([]float32, normRows*cols)
	q8Buf := make([]float32, q8Rows)
	swigluBuf := make([]float32, cols)
	moeBuf := make([]float32, 6*expertRows)
	iqSimdBuf := make([]float32, iqCount)
	iqScalarBuf := make([]float32, iqCount)
	smemBuf := make([]float32, expertRows)
	vecBuf := make([]float32, nVec)

	for pass := -1; pass < cfg.passes; pass++ {
		record := pass >= 0
		passStr := strconv.Itoa(pass)

		// 1. RMSNorm
		launchNorm := cudalaunch.OwnedKernel(fnNorm).Grid(uint32(normRows)).Block(256).On(stream)
		var nsNorm int64
		cudaprof.Do(ctx, cudaprof.Labels(
			"layer", "0",
			"phase", "norm",
			"kernel", "ds4_rmsnorm",
			"kernel.impl", "warp_reduce",
			"pass", passStr,
		), func(ctx context.Context) {
			nsNorm, err = runKernel(ctx, launchNorm, []any{dX, dGamma, dXNorm, int32(normRows), int32(cols), float32(1e-5)}, stream, start, end, kNorm, prof, record)
		})
		if err != nil {
			return res, err
		}
		if err := dXNorm.CopyToHost(normBuf); err != nil {
			return res, err
		}
		if err := verifySlice("ds4_rmsnorm", pass, normBuf, wantRMSNorm, 1e-4, &res); err != nil {
			return res, err
		}
		if record {
			res.Intervals["ds4_rmsnorm"] = append(res.Intervals["ds4_rmsnorm"], nsNorm)
		}

		// 2. Q8_0 Matvec
		launchQ8 := cudalaunch.OwnedKernel(fnQ8).Grid(uint32((q8Rows*32 + 255) / 256)).Block(256).On(stream)
		var nsQ8 int64
		cudaprof.Do(ctx, cudaprof.Labels(
			"layer", "0",
			"phase", "q8_0_matvec",
			"kernel", "ds4_q8_0_matvec",
			"kernel.impl", "simd_warp_reduce",
			"pass", passStr,
		), func(ctx context.Context) {
			nsQ8, err = runKernel(ctx, launchQ8, []any{dWQ8, dXNorm, dH, int32(q8Rows), int32(cols)}, stream, start, end, kQ8, prof, record)
		})
		if err != nil {
			return res, err
		}
		if err := dH.CopyToHost(q8Buf); err != nil {
			return res, err
		}
		if err := verifySlice("ds4_q8_0_matvec", pass, q8Buf, wantQ8, 1e-3, &res); err != nil {
			return res, err
		}
		if record {
			res.Intervals["ds4_q8_0_matvec"] = append(res.Intervals["ds4_q8_0_matvec"], nsQ8)
		}

		// 3. SwiGLU
		launchSwiGLU := cudalaunch.OwnedKernel(fnSwiGLU).Grid(uint32((cols + 255) / 256)).Block(256).On(stream)
		var nsSwiGLU int64
		cudaprof.Do(ctx, cudaprof.Labels(
			"layer", "0",
			"phase", "swiglu",
			"kernel", "ds4_swiglu",
			"kernel.impl", "elementwise",
			"pass", passStr,
		), func(ctx context.Context) {
			nsSwiGLU, err = runKernel(ctx, launchSwiGLU, []any{dH, dUp, dSwiGLU, int32(cols)}, stream, start, end, kSwiGLU, prof, record)
		})
		if err != nil {
			return res, err
		}
		if err := dSwiGLU.CopyToHost(swigluBuf); err != nil {
			return res, err
		}
		if err := verifySlice("ds4_swiglu", pass, swigluBuf, wantSwiGLU, 1e-4, &res); err != nil {
			return res, err
		}
		if record {
			res.Intervals["ds4_swiglu"] = append(res.Intervals["ds4_swiglu"], nsSwiGLU)
		}

		// 4. MoE Expert Dispatch for all 6 slots
		launchMoE := cudalaunch.OwnedKernel(fnMoE).Grid(uint32((expertRows + 255) / 256)).Block(256).On(stream)
		for slot := 0; slot < 6; slot++ {
			expertID := expertIDs[slot]
			var nsMoE int64
			cudaprof.Do(ctx, cudaprof.Labels(
				"moe.slot", strconv.Itoa(slot),
				"moe.expert", strconv.Itoa(int(expertID)),
				"kernel.impl", "simd",
				"layer", "0",
				"phase", "moe_dispatch",
				"kernel", "ds4_moe_expert_dispatch",
				"pass", passStr,
			), func(ctx context.Context) {
				nsMoE, err = runKernel(ctx, launchMoE, []any{dExpertWeights, dSwiGLU, dMoEOut, dExpertIDs, int32(slot), int32(expertRows), int32(cols)}, stream, start, end, kMoE, prof, record)
			})
			if err != nil {
				return res, err
			}
			if record {
				res.Intervals["ds4_moe_expert_dispatch"] = append(res.Intervals["ds4_moe_expert_dispatch"], nsMoE)
			}
		}
		if err := dMoEOut.CopyToHost(moeBuf); err != nil {
			return res, err
		}
		for slot := 0; slot < 6; slot++ {
			slotSlice := moeBuf[slot*expertRows : (slot+1)*expertRows]
			if err := verifySlice(fmt.Sprintf("ds4_moe_slot_%d", slot), pass, slotSlice, wantMoE[slot], 1e-4, &res); err != nil {
				return res, err
			}
		}

		// 5. Quantized Sub-Byte Dot Product: SIMD vs Scalar
		launchIQSimd := cudalaunch.OwnedKernel(fnIQ2Simd).Grid(uint32((iqCount + 255) / 256)).Block(256).On(stream)
		var nsIQSimd int64
		cudaprof.Do(ctx, cudaprof.Labels(
			"kernel", "ds4_iq2_simd",
			"phase", "quant_subbyte",
			"kernel.impl", "simd_dp4a",
			"pass", passStr,
		), func(ctx context.Context) {
			nsIQSimd, err = runKernel(ctx, launchIQSimd, []any{dIQQ, dIQA, dIQOutSimd, scaleIQ, int32(iqCount)}, stream, start, end, kIQ2Simd, prof, record)
		})
		if err != nil {
			return res, err
		}
		if record {
			res.Intervals["ds4_iq2_simd"] = append(res.Intervals["ds4_iq2_simd"], nsIQSimd)
		}

		launchIQScalar := cudalaunch.OwnedKernel(fnIQ2Scalar).Grid(uint32((iqCount + 255) / 256)).Block(256).On(stream)
		var nsIQScalar int64
		cudaprof.Do(ctx, cudaprof.Labels(
			"kernel", "ds4_iq2_scalar",
			"phase", "quant_subbyte",
			"kernel.impl", "scalar_loop",
			"pass", passStr,
		), func(ctx context.Context) {
			nsIQScalar, err = runKernel(ctx, launchIQScalar, []any{dIQQ, dIQA, dIQOutScalar, scaleIQ, int32(iqCount)}, stream, start, end, kIQ2Scalar, prof, record)
		})
		if err != nil {
			return res, err
		}
		if record {
			res.Intervals["ds4_iq2_scalar"] = append(res.Intervals["ds4_iq2_scalar"], nsIQScalar)
		}

		if err := dIQOutSimd.CopyToHost(iqSimdBuf); err != nil {
			return res, err
		}
		if err := dIQOutScalar.CopyToHost(iqScalarBuf); err != nil {
			return res, err
		}
		if err := verifySlice("ds4_iq2_simd", pass, iqSimdBuf, wantIQ2, 1e-4, &res); err != nil {
			return res, err
		}
		if err := verifySlice("ds4_iq2_scalar", pass, iqScalarBuf, wantIQ2, 1e-4, &res); err != nil {
			return res, err
		}

		// 6. Shared Memory Cooperative Staging
		launchSmem := cudalaunch.OwnedKernel(fnSmem).Grid(uint32(expertRows)).Block(256).SharedMem(uint32(cols * 4)).On(stream)
		var nsSmem int64
		cudaprof.Do(ctx, cudaprof.Labels(
			"kernel", "ds4_smem_stage_matvec",
			"phase", "smem_stage",
			"kernel.impl", "float4_smem",
			"pass", passStr,
		), func(ctx context.Context) {
			nsSmem, err = runKernel(ctx, launchSmem, []any{dSmemA, dSwiGLU, dSmemOut, int32(expertRows), int32(cols)}, stream, start, end, kSmem, prof, record)
		})
		if err != nil {
			return res, err
		}
		if err := dSmemOut.CopyToHost(smemBuf); err != nil {
			return res, err
		}
		if err := verifySlice("ds4_smem_stage_matvec", pass, smemBuf, wantSmem, 1e-4, &res); err != nil {
			return res, err
		}
		if record {
			res.Intervals["ds4_smem_stage_matvec"] = append(res.Intervals["ds4_smem_stage_matvec"], nsSmem)
		}

		// 7. Baseline vector_add & compute
		if fnVecAdd != nil && kVecAdd != nil {
			launchVec := cudalaunch.OwnedKernel(fnVecAdd).Grid(uint32((nVec + 255) / 256)).Block(256).On(stream)
			var nsVec int64
			cudaprof.Do(ctx, cudaprof.Labels("kernel", "vector_add", "phase", "memory", "pass", passStr), func(ctx context.Context) {
				nsVec, err = runKernel(ctx, launchVec, []any{daVec, dbVec, dcVec, int32(nVec)}, stream, start, end, kVecAdd, prof, record)
			})
			if err != nil {
				return res, err
			}
			if err := dcVec.CopyToHost(vecBuf); err != nil {
				return res, err
			}
			for i, got := range vecBuf {
				want := aVecHost[i] + bVecHost[i]
				diff := math.Abs(float64(got) - float64(want))
				if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || diff > 1e-5 {
					return res, fmt.Errorf("vector_add pass %d: c[%d] = %g, want %g", pass, i, got, want)
				}
				res.MaxError = math.Max(res.MaxError, diff)
				if record {
					res.Verified++
				}
			}
			if record {
				res.Intervals["vector_add"] = append(res.Intervals["vector_add"], nsVec)
			}
		}

		if fnCompute != nil && kCompute != nil {
			launchComp := cudalaunch.OwnedKernel(fnCompute).Grid(uint32((nVec + 255) / 256)).Block(256).On(stream)
			var nsComp int64
			cudaprof.Do(ctx, cudaprof.Labels("kernel", "compute", "phase", "compute", "pass", passStr), func(ctx context.Context) {
				nsComp, err = runKernel(ctx, launchComp, []any{dcVec, int32(nVec), int32(cfg.rounds)}, stream, start, end, kCompute, prof, record)
			})
			if err != nil {
				return res, err
			}
			if err := dcVec.CopyToHost(vecBuf); err != nil {
				return res, err
			}
			for i, got := range vecBuf {
				want := wantCompute[i%len(wantCompute)]
				tol := 1e-6*math.Abs(float64(want)) + 1e-6
				diff := math.Abs(float64(got) - float64(want))
				if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || diff > tol {
					return res, fmt.Errorf("compute pass %d: c[%d] = %g, want %g", pass, i, got, want)
				}
				res.MaxError = math.Max(res.MaxError, diff)
				if record {
					res.Verified++
				}
			}
			if record {
				res.Intervals["compute"] = append(res.Intervals["compute"], nsComp)
			}
		}

		trace.Logf(ctx, "cuda.validation", "pipeline pass=%d verified total=%d max_err=%g", pass, res.Verified, res.MaxError)
	}

	return res, nil
}

func verifySlice(name string, pass int, got, want []float32, tol float64, res *workloadResult) error {
	for i := range got {
		g, w := got[i], want[i]
		diff := math.Abs(float64(g) - float64(w))
		if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) || diff > tol {
			return fmt.Errorf("%s pass %d: [%d] = %g, want %g (diff %g > tol %g)", name, pass, i, g, w, diff, tol)
		}
		res.MaxError = math.Max(res.MaxError, diff)
		if pass >= 0 {
			res.Verified++
		}
	}
	return nil
}

func runKernel(ctx context.Context, launcher *cudalaunch.Launcher, args []any, stream *cuda.Stream, start, end *cuda.Event, k *cudacapture.Kernel, prof *cudaprof.Session, record bool) (int64, error) {
	region := trace.StartRegion(ctx, k.Name+": host launch and wait")
	defer region.End()
	if err := start.Record(stream); err != nil {
		return 0, err
	}
	if err := launcher.Launch(args...); err != nil {
		return 0, err
	}
	if err := end.Record(stream); err != nil {
		return 0, err
	}
	if err := end.Synchronize(ctx); err != nil {
		return 0, err
	}
	elapsed, err := end.ElapsedSince(start)
	if err != nil {
		return 0, err
	}
	if elapsed <= 0 {
		return 0, fmt.Errorf("%s: nonpositive event interval %s", k.Name, elapsed)
	}
	trace.Logf(ctx, "cuda.event", "kernel=%s measured=%v gpu_time=%s", k.Name, record, elapsed)
	if record {
		vals := cudaprof.SampleValues{
			GPUNanos: elapsed.Nanoseconds(),
		}
		sassLine := 1
		if len(k.Instructions) > 0 {
			sassLine = k.Instructions[0].SASSLine
		}
		_, _, line, _ := runtime.Caller(0)
		prof.RecordStackSample(ctx, vals, []cudaprof.Frame{
			{FunctionName: k.Name + ".SASS", FileName: "kernels.sass", Line: int64(sassLine), Address: cudaprof.DefaultMappingStart + k.Offset},
			{FunctionName: k.Name + ".PTX", FileName: "kernels.ptx", Line: int64(k.PTXLine)},
			{FunctionName: k.Name, FileName: "kernels.cu", Line: int64(k.SourceLine)},
			{FunctionName: "main.runKernel", FileName: "workload.go", Line: int64(line)},
		})
	}
	return elapsed.Nanoseconds(), nil
}

func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	frac := bits & 0x7fffff

	if exp <= 0 {
		return sign
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(frac>>13)
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	frac := uint32(h & 0x3ff)

	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		for (frac & 0x400) == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
		return math.Float32frombits(sign | ((uint32(exp) + 127 - 15) << 23) | (frac << 13))
	} else if exp == 31 {
		if frac == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7f800000 | (frac << 13))
	}
	return math.Float32frombits(sign | ((uint32(exp) - 15 + 127) << 23) | (frac << 13))
}

func refRMSNorm(x, gamma []float32, rows, cols int, eps float32) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		var sumsq float64
		for c := 0; c < cols; c++ {
			v := float64(x[r*cols+c])
			sumsq += v * v
		}
		invRMS := float32(1.0 / math.Sqrt(sumsq/float64(cols)+float64(eps)))
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			out[idx] = x[idx] * invRMS * gamma[c]
		}
	}
	return out
}

func refQ8_0Matvec(w []BlockQ8_0, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	blocksPerRow := cols / 32
	for r := 0; r < rows; r++ {
		var rowSum float64
		for b := 0; b < blocksPerRow; b++ {
			blk := w[r*blocksPerRow+b]
			scale := float64(float16ToFloat32(blk.D))
			var blockSum float64
			for i := 0; i < 32; i++ {
				blockSum += float64(blk.QS[i]) * float64(x[b*32+i])
			}
			rowSum += blockSum * scale
		}
		out[r] = float32(rowSum)
	}
	return out
}

func refSwiGLU(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	for i := range gate {
		g := float64(gate[i])
		silu := g / (1.0 + math.Exp(-g))
		out[i] = float32(silu * float64(up[i]))
	}
	return out
}

func refMoEDispatch(weights, x []float32, expertIDs []int32, slot, rows, cols int) []float32 {
	out := make([]float32, rows)
	expert := int(expertIDs[slot])
	for r := 0; r < rows; r++ {
		offset := (expert*rows + r) * cols
		var sum float64
		for c := 0; c < cols; c++ {
			sum += float64(weights[offset+c]) * float64(x[c])
		}
		out[r] = float32(sum)
	}
	return out
}

func refIQ2(q []uint32, a []int8, scale float32) []float32 {
	out := make([]float32, len(q))
	for idx := range q {
		qval := q[idx]
		var sum int32
		for i := 0; i < 16; i++ {
			shift := i * 2
			qi := int32((qval >> shift) & 0x3)
			ai := int32(a[idx*16+i])
			sum += qi * ai
		}
		out[idx] = float32(sum) * scale
	}
	return out
}

func refSmemStageMatvec(a, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var sum float64
		for c := 0; c < cols; c++ {
			sum += float64(a[r*cols+c]) * float64(x[c])
		}
		out[r] = float32(sum)
	}
	return out
}

func reference(rounds int) []float32 {
	out := make([]float32, 1024)
	for i := range out {
		v := float32(3*i + 1)
		for j := 0; j < rounds; j++ {
			v = float32(math.FMA(float64(v), float64(float32(1.000001)), float64(float32(0.0001))))
		}
		out[i] = v
	}
	return out
}
