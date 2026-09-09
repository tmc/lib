package main

import (
	"context"
	"fmt"
	"math"
	"runtime/pprof"
	"runtime/trace"
	"strconv"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalcapture"
	"github.com/tmc/lib/apple/x/metal/metalprof"
)

type workloadResult struct {
	Intervals       map[string][]int64
	Verified        int
	MaxError        float64
	RequestedTiming string
	ActualTiming    string
	TimingScope     string
	FallbackReason  string
}

func runWorkload(ctx context.Context, dev *metal.Device, lib apple.MTLLibrary, profBuilder *metalprof.Builder, code *metalcapture.Code, cfg config) (res workloadResult, err error) {
	res.RequestedTiming = cfg.timing
	n := cfg.elements
	a := make([]float32, n)
	b := make([]float32, n)
	c := make([]float32, n)
	for i := range a {
		a[i] = float32(i % 1024)
		b[i] = float32(2*(i%1024) + 1)
	}

	wantCompute := reference(cfg.rounds)

	bufA, err := dev.AllocBuffer(n * 4)
	if err != nil {
		return res, err
	}
	defer objectivec.ObjectFromID(bufA.GetID()).Release()

	bufB, err := dev.AllocBuffer(n * 4)
	if err != nil {
		return res, err
	}
	defer objectivec.ObjectFromID(bufB.GetID()).Release()

	bufC, err := dev.AllocBuffer(n * 4)
	if err != nil {
		return res, err
	}
	defer objectivec.ObjectFromID(bufC.GetID()).Release()

	psoVectorAdd, err := dev.NewComputePipeline(lib, "vector_add")
	if err != nil {
		return res, err
	}
	defer objectivec.ObjectFromID(psoVectorAdd.GetID()).Release()

	psoCompute, err := dev.NewComputePipeline(lib, "compute")
	if err != nil {
		return res, err
	}
	defer objectivec.ObjectFromID(psoCompute.GetID()).Release()

	// Hardware sample buffer if supported and requested.
	// Explicit counter timing MUST be rejected before allocating sample buffers
	// if the required stage boundary sampling point is unsupported.
	var sampleBuf *metalprof.SampleBuffer
	supportsStage := dev.Raw().SupportsCounterSampling(apple.MTLCounterSamplingPointAtStageBoundary)

	switch cfg.timing {
	case "counter":
		if !supportsStage {
			return res, fmt.Errorf("explicit counter timing requested but device does not support stage boundary sampling")
		}
		sb, err := metalprof.NewSampleBuffer(dev, 4)
		if err != nil {
			return res, fmt.Errorf("explicit counter timing requested but failed to create sample buffer: %w", err)
		}
		sampleBuf = sb
		res.ActualTiming = "counter"
		res.TimingScope = "stage_boundary"
	case "command-buffer":
		res.ActualTiming = "command-buffer"
		res.TimingScope = "command_buffer"
	case "auto":
		if supportsStage {
			if sb, err := metalprof.NewSampleBuffer(dev, 4); err == nil {
				sampleBuf = sb
				res.ActualTiming = "counter"
				res.TimingScope = "stage_boundary"
			} else {
				res.ActualTiming = "command-buffer"
				res.TimingScope = "command_buffer"
				res.FallbackReason = fmt.Sprintf("sample buffer creation failed: %v", err)
			}
		} else {
			res.ActualTiming = "command-buffer"
			res.TimingScope = "command_buffer"
			res.FallbackReason = "device does not support stage boundary sampling"
		}
	default:
		return res, fmt.Errorf("unknown timing mode: %q", cfg.timing)
	}

	res.Intervals = make(map[string][]int64)

	// Copy initial inputs into Metal buffers
	copySliceToBuffer(bufA, a)
	copySliceToBuffer(bufB, b)

	for pass := -1; pass < cfg.passes; pass++ {
		// Stage 1: Vector Add
		var stAdd stageTiming
		labelsAdd := pprof.Labels("kernel", "vector_add", "phase", "memory", "pass", strconv.Itoa(pass))
		pprof.Do(ctx, labelsAdd, func(ctx context.Context) {
			stAdd, err = runStage(ctx, dev, psoVectorAdd, bufA, bufB, bufC, n, cfg.rounds, false, sampleBuf, cfg.timing)
		})
		if err != nil {
			return res, fmt.Errorf("vector_add pass %d: %w", pass, err)
		}
		if pass >= 0 {
			if res.ActualTiming == "" {
				res.ActualTiming = stAdd.mode
				res.TimingScope = stAdd.scope
			} else if res.ActualTiming != stAdd.mode {
				res.ActualTiming = "mixed"
			}
			if stAdd.reason != "" && res.FallbackReason == "" {
				res.FallbackReason = stAdd.reason
			}
		}

		copyBufferToSlice(c, bufC)
		for i, got := range c {
			want := a[i] + b[i]
			diff := math.Abs(float64(got) - float64(want))
			if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || diff > 1e-5 {
				return res, fmt.Errorf("vector_add pass %d verification failed at [%d]: got %g, want %g", pass, i, got, want)
			}
			res.MaxError = math.Max(res.MaxError, diff)
		}
		if pass >= 0 {
			res.Intervals["vector_add"] = append(res.Intervals["vector_add"], stAdd.nanos)
			res.Verified += n
			recordSample(profBuilder, "vector_add", stAdd.nanos, pass, "memory")
		}

		// Stage 2: Compute (dependent FMA chain)
		var stCompute stageTiming
		labelsCompute := pprof.Labels("kernel", "compute", "phase", "compute", "pass", strconv.Itoa(pass))
		pprof.Do(ctx, labelsCompute, func(ctx context.Context) {
			stCompute, err = runStage(ctx, dev, psoCompute, bufC, nil, nil, n, cfg.rounds, true, sampleBuf, cfg.timing)
		})
		if err != nil {
			return res, fmt.Errorf("compute pass %d: %w", pass, err)
		}
		if pass >= 0 {
			if res.ActualTiming != "" && res.ActualTiming != stCompute.mode {
				res.ActualTiming = "mixed"
			}
			if stCompute.reason != "" && res.FallbackReason == "" {
				res.FallbackReason = stCompute.reason
			}
		}

		copyBufferToSlice(c, bufC)
		for i, got := range c {
			want := wantCompute[i%len(wantCompute)]
			tolerance := 1e-6*math.Abs(float64(want)) + 1e-5
			diff := math.Abs(float64(got) - float64(want))
			if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || diff > tolerance {
				return res, fmt.Errorf("compute pass %d verification failed at [%d]: got %g, want %g (tolerance %g)", pass, i, got, want, tolerance)
			}
			res.MaxError = math.Max(res.MaxError, diff)
		}
		if pass >= 0 {
			res.Intervals["compute"] = append(res.Intervals["compute"], stCompute.nanos)
			res.Verified += n
			recordSample(profBuilder, "compute", stCompute.nanos, pass, "compute")
		}

		trace.Logf(ctx, "metal.validation", "pass=%d verified %d elements for vector_add and compute", pass, n)
	}

	return res, nil
}

type stageTiming struct {
	nanos  int64
	mode   string
	scope  string
	reason string
}

func runStage(ctx context.Context, dev *metal.Device, pso apple.MTLComputePipelineState, b0, b1, b2 apple.MTLBuffer, n, rounds int, isCompute bool, sb *metalprof.SampleBuffer, reqMode string) (stageTiming, error) {
	var st stageTiming
	st.mode = reqMode
	name := "vector_add"
	if isCompute {
		name = "compute"
	}
	region := trace.StartRegion(ctx, name+": host launch and wait")
	defer region.End()

	cb := dev.Queue().CommandBuffer()
	var passDesc apple.MTLComputePassDescriptor
	usePassDesc := sb != nil && dev.Raw().SupportsCounterSampling(apple.MTLCounterSamplingPointAtStageBoundary)
	if usePassDesc {
		passDesc = apple.GetMTLComputePassDescriptorClass().Alloc().Init()
		att := passDesc.SampleBufferAttachments().ObjectAtIndexedSubscript(0)
		att.SetSampleBuffer(sb.Raw())
		att.SetStartOfEncoderSampleIndex(0)
		att.SetEndOfEncoderSampleIndex(1)
	}

	var enc apple.MTLComputeCommandEncoder
	if usePassDesc {
		enc = cb.ComputeCommandEncoderWithDescriptor(passDesc)
	} else {
		enc = cb.ComputeCommandEncoder()
	}

	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(b0, 0, 0)
	if !isCompute {
		enc.SetBufferWithOffsetAtIndex(b1, 0, 1)
		enc.SetBufferWithOffsetAtIndex(b2, 0, 2)
		metal.SetScalar(enc, uint32(n), 3)
	} else {
		metal.SetScalar(enc, uint32(n), 1)
		metal.SetScalar(enc, uint32(rounds), 2)
	}

	threadsPerGrid := apple.MTLSize{Width: uint(n), Height: 1, Depth: 1}
	threadsPerGroup := apple.MTLSize{Width: 256, Height: 1, Depth: 1}
	if n < 256 {
		threadsPerGroup.Width = uint(n)
	}
	enc.DispatchThreadsThreadsPerThreadgroup(threadsPerGrid, threadsPerGroup)
	enc.EndEncoding()

	cb.Commit()
	if err := metal.WaitCommandBuffer(ctx, cb); err != nil {
		return st, err
	}

	// Calculate measured GPU interval
	var nanos int64
	var counterErr error
	if usePassDesc {
		ts, err := sb.ResolveTimestamps(0, 2)
		if err != nil {
			counterErr = fmt.Errorf("resolve timestamps: %w", err)
		} else if len(ts) != 2 {
			counterErr = fmt.Errorf("expected 2 counter samples, got %d", len(ts))
		} else if ts[0] == 0 && ts[1] == 0 {
			counterErr = fmt.Errorf("counter samples remained unlatched (0, 0)")
		} else if ts[1] <= ts[0] {
			counterErr = fmt.Errorf("nonpositive counter interval: start=%d end=%d", ts[0], ts[1])
		} else {
			delta := ts[1] - ts[0]
			if delta > math.MaxInt64 {
				counterErr = fmt.Errorf("counter timestamp delta overflows int64: %d", delta)
			} else {
				nanos = int64(delta)
				st.nanos = nanos
				st.mode = "counter"
				st.scope = "stage_boundary"
				return st, nil
			}
		}
	}

	// If explicit counter timing was requested, failure to obtain valid counter samples is fatal
	if reqMode == "counter" {
		if counterErr != nil {
			return st, fmt.Errorf("explicit counter timing requested but counter sampling failed: %w", counterErr)
		}
		return st, fmt.Errorf("explicit counter timing requested but counter sampling was not available")
	}

	// Fallback to command-buffer GPU duration
	durSec := cb.GPUEndTime() - cb.GPUStartTime()
	if math.IsNaN(durSec) || math.IsInf(durSec, 0) || durSec <= 0 {
		return st, fmt.Errorf("command-buffer gpu duration unavailable or nonpositive: start=%g end=%g delta=%g",
			cb.GPUStartTime(), cb.GPUEndTime(), durSec)
	}
	nanos = int64(durSec * 1e9)
	if nanos <= 0 {
		return st, fmt.Errorf("command-buffer duration converted to nonpositive nanoseconds: %d ns", nanos)
	}

	st.nanos = nanos
	st.mode = "command-buffer"
	st.scope = "command_buffer"
	if counterErr != nil {
		st.reason = counterErr.Error()
	}
	return st, nil
}

func recordSample(b *metalprof.Builder, name string, nanos int64, pass int, phase string) {
	if b == nil {
		return
	}
	// Source line numbers 5 and 15 correspond to kernel entrypoint definitions in kernels.metal
	// and serve as source navigation anchors (not instruction-level hotness).
	line := int64(5)
	if name == "compute" {
		line = 15
	}
	fnID := b.AddFunction(name, "kernels.metal", line)
	locID := b.AddLocation(fnID, line)

	vals := metalprof.SampleValues{
		Samples:   1,
		GPUTimeNs: nanos,
	}
	labels := map[string]string{
		"kernel": name,
		"phase":  phase,
		"pass":   strconv.Itoa(pass),
	}
	b.AddSample([]uint64{locID}, vals, labels)
}

func copySliceToBuffer(dst apple.MTLBuffer, src []float32) {
	ptr := dst.Contents()
	slice := unsafeSlice[float32](ptr, len(src))
	copy(slice, src)
}

func copyBufferToSlice(dst []float32, src apple.MTLBuffer) {
	ptr := src.Contents()
	slice := unsafeSlice[float32](ptr, len(dst))
	copy(dst, slice)
}

func unsafeSlice[T any](ptr unsafePointer, length int) []T {
	return unsafeSliceCast[T](ptr, length)
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
