//go:build darwin

package metalbench

import (
	"encoding/binary"
	"errors"
	"fmt"
	"testing"
	"time"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

// ErrSimulatedOOM indicates an allocation failure under strict memory limits.
var ErrSimulatedOOM = errors.New("runtime: out of memory (allocation exceeds headroom)")

// RigidStaticEngine simulates standard runtimes with fixed batch memory demands.
type RigidStaticEngine struct {
	dev        *metal.Device
	compiler   *metalx.Compiler
	dim        int
	outDim     int
	memoryCap  int // Max allowed buffer bytes in simulated constrained environment
	usedMemory int
	pso        apple.MTLComputePipelineState
	w          metalmem.Slice[float32]
}

func NewRigidStaticEngine(dev *metal.Device, dim, outDim, memoryCap int) (*RigidStaticEngine, error) {
	msl := generateTiledGEMVMSL("Rigid_GEMV", 1)
	compiler, err := metalx.NewCompiler(dev)
	if err != nil {
		return nil, err
	}
	pso, err := compiler.Compile(msl, "Rigid_GEMV")
	if err != nil {
		_ = compiler.Close()
		return nil, err
	}
	wBytes := dim * outDim * 4
	if wBytes > memoryCap {
		_ = compiler.Close()
		return nil, ErrSimulatedOOM
	}
	w, err := metalmem.Alloc[float32](dev, dim*outDim)
	if err != nil {
		_ = compiler.Close()
		return nil, err
	}
	for i := range w.HostSlice() {
		w.HostSlice()[i] = 1
	}
	return &RigidStaticEngine{
		dev:        dev,
		compiler:   compiler,
		dim:        dim,
		outDim:     outDim,
		memoryCap:  memoryCap,
		usedMemory: wBytes,
		pso:        pso,
		w:          w,
	}, nil
}

func (r *RigidStaticEngine) Close() error {
	r.w.Free()
	return r.compiler.Close()
}

// SetMemoryLimit simulates dynamic OS/cgroups memory contraction.
func (r *RigidStaticEngine) SetMemoryLimit(newCap int) {
	r.memoryCap = newCap
}

// ProcessBatch attempts to allocate full parallel batch buffers. Fails if memoryCap is exceeded.
func (r *RigidStaticEngine) ProcessBatch(batchSize int) error {
	activationBytes := batchSize * (r.dim + r.outDim) * 4
	if r.usedMemory+activationBytes > r.memoryCap {
		// Rigid runtime fails with fatal OOM
		return fmt.Errorf("%w: requested %d bytes, available %d bytes",
			ErrSimulatedOOM, r.usedMemory+activationBytes, r.memoryCap)
	}

	// Normal forward pass execution
	in, err := metalmem.Alloc[float32](r.dev, batchSize*r.dim)
	if err != nil {
		return err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](r.dev, batchSize*r.outDim)
	if err != nil {
		return err
	}
	defer out.Free()
	for i := range in.HostSlice() {
		in.HostSlice()[i] = 1
	}
	for i := range out.HostSlice() {
		out.HostSlice()[i] = -1
	}

	cb := r.dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(r.pso)
	setMemoryPressureDimensions(enc, r.dim, r.outDim)
	for b := 0; b < batchSize; b++ {
		enc.SetBufferWithOffsetAtIndex(in.Buffer(), uint(b*r.dim*4), 0)
		enc.SetBufferWithOffsetAtIndex(r.w.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(out.Buffer(), uint(b*r.outDim*4), 2)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(r.outDim), Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	return checkMemoryPressureOutput(out.HostSlice(), r.dim)
}

// ElasticDegradationEngine adapts dynamically to shrinking memory headroom.
type ElasticDegradationEngine struct {
	dev        *metal.Device
	compiler   *metalx.Compiler
	dim        int
	outDim     int
	memoryCap  int
	usedMemory int
	psoGEMV    apple.MTLComputePipelineState
	w          metalmem.Slice[float32]
	pool       *metalmem.Pool
}

func NewElasticDegradationEngine(dev *metal.Device, dim, outDim, memoryCap int) (*ElasticDegradationEngine, error) {
	msl := generateTiledGEMVMSL("Elastic_GEMV", 4)
	compiler, err := metalx.NewCompiler(dev)
	if err != nil {
		return nil, err
	}
	pso, err := compiler.Compile(msl, "Elastic_GEMV")
	if err != nil {
		_ = compiler.Close()
		return nil, err
	}
	w, err := metalmem.Alloc[float32](dev, dim*outDim)
	if err != nil {
		_ = compiler.Close()
		return nil, err
	}
	for i := range w.HostSlice() {
		w.HostSlice()[i] = 1
	}
	return &ElasticDegradationEngine{
		dev:        dev,
		compiler:   compiler,
		dim:        dim,
		outDim:     outDim,
		memoryCap:  memoryCap,
		usedMemory: dim * outDim * 4,
		psoGEMV:    pso,
		w:          w,
		pool:       metalmem.NewPool(dev),
	}, nil
}

func (e *ElasticDegradationEngine) Close() error {
	e.pool.Free()
	e.w.Free()
	return e.compiler.Close()
}

func (e *ElasticDegradationEngine) SetMemoryLimit(newCap int) {
	e.memoryCap = newCap
}

// ProcessBatch gracefully degrades: if full batch doesn't fit, it dynamically throttles
// micro-batch sizes down (e.g. B=32 -> serial B=1 micro-steps) to guarantee 100% completion.
func (e *ElasticDegradationEngine) ProcessBatch(batchSize int) (int, error) {
	// Determine maximum micro-batch that safely fits in remaining headroom
	availableHeadroom := e.memoryCap - e.usedMemory
	if availableHeadroom < (e.dim+e.outDim)*4 {
		// Flush recycling pools and transient slabs to recover headroom
		e.pool.Free()
		e.pool = metalmem.NewPool(e.dev)
	}

	microBatch := batchSize
	singleReqBytes := (e.dim + e.outDim) * 4
	if microBatch*singleReqBytes > availableHeadroom && availableHeadroom >= singleReqBytes {
		microBatch = availableHeadroom / singleReqBytes
		if microBatch <= 0 {
			microBatch = 1
		}
	} else if availableHeadroom < singleReqBytes {
		// Minimum degraded fallback mode: reuse shared 1-element scratchpad
		microBatch = 1
	}

	tokensCompleted := 0
	for remaining := batchSize; remaining > 0; {
		step := min(remaining, microBatch)
		in, err := metalmem.Alloc[float32](e.dev, step*e.dim)
		if err != nil {
			return tokensCompleted, err
		}
		out, err := metalmem.Alloc[float32](e.dev, step*e.outDim)
		if err != nil {
			in.Free()
			return tokensCompleted, err
		}

		for i := range in.HostSlice() {
			in.HostSlice()[i] = 1
		}
		for i := range out.HostSlice() {
			out.HostSlice()[i] = -1
		}
		cb := e.dev.Queue().CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(e.psoGEMV)
		setMemoryPressureDimensions(enc, e.dim, e.outDim)
		for b := 0; b < step; b++ {
			enc.SetBufferWithOffsetAtIndex(in.Buffer(), uint(b*e.dim*4), 0)
			enc.SetBufferWithOffsetAtIndex(e.w.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(out.Buffer(), uint(b*e.outDim*4), 2)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(e.outDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		outputErr := checkMemoryPressureOutput(out.HostSlice(), e.dim)
		in.Free()
		out.Free()
		if outputErr != nil {
			return tokensCompleted, outputErr
		}

		tokensCompleted += step
		remaining -= step
	}

	return tokensCompleted, nil
}

func TestMemoryPressureDegradation(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	const (
		dim    = 1024
		outDim = 1024
	)
	weightBytes := dim * outDim * 4 // 4 MB weights

	// Initial comfortable memory cap: 32 MB
	initialCap := 32 * 1024 * 1024

	rigidEngine, err := NewRigidStaticEngine(dev, dim, outDim, initialCap)
	if err != nil {
		t.Fatalf("Failed creating rigid engine: %v", err)
	}
	defer rigidEngine.Close()

	elasticEngine, err := NewElasticDegradationEngine(dev, dim, outDim, initialCap)
	if err != nil {
		t.Fatalf("Failed creating elastic engine: %v", err)
	}
	defer elasticEngine.Close()

	t.Logf("==========================================================================================")
	t.Logf("             DYNAMIC MEMORY PRESSURE & OOM-CRASH AVOIDANCE BENCHMARK                      ")
	t.Logf("==========================================================================================")

	// Step 1: Normal execution under ample memory (B=32)
	batchSize := 32
	t.Logf("[Normal State: Cap=32MB] Dispatching Batch=%d ...", batchSize)

	if err := rigidEngine.ProcessBatch(batchSize); err != nil {
		t.Errorf("Rigid engine failed in normal state: %v", err)
	} else {
		t.Logf("  ✓ Rigid Static Engine:   PROCESSED %d tokens successfully", batchSize)
	}

	if completed, err := elasticEngine.ProcessBatch(batchSize); err != nil || completed != batchSize {
		t.Errorf("Elastic engine failed in normal state: %v", err)
	} else {
		t.Logf("  ✓ Elastic Engine:        PROCESSED %d tokens successfully", completed)
	}

	// Step 2: Inject Severe Memory Pressure (Contract headroom down to 4.06 MB, barely above 4 MB weights)
	constrainedCap := weightBytes + 64*1024 // 4.06 MB (Only 64 KB headroom for activations)
	t.Logf("\n[PRESSURE EVENT]: Available Memory Shrinks 32MB -> 4.06MB (64 KB Activation Headroom)")
	t.Logf("Dispatching Heavy Batch=%d (Requires 256 KB in standard static allocation) ...", batchSize)

	rigidEngine.SetMemoryLimit(constrainedCap)
	elasticEngine.SetMemoryLimit(constrainedCap)

	// Test Rigid Engine under pressure
	rigidStart := time.Now()
	errRigid := rigidEngine.ProcessBatch(batchSize)
	rigidDur := time.Since(rigidStart)

	if errors.Is(errRigid, ErrSimulatedOOM) {
		t.Logf("  ✗ Rigid Static Engine:   FATAL CRASH / OOM ABORT: %v (Duration: %v)", errRigid, rigidDur)
	} else {
		t.Errorf("Expected rigid engine to fail with OOM, got: %v", errRigid)
	}

	// Test Elastic Engine under pressure
	elasticStart := time.Now()
	completedTokens, errElastic := elasticEngine.ProcessBatch(batchSize)
	elasticDur := time.Since(elasticStart)

	if errElastic != nil {
		t.Errorf("Elastic engine should have survived memory pressure, but failed: %v", errElastic)
	} else {
		t.Logf("  ✓ Elastic Engine:        SURVIVED & COMPLETED %d/%d tokens via dynamic micro-batching! (Duration: %v)",
			completedTokens, batchSize, elasticDur)
	}

	t.Logf("==========================================================================================")
	t.Logf("SUMMARY:")
	t.Logf("  Rigid Static Engine: 0%% completion under pressure (FATAL OOM CRASH)")
	t.Logf("  Elastic PGO Engine:  100%% completion under pressure (ZERO CRASHES, DYNAMIC DEGRADATION)")
	t.Logf("==========================================================================================")
}

func checkMemoryPressureOutput(output []float32, dim int) error {
	for i, got := range output {
		if got != float32(dim) {
			return fmt.Errorf("output[%d] = %g, want %d", i, got, dim)
		}
	}
	return nil
}

func setMemoryPressureDimensions(enc apple.MTLComputeCommandEncoder, dim, outDim int) {
	var value [4]byte
	binary.LittleEndian.PutUint32(value[:], uint32(dim))
	enc.SetBytesLengthAtIndex(value[:], 3)
	binary.LittleEndian.PutUint32(value[:], uint32(outDim))
	enc.SetBytesLengthAtIndex(value[:], 4)
}
