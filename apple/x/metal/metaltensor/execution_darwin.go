//go:build darwin

package metaltensor

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalir"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

// TensorExecutionResult owns the intermediate buffers allocated while
// executing a plan. Output is borrowed from the result and remains valid until
// Close. External input buffers remain caller-owned.
type TensorExecutionResult struct {
	Output apple.MTLBuffer

	mu     sync.Mutex
	closed bool
	owned  []apple.MTLBuffer
}

// TensorPlanExecutor reuses compiled pipelines for repeated executions on one
// Metal device. Close waits for an active execution and releases the pipelines.
type TensorPlanExecutor struct {
	mu       sync.Mutex
	dev      *metal.Device
	compiler *metalx.Compiler
	closed   bool
}

// NewTensorPlanExecutor creates an executor bound to dev.
func NewTensorPlanExecutor(dev *metal.Device) (*TensorPlanExecutor, error) {
	compiler, err := metalx.NewCompiler(dev)
	if err != nil {
		return nil, fmt.Errorf("tensor execution: create Metal compiler: %w", err)
	}
	return &TensorPlanExecutor{dev: dev, compiler: compiler}, nil
}

// Close releases cached pipelines. It is safe to call more than once.
func (e *TensorPlanExecutor) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	return e.compiler.Close()
}

// Execute validates external buffers, allocates intermediate buffers, and
// dispatches plan using pipelines cached by this executor.
func (e *TensorPlanExecutor) Execute(plan *TensorExecutionPlan, inputs map[TensorBufferRef]apple.MTLBuffer) (*TensorExecutionResult, error) {
	if e == nil {
		return nil, fmt.Errorf("tensor execution: nil executor")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("tensor execution: executor is closed")
	}
	return executeTensorExecutionPlan(e.dev, e.compiler, plan, inputs)
}

// Close releases intermediate buffers. It is safe to call more than once.
func (r *TensorExecutionResult) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	for i := len(r.owned) - 1; i >= 0; i-- {
		buffer := r.owned[i]
		if buffer != nil && buffer.GetID() != 0 {
			objectivec.ObjectFromID(buffer.GetID()).Release()
		}
	}
	r.owned = nil
	r.Output = nil
	return nil
}

// ExecuteTensorExecutionPlan validates external buffer sizes, allocates the
// intermediate buffers, compiles each stage, and dispatches the plan in one
// Metal command buffer. The result owns the intermediate buffers; input buffers
// remain owned by the caller.
func ExecuteTensorExecutionPlan(dev *metal.Device, plan *TensorExecutionPlan, inputs map[TensorBufferRef]apple.MTLBuffer) (_ *TensorExecutionResult, err error) {
	executor, err := NewTensorPlanExecutor(dev)
	if err != nil {
		return nil, err
	}
	defer executor.Close()
	return executor.Execute(plan, inputs)
}

func executeTensorExecutionPlan(dev *metal.Device, compiler *metalx.Compiler, plan *TensorExecutionPlan, inputs map[TensorBufferRef]apple.MTLBuffer) (_ *TensorExecutionResult, err error) {
	if dev == nil || dev.Raw() == nil || dev.Raw().GetID() == 0 {
		return nil, fmt.Errorf("tensor execution: nil or uninitialized Metal device")
	}
	if plan == nil {
		return nil, fmt.Errorf("tensor execution: nil plan")
	}
	layouts, err := validateTensorExecutionPlan(plan)
	if err != nil {
		return nil, err
	}
	buffers := make(map[TensorBufferRef]apple.MTLBuffer, len(layouts)+len(plan.Stages))
	for _, input := range plan.InputLayouts {
		buffer := inputs[input.Buffer]
		if buffer == nil || buffer.GetID() == 0 {
			return nil, fmt.Errorf("tensor execution: missing external buffer %q", input.Buffer.Name)
		}
		want := layouts[input.Buffer]
		if uint64(buffer.Length()) < want {
			return nil, fmt.Errorf("tensor execution: external buffer %q has %d bytes, need %d", input.Buffer.Name, buffer.Length(), want)
		}
		buffers[input.Buffer] = buffer
	}
	owned := make([]apple.MTLBuffer, 0, len(plan.Stages))
	transferred := false
	defer func() {
		if !transferred {
			releaseTensorBuffers(owned)
		}
	}()
	for _, stage := range plan.Stages {
		if _, exists := buffers[stage.Output]; exists {
			return nil, fmt.Errorf("tensor execution: duplicate buffer reference %q", stage.Output.Name)
		}
		bytes, sizeErr := tensorLayoutBytes(stage.Shape, stage.DataType)
		if sizeErr != nil {
			return nil, fmt.Errorf("tensor execution: stage %s output size: %w", stage.KernelName, sizeErr)
		}
		if bytes > uint64(math.MaxInt) {
			return nil, fmt.Errorf("tensor execution: stage %s output is too large to allocate", stage.KernelName)
		}
		buffer, allocErr := dev.AllocBuffer(int(bytes))
		if allocErr != nil {
			return nil, fmt.Errorf("tensor execution: allocate %s output: %w", stage.KernelName, allocErr)
		}
		owned = append(owned, buffer)
		buffers[stage.Output] = buffer
		layouts[stage.Output] = bytes
	}
	output, ok := buffers[plan.Output]
	if !ok {
		return nil, fmt.Errorf("tensor execution: plan output %q has no buffer", plan.Output.Name)
	}
	if len(plan.Stages) == 0 {
		transferred = true
		return &TensorExecutionResult{Output: output}, nil
	}
	pipelines := make([]apple.MTLComputePipelineState, len(plan.Stages))
	for i, stage := range plan.Stages {
		if stage.Grid.Width == 0 {
			continue
		}
		pipeline, compileErr := compiler.Compile(stage.Source, stage.KernelName)
		if compileErr != nil {
			return nil, fmt.Errorf("tensor execution: compile stage %s: %w", stage.KernelName, compileErr)
		}
		pipelines[i] = pipeline
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	commandBuffer := dev.Queue().CommandBuffer()
	if commandBuffer == nil || commandBuffer.GetID() == 0 {
		return nil, fmt.Errorf("tensor execution: create command buffer failed")
	}
	encoder := commandBuffer.ComputeCommandEncoder()
	if encoder == nil || encoder.GetID() == 0 {
		return nil, fmt.Errorf("tensor execution: create compute encoder failed")
	}
	for i, stage := range plan.Stages {
		if stage.Grid.Width == 0 {
			continue
		}
		encoder.SetComputePipelineState(pipelines[i])
		for slot, ref := range stage.Inputs {
			buffer, exists := buffers[ref]
			if !exists {
				return nil, fmt.Errorf("tensor execution: stage %s input %q has no buffer", stage.KernelName, ref.Name)
			}
			encoder.SetBufferWithOffsetAtIndex(buffer, 0, uint(slot))
		}
		encoder.SetBufferWithOffsetAtIndex(buffers[stage.Output], 0, uint(len(stage.Inputs)))
		encoder.DispatchThreadsThreadsPerThreadgroup(apple.MTLSize{
			Width: uint(stage.Grid.Width), Height: uint(stage.Grid.Height), Depth: uint(stage.Grid.Depth),
		}, apple.MTLSize{
			Width: uint(stage.Threadgroup.Width), Height: uint(stage.Threadgroup.Height), Depth: uint(stage.Threadgroup.Depth),
		})
	}
	encoder.EndEncoding()
	commandBuffer.Commit()
	if err := metal.WaitCommandBuffer(context.Background(), commandBuffer); err != nil {
		return nil, fmt.Errorf("tensor execution: Metal command buffer: %w", err)
	}
	transferred = true
	return &TensorExecutionResult{Output: output, owned: owned}, nil
}

func validateTensorExecutionPlan(plan *TensorExecutionPlan) (map[TensorBufferRef]uint64, error) {
	if plan.Output.Name == "" {
		return nil, fmt.Errorf("tensor execution: plan has no output reference")
	}
	layouts := make(map[TensorBufferRef]uint64, len(plan.InputLayouts)+len(plan.Stages))
	inputs := make(map[TensorBufferRef]bool, len(plan.Inputs))
	for _, ref := range plan.Inputs {
		if ref.Kind != ExternalTensorBuffer || ref.Name == "" || inputs[ref] {
			return nil, fmt.Errorf("tensor execution: invalid or duplicate external input reference %q", ref.Name)
		}
		inputs[ref] = true
	}
	for _, layout := range plan.InputLayouts {
		if !inputs[layout.Buffer] {
			return nil, fmt.Errorf("tensor execution: layout for unlisted input %q", layout.Buffer.Name)
		}
		bytes, err := tensorLayoutBytes(layout.Shape, layout.DataType)
		if err != nil {
			return nil, fmt.Errorf("tensor execution: input %q layout: %w", layout.Buffer.Name, err)
		}
		if err := validateTensorMSLShape(layout.Shape); err != nil {
			return nil, fmt.Errorf("tensor execution: input %q layout: %w", layout.Buffer.Name, err)
		}
		if _, exists := layouts[layout.Buffer]; exists {
			return nil, fmt.Errorf("tensor execution: duplicate input layout %q", layout.Buffer.Name)
		}
		layouts[layout.Buffer] = bytes
	}
	if len(layouts) != len(inputs) {
		return nil, fmt.Errorf("tensor execution: got %d input layouts for %d inputs", len(layouts), len(inputs))
	}
	produced := make(map[TensorBufferRef]bool, len(plan.Stages))
	for i, stage := range plan.Stages {
		if stage.KernelName == "" || stage.Source == "" || stage.Output.Kind != IntermediateTensorBuffer || stage.Output.Name == "" {
			return nil, fmt.Errorf("tensor execution: stage %d has invalid kernel or output", i)
		}
		count, err := tensorElementCount(stage.Shape)
		if err != nil {
			return nil, fmt.Errorf("tensor execution: stage %s output shape: %w", stage.KernelName, err)
		}
		wantGrid, wantThreadgroup := tensorDispatch(stage.Shape, stage.Op)
		if stage.Grid != wantGrid || stage.Threadgroup != wantThreadgroup {
			return nil, fmt.Errorf("tensor execution: stage %s dispatch does not match its output shape", stage.KernelName)
		}
		if count != 0 && (!validTensorDispatch(stage.Grid) || !validTensorDispatch(stage.Threadgroup)) {
			return nil, fmt.Errorf("tensor execution: stage %s has invalid dispatch sizes", stage.KernelName)
		}
		for _, ref := range stage.Inputs {
			if ref.Kind == IntermediateTensorBuffer && !produced[ref] {
				return nil, fmt.Errorf("tensor execution: stage %s reads intermediate %q before it is produced", stage.KernelName, ref.Name)
			}
			if ref.Kind != ExternalTensorBuffer && ref.Kind != IntermediateTensorBuffer {
				return nil, fmt.Errorf("tensor execution: stage %s has invalid input reference %q", stage.KernelName, ref.Name)
			}
			if _, ok := layouts[ref]; !ok && ref.Kind == ExternalTensorBuffer {
				return nil, fmt.Errorf("tensor execution: stage %s input %q has no layout", stage.KernelName, ref.Name)
			}
		}
		bytes, err := tensorLayoutBytes(stage.Shape, stage.DataType)
		if err != nil {
			return nil, fmt.Errorf("tensor execution: stage %s output layout: %w", stage.KernelName, err)
		}
		if err := validateTensorMSLShape(stage.Shape); err != nil {
			return nil, fmt.Errorf("tensor execution: stage %s output layout: %w", stage.KernelName, err)
		}
		if _, exists := layouts[stage.Output]; exists || produced[stage.Output] {
			return nil, fmt.Errorf("tensor execution: duplicate output reference %q", stage.Output.Name)
		}
		layouts[stage.Output] = bytes
		produced[stage.Output] = true
	}
	if !inputs[plan.Output] && !produced[plan.Output] {
		return nil, fmt.Errorf("tensor execution: output %q is neither an input nor a stage output", plan.Output.Name)
	}
	return layouts, nil
}

func tensorLayoutBytes(shape []int, dtype metalir.DataType) (uint64, error) {
	if dtype.String() == "unknown" {
		return 0, fmt.Errorf("unknown tensor data type")
	}
	count, err := tensorElementCount(shape)
	if err != nil {
		return 0, err
	}
	unit := uint64(dtype.SizeBytes())
	if unit == 0 || count > math.MaxUint64/unit {
		return 0, fmt.Errorf("tensor byte size overflows")
	}
	return count * unit, nil
}

func validTensorDispatch(size TensorDispatchSize) bool {
	maxUint := uint64(^uint(0))
	return size.Width > 0 && size.Height > 0 && size.Depth > 0 &&
		size.Width <= maxUint && size.Height <= maxUint && size.Depth <= maxUint
}

func releaseTensorBuffers(buffers []apple.MTLBuffer) {
	for i := len(buffers) - 1; i >= 0; i-- {
		buffer := buffers[i]
		if buffer != nil && buffer.GetID() != 0 {
			objectivec.ObjectFromID(buffer.GetID()).Release()
		}
	}
}
