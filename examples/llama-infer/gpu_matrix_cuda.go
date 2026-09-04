//go:build !darwin

package main

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const cudaMatrixWorkspaceBytes = 32 << 20

// cudaMatrixProjection owns immutable cuBLASLt plans for one [B,K] by [N,K]
// row-major projection at every supported batch width.
type cudaMatrixProjection struct {
	handle    cublaslt.Handle
	plans     []cudaMatrixPlan
	workspace cudamem.Slice[byte]
}

type cudaMatrixPlan struct {
	desc      cublaslt.MatmulDesc
	input     cublaslt.MatrixLayout
	weight    cublaslt.MatrixLayout
	output    cublaslt.MatrixLayout
	algorithm cublaslt.MatmulAlgo
	workspace uint
}

func newCUDAMatrixProjection(dev *cuda.Device, arena *cudamem.Arena, input, output, capacity int) (*cudaMatrixProjection, error) {
	if dev == nil || arena == nil || input <= 0 || output <= 0 || capacity <= 0 {
		return nil, errors.New("cuda matrix projection: invalid configuration")
	}
	projection := &cudaMatrixProjection{
		plans:     make([]cudaMatrixPlan, capacity+1),
		workspace: cudamem.AllocIn[byte](arena, "batch-matrix-workspace", cudaMatrixWorkspaceBytes),
	}
	if err := arena.Err(); err != nil {
		return nil, fmt.Errorf("cuda matrix projection: allocate workspace: %w", err)
	}
	if err := dev.WithContext(func() error {
		var err error
		projection.handle, err = cublaslt.Create()
		if err != nil {
			return fmt.Errorf("create cuBLASLt handle: %w", err)
		}
		for batch := 1; batch <= capacity; batch++ {
			if err := projection.initPlan(batch, input, output); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		_ = projection.close()
		return nil, fmt.Errorf("cuda matrix projection: %w", err)
	}
	return projection, nil
}

func (p *cudaMatrixProjection) initPlan(batch, input, output int) error {
	plan := &p.plans[batch]
	var err error
	plan.desc, err = cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
	if err != nil {
		return fmt.Errorf("batch %d descriptor: %w", batch, err)
	}
	transWeight := cublaslt.CublasOpT
	if err := plan.desc.SetAttribute(cublaslt.MatmulDescTransb, uintptr(unsafe.Pointer(&transWeight)), uint(unsafe.Sizeof(transWeight))); err != nil {
		return fmt.Errorf("batch %d transpose weight: %w", batch, err)
	}
	plan.input, err = newCUDARowMajorLayout(uint64(batch), uint64(input), int64(input))
	if err != nil {
		return fmt.Errorf("batch %d input layout: %w", batch, err)
	}
	plan.weight, err = newCUDARowMajorLayout(uint64(output), uint64(input), int64(input))
	if err != nil {
		return fmt.Errorf("batch %d weight layout: %w", batch, err)
	}
	plan.output, err = newCUDARowMajorLayout(uint64(batch), uint64(output), int64(output))
	if err != nil {
		return fmt.Errorf("batch %d output layout: %w", batch, err)
	}
	preference, err := cublaslt.NewMatmulPreference()
	if err != nil {
		return fmt.Errorf("batch %d preference: %w", batch, err)
	}
	defer preference.Destroy()
	workspace := uint64(cudaMatrixWorkspaceBytes)
	if err := preference.SetAttribute(cublaslt.MatmulPrefMaxWorkspaceBytes, uintptr(unsafe.Pointer(&workspace)), uint(unsafe.Sizeof(workspace))); err != nil {
		return fmt.Errorf("batch %d workspace preference: %w", batch, err)
	}
	var result cublaslt.MatmulHeuristicResult
	count, err := p.handle.MatmulAlgoGetHeuristic(plan.desc, plan.input, plan.weight, plan.output, plan.output, preference, 1, &result)
	if err != nil {
		return fmt.Errorf("batch %d select algorithm: %w", batch, err)
	}
	if count != 1 || result.State != cublaslt.Success || result.WorkspaceSize > cudaMatrixWorkspaceBytes {
		return fmt.Errorf("batch %d select algorithm: no usable result", batch)
	}
	plan.algorithm = result.Algo
	plan.workspace = result.WorkspaceSize
	return nil
}

func newCUDARowMajorLayout(rows, columns uint64, leadingDimension int64) (cublaslt.MatrixLayout, error) {
	layout, err := cublaslt.NewMatrixLayout(cublaslt.R32F, rows, columns, leadingDimension)
	if err != nil {
		return 0, err
	}
	order := cublaslt.OrderRow
	if err := layout.SetAttribute(cublaslt.MatrixLayoutOrder, uintptr(unsafe.Pointer(&order)), uint(unsafe.Sizeof(order))); err != nil {
		_ = layout.Destroy()
		return 0, err
	}
	return layout, nil
}

func (p *cudaMatrixProjection) run(stream *cuda.Stream, input, weight, output cudamem.Slice[float32], batch int) error {
	if p == nil || stream == nil || batch <= 0 || batch >= len(p.plans) {
		return errors.New("cuda matrix projection: invalid launch")
	}
	plan := &p.plans[batch]
	alpha, beta := float32(1), float32(0)
	err := p.handle.Matmul(plan.desc,
		uintptr(unsafe.Pointer(&alpha)), uintptr(input.Ptr()), plan.input,
		uintptr(weight.Ptr()), plan.weight,
		uintptr(unsafe.Pointer(&beta)), uintptr(output.Ptr()), plan.output,
		uintptr(output.Ptr()), plan.output, &plan.algorithm,
		uintptr(p.workspace.Ptr()), plan.workspace, cublaslt.Stream(stream.Raw()))
	if err != nil {
		return fmt.Errorf("cuda matrix projection: batch %d: %w", batch, err)
	}
	return nil
}

func (p *cudaMatrixProjection) close() error {
	if p == nil {
		return nil
	}
	var err error
	for i := range p.plans {
		plan := &p.plans[i]
		if plan.output != 0 {
			err = errors.Join(err, plan.output.Destroy())
		}
		if plan.weight != 0 {
			err = errors.Join(err, plan.weight.Destroy())
		}
		if plan.input != 0 {
			err = errors.Join(err, plan.input.Destroy())
		}
		if plan.desc != 0 {
			err = errors.Join(err, plan.desc.Destroy())
		}
	}
	if p.handle != 0 {
		err = errors.Join(err, p.handle.Destroy())
	}
	p.handle = 0
	return err
}
