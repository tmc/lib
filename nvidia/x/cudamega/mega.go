package cudamega

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Task represents a unit of work submitted to the persistent mega-kernel queue.
type Task struct {
	OpCode      uint32
	NumElements uint32
	Arg0        uint64
	Arg1        uint64
	Arg2        uint64
	Arg3        uint64
}

// MegaKernelTemplate is the embedded CUDA C++ source for persistent grid dispatching.
const MegaKernelTemplate = `
#pragma once

#ifndef CUDAMEGA_CUH
#define CUDAMEGA_CUH

struct Task {
    unsigned int op_code;
    unsigned int num_elements;
    unsigned long long arg0;
    unsigned long long arg1;
    unsigned long long arg2;
    unsigned long long arg3;
};

extern "C" __global__ void PersistentMegaKernel(
    Task* __restrict__ queue,
    unsigned int* __restrict__ head,
    unsigned int* __restrict__ tail,
    volatile unsigned int* __restrict__ is_running,
    unsigned int queue_capacity
) {
    __shared__ Task current_task;
    __shared__ int task_valid;

    while (*is_running) {
        if (threadIdx.x == 0) {
            unsigned int h = *head;
            unsigned int t = *tail;
            if (h < t) {
                unsigned int my_task_idx = atomicAdd(head, 1);
                if (my_task_idx < t) {
                    current_task = queue[my_task_idx % queue_capacity];
                    task_valid = 1;
                } else {
                    task_valid = 0;
                }
            } else {
                task_valid = 0;
            }
        }
        __syncthreads();

        if (task_valid) {
            // Execute task in parallel across block threads
            unsigned int n = current_task.num_elements;
            for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
                // User custom operation dispatcher based on current_task.op_code
            }
        }
        __syncthreads();
    }
}

#endif // CUDAMEGA_CUH
`

// Dispatcher coordinates task queue staging and persistent grid execution.
type Dispatcher struct {
	dev        *cuda.Device
	capacity   int
	queueSlice cudamem.Slice[Task]
	headSlice  cudamem.Slice[uint32]
	tailSlice  cudamem.Slice[uint32]
	runSlice   cudamem.Slice[uint32]
}

// NewDispatcher creates a persistent megakernel task dispatcher.
func NewDispatcher(dev *cuda.Device, maxTasks int) (*Dispatcher, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudamega: nil device")
	}

	queue, err := cudamem.Alloc[Task](dev, maxTasks)
	if err != nil {
		return nil, fmt.Errorf("cudamega: alloc queue: %w", err)
	}

	head, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		_ = queue.Free()
		return nil, fmt.Errorf("cudamega: alloc head: %w", err)
	}

	tail, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		_ = queue.Free()
		_ = head.Free()
		return nil, fmt.Errorf("cudamega: alloc tail: %w", err)
	}

	running, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		_ = queue.Free()
		_ = head.Free()
		_ = tail.Free()
		return nil, fmt.Errorf("cudamega: alloc running: %w", err)
	}

	return &Dispatcher{
		dev:        dev,
		capacity:   maxTasks,
		queueSlice: queue,
		headSlice:  head,
		tailSlice:  tail,
		runSlice:   running,
	}, nil
}

// Capacity returns the maximum number of queued tasks in the ring buffer.
func (d *Dispatcher) Capacity() int {
	return d.capacity
}

// Close deallocates queue memory on the GPU.
func (d *Dispatcher) Close() error {
	_ = d.queueSlice.Free()
	_ = d.headSlice.Free()
	_ = d.tailSlice.Free()
	_ = d.runSlice.Free()
	return nil
}
