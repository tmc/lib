package modecount

import "github.com/tmc/lib/nvidia/cutensor"

func contraction(h cutensor.Handle, compute cutensor.ComputeDescriptor) error {
	dA, err := h.CreateTensorDescriptor([]int64{4, 8}, nil, cutensor.R32F, 128)
	if err != nil {
		return err
	}
	defer dA.Close()
	dB, err := h.CreateTensorDescriptor([]int64{8}, nil, cutensor.R32F, 128)
	if err != nil {
		return err
	}
	defer dB.Close()
	modeB := []int32{'k', 'j'}
	op, err := h.CreateContraction(dA, []int32{'i'}, cutensor.OpIdentity, dB, modeB, cutensor.OpIdentity, dB, []int32{'k'}, cutensor.OpIdentity, dB, []int32{'k'}, compute) // want `\[\]int32\{'i'\} has 1 modes, but dA was created with 2 extents` `modeB has 2 modes, but dB was created with 1 extents`
	if err != nil {
		return err
	}
	return op.Close()
}

func matching(h cutensor.Handle, compute cutensor.ComputeDescriptor, modeC []int32) error {
	dA, err := h.CreateTensorDescriptor([]int64{4, 8}, nil, cutensor.R32F, 128)
	if err != nil {
		return err
	}
	defer dA.Close()
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dA, []int32{'k', 'j'}, cutensor.OpIdentity, dA, modeC, cutensor.OpIdentity, dA, modeC, compute)
	if err != nil {
		return err
	}
	return op.Close()
}
