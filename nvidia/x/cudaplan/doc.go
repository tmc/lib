// Package cudaplan provides executable plans, resource ownership, and completion tracking
// for CUDA execution graphs and compiled kernels.
//
// cudaplan compiles validated IR graphs (from [cudair]) into executable plans with explicit
// inputs, outputs, memory bindings, workspace management, and completion ownership. It supports
// both ordinary kernel dispatch via [cudalaunch] and CUDA Graph replay via [cudagraph].
//
// # Basic Usage
//
// To compile and execute a computation graph:
//
//	plan, err := cudaplan.Compile(snap, cudaplan.WithDevice(dev))
//	if err != nil {
//	    return err
//	}
//	defer plan.Close()
//
//	comp, err := plan.Execute(stream, bindings)
//	if err != nil {
//	    return err
//	}
//	if err := comp.Wait(ctx); err != nil {
//	    return err
//	}
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaplan
