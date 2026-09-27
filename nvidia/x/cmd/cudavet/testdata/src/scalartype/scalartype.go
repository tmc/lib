package scalartype

import (
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/cusparse"
	"github.com/tmc/lib/nvidia/cutensor"
	"github.com/tmc/lib/nvidia/driver"
)

func contract(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc32F()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dB, []int32{'k', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Close()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Close()
	beta := cutensor.Float64Scalar(0)
	return h.Contract(plan, cutensor.Float64Scalar(1), a, b, beta, c, c, 0, 0, s) // want `cutensor.Float64Scalar\(1\) is a float64 Scalar, but ComputeDesc32F takes float32 or complex64` `beta is a float64 Scalar, but ComputeDesc32F takes float32 or complex64`
}

func contract32(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc32F()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dB, []int32{'k', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Close()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Close()
	return h.Contract(plan, cutensor.Float32Scalar(1), a, b, cutensor.Float32Scalar(0), c, c, 0, 0, s)
}

func contract8XInt8(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc8XInt8()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dB, []int32{'k', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Close()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Close()
	alpha, beta := cutensor.Float32Scalar(1), cutensor.Float32Scalar(0)
	return h.Contract(plan, alpha, a, b, beta, c, c, 0, 0, s) // want `alpha is a float32 Scalar, but ComputeDesc8XInt8 takes float64 or complex128` `beta is a float32 Scalar, but ComputeDesc8XInt8 takes float64 or complex128`
}

func contract8XInt8Clean(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc8XInt8()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dB, []int32{'k', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Close()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Close()
	alpha, beta := cutensor.Float64Scalar(1), cutensor.Float64Scalar(0)
	return h.Contract(plan, alpha, a, b, beta, c, c, 0, 0, s)
}

func contract9X16BF(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc9X16BF()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dB, []int32{'k', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Close()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Close()
	alpha, beta := cutensor.Float64Scalar(1), cutensor.Float64Scalar(0)
	return h.Contract(plan, alpha, a, b, beta, c, c, 0, 0, s) // want `alpha is a float64 Scalar, but ComputeDesc9X16BF takes float32 or complex64` `beta is a float64 Scalar, but ComputeDesc9X16BF takes float32 or complex64`
}

func contract9X16BFClean(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc9X16BF()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dB, []int32{'k', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Close()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Close()
	alpha, beta := cutensor.Float32Scalar(1), cutensor.Float32Scalar(0)
	return h.Contract(plan, alpha, a, b, beta, c, c, 0, 0, s)
}

func contract4X16F(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc4X16F()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dB, []int32{'k', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Close()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Close()
	alpha, beta := cutensor.Float64Scalar(1), cutensor.Float64Scalar(0)
	return h.Contract(plan, alpha, a, b, beta, c, c, 0, 0, s) // want `alpha is a float64 Scalar, but ComputeDesc4X16F takes float32 or complex64` `beta is a float64 Scalar, but ComputeDesc4X16F takes float32 or complex64`
}

func contract4X16FClean(h cutensor.Handle, dA, dB, dC cutensor.TensorDescriptor, a, b, c driver.DevicePtr, s driver.Stream) error {
	compute, err := cutensor.ComputeDesc4X16F()
	if err != nil {
		return err
	}
	op, err := h.CreateContraction(dA, []int32{'i', 'k'}, cutensor.OpIdentity, dB, []int32{'k', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, cutensor.OpIdentity, dC, []int32{'i', 'j'}, compute)
	if err != nil {
		return err
	}
	defer op.Close()
	plan, err := h.CreatePlan(op, 0, 0)
	if err != nil {
		return err
	}
	defer plan.Close()
	alpha, beta := cutensor.Float32Scalar(1), cutensor.Float32Scalar(0)
	return h.Contract(plan, alpha, a, b, beta, c, c, 0, 0, s)
}

func gemm(h cublas.Handle, a, b, c driver.DevicePtr, n int32) error {
	alpha, beta := cublas.Float32Scalar(1), cublas.Float32Scalar(0)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, alpha, a, cublas.R64F, n, b, cublas.R64F, n, beta, c, cublas.R64F, n, cublas.Compute64F, cublas.GemmDefault) // want `alpha is a float32 Scalar, but Compute64F takes float64 or complex128` `beta is a float32 Scalar, but Compute64F takes float64 or complex128`
}

func gemm32(h cublas.Handle, a, b, c driver.DevicePtr, n int32) error {
	alpha, beta := cublas.Float32Scalar(1), cublas.Float32Scalar(0)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, alpha, a, cublas.R32F, n, b, cublas.R32F, n, beta, c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

// gemmComplex is a complex GEMM, which reads complex64 scalars in Compute32F.
func gemmComplex(h cublas.Handle, a, b, c driver.DevicePtr, n int32) error {
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.Complex64Scalar(1), a, cublas.R32F, n, b, cublas.R32F, n, cublas.Complex64Scalar(0), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

// gemmDevice passes scalars whose kind is not known here.
func gemmDevice(h cublas.Handle, a, b, c, alpha driver.DevicePtr, n int32) error {
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.DeviceScalar(alpha), a, cublas.R64F, n, b, cublas.R64F, n, cublas.DeviceScalar(alpha), c, cublas.R64F, n, cublas.Compute64F, cublas.GemmDefault)
}

func matmul(h cublaslt.Handle, a, b, c driver.DevicePtr, l cublaslt.MatrixLayout, s driver.Stream) error {
	desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
	if err != nil {
		return err
	}
	return h.Matmul(desc, cublaslt.Float64Scalar(1), a, l, b, l, cublaslt.Float32Scalar(0), c, l, c, l, nil, 0, 0, s) // want `cublaslt.Float64Scalar\(1\) is a float64 Scalar, but R32F takes float32`
}

func matmul32(h cublaslt.Handle, a, b, c driver.DevicePtr, l cublaslt.MatrixLayout, s driver.Stream) error {
	desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
	if err != nil {
		return err
	}
	return h.Matmul(desc, cublaslt.Float32Scalar(1), a, l, b, l, cublaslt.Float32Scalar(0), c, l, c, l, nil, 0, 0, s)
}

func spmv(h cusparse.Handle, m cusparse.ConstSpMatDescr, x cusparse.ConstDnVecDescr, y cusparse.DnVecDescr) error {
	return h.SpMV(cusparse.OperationNonTranspose, cusparse.Float32Scalar(1), m, x, cusparse.Float32Scalar(0), y, cusparse.R64F, cusparse.SpmvAlgDefault, 0) // want `cusparse.Float32Scalar\(1\) is a float32 Scalar, but R64F takes float64` `cusparse.Float32Scalar\(0\) is a float32 Scalar, but R64F takes float64`
}

func spmv64(h cusparse.Handle, m cusparse.ConstSpMatDescr, x cusparse.ConstDnVecDescr, y cusparse.DnVecDescr) error {
	return h.SpMV(cusparse.OperationNonTranspose, cusparse.Float64Scalar(1), m, x, cusparse.Float64Scalar(0), y, cusparse.R64F, cusparse.SpmvAlgDefault, 0)
}
