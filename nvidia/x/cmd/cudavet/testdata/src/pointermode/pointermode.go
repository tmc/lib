package pointermode

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/cusparse"
	"github.com/tmc/lib/nvidia/driver"
)

// deviceOnHost passes a device address to a handle still in host mode.
func deviceOnHost(a, b, c, alpha driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	beta := cublas.DeviceScalar(alpha)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.DeviceScalar(alpha), a, cublas.R32F, n, b, cublas.R32F, n, beta, c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault) // want `cublas.DeviceScalar\(alpha\) is a device Scalar, but h is in host pointer mode` `beta is a device Scalar, but h is in host pointer mode`
}

// hostOnDevice passes host values after switching to device mode.
func hostOnDevice(a, b, c driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	if err := h.SetPointerMode(cublas.PointerModeDevice); err != nil {
		return err
	}
	alpha := cublas.Float32Scalar(1)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, alpha, a, cublas.R32F, n, b, cublas.R32F, n, cublas.Float32Scalar(0), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault) // want `alpha is a host float32 Scalar, but h is in device pointer mode` `cublas.Float32Scalar\(0\) is a host float32 Scalar, but h is in device pointer mode`
}

// switchBack returns to host mode before passing host values.
func switchBack(a, b, c, dAlpha, dBeta driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	h.SetPointerMode(cublas.PointerModeDevice)
	if err := h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.DeviceScalar(dAlpha), a, cublas.R32F, n, b, cublas.R32F, n, cublas.DeviceScalar(dBeta), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault); err != nil {
		return err
	}
	h.SetPointerMode(cublas.PointerModeHost)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.Float32Scalar(1), a, cublas.R32F, n, b, cublas.R32F, n, cublas.Float32Scalar(0), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

// hostDefault passes host values in the default mode.
func hostDefault(a, b, c driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.Float32Scalar(1), a, cublas.R32F, n, b, cublas.R32F, n, cublas.Float32Scalar(0), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

// parameter takes a handle whose mode is set elsewhere.
func parameter(h cublas.Handle, a, b, c, alpha driver.DevicePtr, n int32) error {
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.DeviceScalar(alpha), a, cublas.R32F, n, b, cublas.R32F, n, cublas.DeviceScalar(alpha), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

// conditional sets the mode on one path only.
func conditional(dev bool, a, b, c, alpha driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	if dev {
		h.SetPointerMode(cublas.PointerModeDevice)
	}
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.DeviceScalar(alpha), a, cublas.R32F, n, b, cublas.R32F, n, cublas.DeviceScalar(alpha), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

// conditionalHost passes host values after a mode set on one path only.
func conditionalHost(dev bool, a, b, c driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	if dev {
		h.SetPointerMode(cublas.PointerModeDevice)
	}
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.Float32Scalar(1), a, cublas.R32F, n, b, cublas.R32F, n, cublas.Float32Scalar(0), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

// loop sets device mode at the bottom of a loop, which the next iteration
// sees.
func loop(a, b, c, alpha driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	for range 2 {
		h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.DeviceScalar(alpha), a, cublas.R32F, n, b, cublas.R32F, n, cublas.DeviceScalar(alpha), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
		h.SetPointerMode(cublas.PointerModeDevice)
	}
	return nil
}

// configured hands the handle to a function that may set its mode.
func configured(a, b, c, alpha driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	useDevice(h)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.DeviceScalar(alpha), a, cublas.R32F, n, b, cublas.R32F, n, cublas.DeviceScalar(alpha), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

func useDevice(h cublas.Handle) { h.SetPointerMode(cublas.PointerModeDevice) }

// variableMode sets a mode that is not a constant.
func variableMode(m cublas.PointerMode, a, b, c, alpha driver.DevicePtr, n int32) error {
	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	h.SetPointerMode(m)
	return h.GemmEx(cublas.OpN, cublas.OpN, n, n, n, cublas.DeviceScalar(alpha), a, cublas.R32F, n, b, cublas.R32F, n, cublas.DeviceScalar(alpha), c, cublas.R32F, n, cublas.Compute32F, cublas.GemmDefault)
}

func sparseDevice(m cusparse.ConstSpMatDescr, x cusparse.ConstDnVecDescr, y cusparse.DnVecDescr, alpha driver.DevicePtr) error {
	h, err := cusparse.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	return h.SpMV(cusparse.OperationNonTranspose, cusparse.DeviceScalar(alpha), m, x, cusparse.Float32Scalar(0), y, cusparse.R32F, cusparse.SpmvAlgDefault, 0) // want `cusparse.DeviceScalar\(alpha\) is a device Scalar, but h is in host pointer mode`
}

func sparseHost(m cusparse.ConstSpMatDescr, x cusparse.ConstDnVecDescr, y cusparse.DnVecDescr, alpha, beta driver.DevicePtr) error {
	h, err := cusparse.Create()
	if err != nil {
		return err
	}
	defer h.Close()
	h.SetPointerMode(cusparse.PointerModeDevice)
	if err := h.SpMV(cusparse.OperationNonTranspose, cusparse.DeviceScalar(alpha), m, x, cusparse.DeviceScalar(beta), y, cusparse.R32F, cusparse.SpmvAlgDefault, 0); err != nil {
		return err
	}
	return h.SpMV(cusparse.OperationNonTranspose, cusparse.Float32Scalar(1), m, x, cusparse.DeviceScalar(beta), y, cusparse.R32F, cusparse.SpmvAlgDefault, 0) // want `cusparse.Float32Scalar\(1\) is a host float32 Scalar, but h is in device pointer mode`
}

// sparseParameter takes a handle whose mode is set elsewhere.
func sparseParameter(h cusparse.Handle, m cusparse.ConstSpMatDescr, x cusparse.ConstDnVecDescr, y cusparse.DnVecDescr, alpha, beta driver.DevicePtr) error {
	return h.SpMV(cusparse.OperationNonTranspose, cusparse.DeviceScalar(alpha), m, x, cusparse.DeviceScalar(beta), y, cusparse.R32F, cusparse.SpmvAlgDefault, 0)
}

func matmulDevice(h cublaslt.Handle, a, b, c, alpha driver.DevicePtr, l cublaslt.MatrixLayout, s driver.Stream) error {
	desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
	if err != nil {
		return err
	}
	return h.Matmul(desc, cublaslt.DeviceScalar(alpha), a, l, b, l, cublaslt.Float32Scalar(0), c, l, c, l, nil, 0, 0, s) // want `cublaslt.DeviceScalar\(alpha\) is a device Scalar, but desc is in host pointer mode`
}

// matmulDeviceMode sets the pointer mode attribute, whose value is not
// followed.
func matmulDeviceMode(h cublaslt.Handle, a, b, c, alpha, beta driver.DevicePtr, l cublaslt.MatrixLayout, s driver.Stream) error {
	desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
	if err != nil {
		return err
	}
	mode := int32(1)
	if err := desc.SetAttribute(cublaslt.MatmulDescPointerMode, unsafe.Pointer(&mode), 4); err != nil {
		return err
	}
	return h.Matmul(desc, cublaslt.DeviceScalar(alpha), a, l, b, l, cublaslt.DeviceScalar(beta), c, l, c, l, nil, 0, 0, s)
}

// matmulOtherAttribute sets an attribute that leaves the mode host.
func matmulOtherAttribute(h cublaslt.Handle, a, b, c, alpha driver.DevicePtr, l cublaslt.MatrixLayout, s driver.Stream) error {
	desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
	if err != nil {
		return err
	}
	scale := int32(0)
	desc.SetAttribute(cublaslt.MatmulDescScaleType, unsafe.Pointer(&scale), 4)
	return h.Matmul(desc, cublaslt.DeviceScalar(alpha), a, l, b, l, cublaslt.Float32Scalar(0), c, l, c, l, nil, 0, 0, s) // want `cublaslt.DeviceScalar\(alpha\) is a device Scalar, but desc is in host pointer mode`
}
