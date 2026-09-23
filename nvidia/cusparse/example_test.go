package cusparse_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cusparse"
)

// Example creates a cuSPARSE handle and a CSR matrix descriptor for the
// generic API. It returns early when the cuSPARSE library is not
// available. The row-offset, column-index, and value arrays are CUDA
// device pointers, e.g. from the driver package's MemAlloc. It has no Output
// comment, so it is compiled but not run: what it prints depends on the
// host's cuSPARSE.
func Example() {
	h, err := cusparse.Create()
	if err != nil {
		fmt.Println("cuSPARSE unavailable:", err)
		return
	}
	defer h.Destroy()

	version, err := h.GetVersion()
	if err != nil {
		fmt.Println("version:", err)
		return
	}
	fmt.Println("cuSPARSE version:", version)

	var dRowOffsets, dColInd, dValues uintptr // device allocations
	mat, err := cusparse.CreateCsr(4, 4, 9, dRowOffsets, dColInd, dValues,
		cusparse.Index32I, cusparse.Index32I,
		cusparse.IndexBaseZero, cusparse.R32F)
	if err != nil {
		fmt.Println("csr:", err)
		return
	}
	defer cusparse.ConstSpMatDescr(mat).DestroySpMat()
	fmt.Println("4x4 CSR descriptor ready")
}
