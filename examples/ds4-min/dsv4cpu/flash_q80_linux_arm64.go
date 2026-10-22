package dsv4cpu

// nativeQ80Dot follows the pinned GCC 13.3 DGX owning reference: integer
// reduction within each block, rounded scale product, then one scalar FMADD.
// The retained C0 matvec_q8_0_worker has FMUL at c4bc and FMADD at c52c.
func nativeQ80Dot(row []byte, x []Q80Activation) float32 {
	return scalarQ80Dot(row, x)
}
