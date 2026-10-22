//go:build linux

package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// encodeQuantQ80Strided quantizes batch rows of cols floats separated by stride
// floats. in begins at the first row and ends at the last row's final element.
func (r *cudaFoundation) encodeQuantQ80Strided(in cudamem.Slice[float32], out cudamem.Slice[byte], cols, stride, batch int) error {
	if cols <= 0 || cols%32 != 0 || stride < cols || uint64(stride) > math.MaxUint32 || batch <= 0 ||
		uint64(batch) > math.MaxInt32/uint64(cols/32) ||
		uint64(in.Len()) != uint64(batch-1)*uint64(stride)+uint64(cols) ||
		uint64(out.Len()) != uint64(batch)*uint64(cols/32)*36 {
		return fmt.Errorf("cuda: strided Q8_0 activation quantization: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadQ80Quantizer(); err != nil {
			return err
		}
		if r.quantQ80StridedFn == 0 {
			fn, err := r.quantQ80Mod.Function("ds4_quant_q8_0_strided")
			if err != nil {
				return fmt.Errorf("cuda: find strided Q8_0 quantizer: %w", err)
			}
			r.quantQ80StridedFn = fn
		}
		if err := cudalaunch.Kernel(r.quantQ80StridedFn).Grid(uint32(batch*(cols/32)), 1, 1).Block(32, 1, 1).On(r.stream()).Launch(in, out, uint32(cols/32), uint32(stride)); err != nil {
			return fmt.Errorf("cuda: launch strided Q8_0 activation quantization: %w", err)
		}
		return nil
	})
}
