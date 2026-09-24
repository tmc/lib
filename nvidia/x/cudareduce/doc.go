// Package cudareduce provides deterministic column reductions for backward
// passes: bias gradients, per-group embedding gradients, and the LayerNorm
// weight and bias gradients.
//
// Each reduction runs in two stages. Stage 1 splits the rows into a fixed
// number of ranges, chosen by [Split] from the shape alone, and sums each
// column over each range in double. Stage 2 sums the ranges in order and
// rounds once to float32. There are no atomics, so results are the same on
// every run and every device.
//
// [Header] is the NVRTC-includable cudareduce.cuh. A trainer can include it
// to fuse a reduction into its own stage 1 kernel and finish with
// [Reducer.Finish]. [Reducer] launches the common cases:
//
//	red, err := cudareduce.New(dev)
//	...
//	scratch, err := cudamem.Alloc[float64](dev, cudareduce.ScratchLen(rows, n, 2))
//	...
//	err = red.ColumnSum(stream, dbias, dy, rows, n, scratch, false)
//	err = red.LayerNormGrad(stream, dw, db, dy, x, mean, rstd, rows, n, scratch, true)
package cudareduce
