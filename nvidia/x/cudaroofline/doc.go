// Package cudaroofline models kernel arithmetic intensity (FLOPs/byte) and
// evaluates a launch against device ceilings.
//
// It calculates whether a GPU kernel launch is memory bandwidth-bound, compute
// throughput-bound, or unconstrained, and generates standard SVG roofline
// visualizations.
//
// # What the ceilings are
//
// A ceiling here is one of two kinds of number and says which: Measured, taken
// from a microbenchmark on the part named in its comment, or Unverified,
// carried from vendor documentation and never measured in this repository. A
// ratio against an unverified ceiling is a hypothesis about the hardware, not a
// reading of it. Only the GB10 (sm_121, sm_121a) memory bandwidths are
// measured; every other figure in the table is unverified.
//
// A ceiling is also specific to a Residency, because bandwidth belongs to the
// memory path rather than to the architecture. On GB10 the same kernel reading
// the same bytes attains 239.6 GB/s from device-resident memory and 162.4 GB/s
// when the bytes are read through host page tables, a factor of 1.43. Weights
// mmap'd from a model file take the second path. Dividing by the first is how a
// campaign comes to believe in savings that cannot be collected.
//
// The marking travels with the result. A KernelPoint carries the axis its
// verdict rests on and that axis's provenance, so a consumer holding only the
// point can tell a ratio against a measured ceiling from a ratio against a
// datasheet figure. On GB10 a memory-bound verdict is Measured and a
// compute-bound verdict is Unverified, from the same ceiling.
//
// Efficiency above 1.0 is reported, not clamped: a launch that beats its
// ceiling has refuted the ceiling, and that is the most useful thing this
// package can tell you.
//
// # Basic Usage
//
// Evaluate a kernel launch that reads mmap'd weights on a GB10:
//
//	model, err := cudaroofline.NewModel("sm_121", cudaroofline.HostPageTableResident)
//	if err != nil {
//		log.Fatal(err)
//	}
//	point := model.Evaluate("q8_0_matvec", 1048576, 524288, 12000)
//	fmt.Printf("%s: %.2f TFLOPS, %.2f FLOP/B (%s)\n",
//		point.Name, point.AttainedTFLOPS, point.Intensity, point.Classification)
//
// # Visualization
//
// Render an SVG plot of evaluated kernels against the device ceilings:
//
//	err := model.RenderSVG(w, []cudaroofline.KernelPoint{point})
package cudaroofline
