// Package cudaroofline models empirical arithmetic intensity (FLOPs/byte)
// and evaluates kernel performance against empirical device hardware ceilings.
//
// It calculates whether a GPU kernel launch is memory bandwidth-bound,
// compute throughput-bound, or unconstrained, and generates standard SVG
// roofline visualizations.
//
// # Basic Usage
//
// Evaluate a kernel launch on an NVIDIA Blackwell or Hopper device:
//
//	model, err := cudaroofline.NewModel("sm_121")
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
