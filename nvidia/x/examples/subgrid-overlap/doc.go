// Package main demonstrates sub-grid early release and true intra-kernel pipeline
// overlap using the PTX griddepcontrol instruction.
//
// On Hopper (sm_90) and later architectures, Programmatic Dependent Launch (PDL)
// allows a running grid to release downstream dependent kernel nodes before the
// current grid finishes. When all CTAs in the producer grid execute
// "griddepcontrol.launch_dependents", the runtime system schedules the dependent
// consumer kernel concurrently onto available SMs. The consumer kernel uses
// "griddepcontrol.wait" before accessing later data to ensure memory visibility.
//
// Nothing in this package executes on a GPU. It generates the PTX module, audits
// it, and evaluates an analytic cycle model of the three launch paradigms; every
// duration it reports is model cycles divided by an assumed clock. The reports
// say so in their own text, because a figure printed in microseconds is
// otherwise indistinguishable from one a profiler measured.
//
// Basic usage:
//
//	report, err := DefaultBenchmarkSuite().Run()
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Print(report.Table())
package main
