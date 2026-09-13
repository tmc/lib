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
// Basic usage:
//
//	suite := NewBenchmarkSuite(DefaultConfig())
//	report := suite.Run()
//	fmt.Println(report.Table())
package main
