// Package cudaflightrecorder coordinates continuous in-memory flight recording
// across host Go execution traces and GPU device observations.
//
// Snapshots preserve recent observations and configured artifacts in tar bundles
// for performance analysis, workload comparison, and debugging. Applications choose
// when to snapshot, such as at a workload boundary, on a latency regression,
// periodically, or while investigating a failure. The snapshot reason is a
// caller-supplied label.
//
// # Basic Usage
//
// Configure and start a flight recorder with a 10-second circular buffer and Go tracing:
//
//	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
//		Window:         10 * time.Second,
//		Capacity:       65536,
//		GoTraceEnabled: true,
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	if err := rec.Start(); err != nil {
//		log.Fatal(err)
//	}
//	defer rec.Stop()
//
// # Snapshot Generation
//
// Capture an in-memory snapshot to a diagnostic bundle:
//
//	manifest, err := rec.Snapshot(ctx, f, "workload_boundary")
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Printf("Recorded %d GPU records across %v (overlap: %v, reliable: %t)\n",
//		manifest.GPURecords, manifest.GPUSpan(), manifest.OverlapSpan(), manifest.OverlapReliable)
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaflightrecorder
