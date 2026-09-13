// Package cudaflightrecorder coordinates continuous in-memory flight recording
// across host Go execution traces and GPU device observations.
//
// It produces self-contained, multi-modal diagnostic bundles (.tar) upon application
// crashes, watchdog triggers, or explicit snapshot requests without touching GPU
// hardware during failure paths (zero-device crash dump).
//
// # Basic Usage
//
// Configure and start a flight recorder with a 10-second circular buffer:
//
//	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
//		Window:   10 * time.Second,
//		Capacity: 65536,
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
//	manifest, err := rec.Snapshot(ctx, f, "watchdog_timeout")
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Printf("Recorded %d GPU records across %v\n", manifest.GPURecords, manifest.GPUSpan())
package cudaflightrecorder
