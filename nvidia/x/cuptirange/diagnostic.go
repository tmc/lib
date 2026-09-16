package cuptirange

// ObservedLaunch describes a launch seen in the profiling callback window.
// It is diagnostic evidence, not a measured range or a counter attribution.
// API names the driver entry point; Symbol names a direct kernel or a kernel
// expanded from a retained graph order proof.
type ObservedLaunch struct {
	API, Symbol string
}

// LaunchTrace retains the most recent callback window after a failed capture.
// Launches excludes preparation and contains at most Limits.MaxRanges entries.
// The window may be partial; an absent launch does not prove it did not execute.
// CompletedPasses counts callback windows whose trace checks succeeded, not
// native profiling passes or verified workload outputs. A later Stop may fail.
type LaunchTrace struct {
	Launches        []ObservedLaunch
	CompletedPasses uint
}

func (k *kernelNames) diagnosticTrace() *LaunchTrace {
	if k == nil {
		return nil
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	trace := &LaunchTrace{CompletedPasses: k.passes, Launches: make([]ObservedLaunch, len(k.current))}
	for i, v := range k.current {
		trace.Launches[i] = ObservedLaunch{API: v.API, Symbol: v.Symbol}
	}
	return trace
}
