package main

import (
	"fmt"
	"io"
	"runtime"
	"time"
)

// Sampling seams, replaced in tests.
var (
	readLoad = loadAverage
	sleep    = time.Sleep
)

// baseline reports how often the limit is met on an idle host.
//
// The gate reads the load once before a run and once after, so a pass
// followed by a fail says only that the two readings differ. It cannot
// say whether the limit was ever within reach, and a limit the host
// exceeds while idle refuses every measurement rather than
// discriminating among them.
//
// The host qualifies only if every sample is under the limit. That is
// strictly stricter than a single reading, which is the only direction
// this gate may move. A failure means the quiet windows are unstable,
// not that no interval can qualify: the sampled distribution says
// nothing about a future one.
func baseline(cfg config, stdout io.Writer) error {
	min, max, sum := 0.0, 0.0, 0.0
	over := 0
	for i := 0; i < cfg.samples; i++ {
		if i > 0 {
			sleep(cfg.interval)
		}
		load, err := readLoad()
		if err != nil {
			return err
		}
		if i == 0 || load < min {
			min = load
		}
		if load > max {
			max = load
		}
		sum += load
		if load > cfg.maxLoad {
			over++
		}
	}
	fmt.Fprintf(stdout, "%d samples over %s, %d CPUs, limit %.2f\n",
		cfg.samples, time.Duration(cfg.samples-1)*cfg.interval, runtime.NumCPU(), cfg.maxLoad)
	fmt.Fprintf(stdout, "load average min %.2f max %.2f mean %.2f, %d over the limit\n",
		min, max, sum/float64(cfg.samples), over)
	if over > 0 {
		fmt.Fprintln(stdout, "  UNSTABLE: this host exceeds the limit without the measurement running")
		return errDegraded
	}
	fmt.Fprintln(stdout, "limit is attainable")
	return nil
}
