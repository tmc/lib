// Cuptimetrics lists the CUPTI range-profiling metrics an attached GPU offers.
//
// It exists because a metric name that a chip does not offer fails at
// collection time, well after the run has been set up, and the vendor's
// published lists cover architectures rather than the device in front of you.
//
// Usage:
//
//	cuptimetrics                        list every reachable metric
//	cuptimetrics -device 1              list metrics for device 1
//	cuptimetrics -additive              list only metrics that sum across ranges
//	cuptimetrics -match sm__ctas        list metrics whose name contains a string
//	cuptimetrics -sets                  list the curated sets instead
//
// The listing is an inventory for choosing metrics, not a set to collect: each
// scheduling group costs a replay pass, so asking for everything at once
// exceeds any sane pass budget.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptibench"
)

var (
	device   = flag.Int("device", 0, "CUDA device ordinal")
	additive = flag.Bool("additive", false, "list only metrics that sum across disjoint ranges")
	match    = flag.String("match", "", "list only metrics whose name contains this substring")
	sets     = flag.Bool("sets", false, "list the curated metric sets and exit")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "cuptimetrics: %v\n", err)
		os.Exit(1)
	}
}

func run() (err error) {
	w := tabwriter.NewWriter(os.Stdout, 0, 8, 2, ' ', 0)
	defer w.Flush()
	if *sets {
		for _, name := range cuptibench.Sets() {
			metrics, err := cuptibench.Resolve(name)
			if err != nil {
				return err
			}
			fmt.Fprintf(w, "%s\t%s\n", name, strings.Join(metrics, " "))
		}
		return nil
	}
	dev, err := cuda.NewDevice(*device)
	if err != nil {
		return err
	}
	defer func() {
		if cleanup := dev.Close(); cleanup != nil && err == nil {
			err = cleanup
		}
	}()
	all, err := cuptibench.Reachable(dev)
	if err != nil {
		return err
	}
	fmt.Fprintln(w, "NAME\tUNIT\tADDITIVE\tDESCRIPTION")
	shown := 0
	for _, m := range all {
		if *additive && !m.Additive {
			continue
		}
		if *match != "" && !strings.Contains(m.Name, *match) {
			continue
		}
		shown++
		fmt.Fprintf(w, "%s\t%s\t%t\t%s\n", m.Name, m.Unit, m.Additive, m.Description)
	}
	fmt.Fprintf(w, "\n# %d of %d metrics on device %d\n", shown, len(all), *device)
	return nil
}
