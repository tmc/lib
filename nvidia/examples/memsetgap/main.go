// Memsetgap is a capture target for validating that cudaprof counts CUPTI
// memset activity as device-busy time.
//
// The workload alternates two device operations on a single stream: a
// device-to-device copy, which cudaprof has always collected, and a memset,
// which it collected for the first time in CL 854b067a.
//
// The two arms of the experiment differ in the TOOL, not in this program. Both
// arms run this binary unchanged, with the same flags, doing the same device
// work; what changes is whether the cudaprof building the capture enables
// CUPTI_ACTIVITY_KIND_MEMSET:
//
//	git checkout 854b067a && go build -o cudaprof-with ./x/cmd/cudaprof
//	git checkout 854b067a^ && go build -o cudaprof-without ./x/cmd/cudaprof
//	./cudaprof-with    capture -o with.pprof    -- memsetgap
//	./cudaprof-without capture -o without.pprof -- memsetgap
//	./cudaprof-with gaps --diff_base=without.pprof --json with.pprof
//
// That the arms are two builds rather than a flag is the whole design. An
// earlier version of this target took an environment variable that skipped
// ISSUING the memsets, which measures something else entirely: cudaprof's
// SpanNanos runs from the first start to the greatest end and IdleNanos sums
// the gaps between merged busy intervals, so Span == Active + Idle
// identically. Removing device work shrinks the span and leaves idle alone --
// the host sleeps between pairs are the same in both arms either way. Only an
// arm that RUNS the memsets and fails to COLLECT them moves that time from the
// active column into the idle one, which is the defect under test.
//
// The prediction, with two anchors that fail if the target misbehaves:
//
//	idle(without) - idle(with) == active(with) - active(without) == memset total
//	SpanNanos      unchanged between arms
//	copy count and copy total unchanged between arms
//
// Everything runs on one stream, so the operations serialize and no memset
// overlaps a copy. That is deliberate: ActiveNanos sums MERGED intervals
// rather than raw durations, so an overlapping memset would add less busy time
// than its own duration and the arms could differ by any amount between zero
// and the memset total. Serialized, the prediction is exact and a shortfall is
// a finding rather than an expected artifact.
//
// No loop memset is ever the first or last activity in the capture: the
// priming copy precedes them and a trailing copy follows. SpanNanos is
// measured from the extreme activities, so a memset at either edge would move
// the span between arms rather than moving time between the idle and active
// columns.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaprofagent"
	"github.com/tmc/lib/nvidia/x/nvtx"
)

func main() {
	iters := flag.Int("iters", 64, "alternating copy/memset pairs to run")
	mib := flag.Int("mib", 256, "size of each buffer in MiB")
	gapUS := flag.Int("gap_us", 200, "host sleep between pairs, to separate the device intervals")
	markers := flag.Bool("nvtx", true, "emit an NVTX range around each pair")
	flag.Parse()

	err := run(*iters, *mib, *gapUS, *markers)

	// Under capture this is the only finalization route that fires. The agent's
	// other three all run through libc -- the ELF destructor and atexit from
	// exit(3), and a SIGTERM terminator whose handler the Go runtime has
	// already replaced -- and runtime.exit issues exit_group directly, so a Go
	// target that simply returns from main leaves the journal unfinalized with
	// no activity in it. Measured on GB10: this target collected zero records
	// where a C++ target doing the same device work under the same agent and
	// the same cudaprof collected cleanly. It is not a defer: os.Exit below
	// would skip it.
	if cudaprofagent.Finalize() {
		fmt.Println("memsetgap: finalized the attached capture agent")
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, "memsetgap:", err)
		os.Exit(1)
	}
}

func run(iters, mib, gapUS int, markers bool) error {
	if err := cuda.Init(); err != nil {
		return err
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	defer dev.Close()

	return dev.WithContext(func() error {
		bytes := uint(mib) << 20
		src, err := driver.MemAlloc(bytes)
		if err != nil {
			return fmt.Errorf("allocating %d MiB source: %w", mib, err)
		}
		defer src.MemFree()
		dst, err := driver.MemAlloc(bytes)
		if err != nil {
			return fmt.Errorf("allocating %d MiB destination: %w", mib, err)
		}
		defer dst.MemFree()

		stream, err := dev.NewStream()
		if err != nil {
			return err
		}
		defer stream.Destroy()
		raw := stream.Raw()

		// Commit both buffers before the measured loop. An untouched
		// cuMemAlloc backs no pages, so the first operation on it would pay a
		// fault cost belonging to allocation rather than to the operation being
		// measured. Priming with copies rather than memsets keeps the memset
		// count in the capture equal to the count this program reports: a
		// primer the counter does not see would read as contamination by the
		// check at the end.
		if err := src.MemcpyDtoDAsync(dst, bytes, raw); err != nil {
			return fmt.Errorf("priming source: %w", err)
		}
		if err := dst.MemcpyDtoDAsync(src, bytes, raw); err != nil {
			return fmt.Errorf("priming destination: %w", err)
		}
		if err := stream.Synchronize(context.Background()); err != nil {
			return fmt.Errorf("priming: %w", err)
		}

		for i := range iters {
			if markers {
				nvtx.RangePush(fmt.Sprintf("memsetgap/pair/%d", i))
			}
			// The copy is the anchor: it is collected by both arms, so its
			// count and total must not move between them.
			if err := dst.MemcpyDtoDAsync(src, bytes, raw); err != nil {
				return fmt.Errorf("copy %d: %w", i, err)
			}
			if err := dst.MemsetD32Async(uint32(i), uint(bytes/4), raw); err != nil {
				return fmt.Errorf("memset %d: %w", i, err)
			}
			if err := stream.Synchronize(context.Background()); err != nil {
				return fmt.Errorf("pair %d: %w", i, err)
			}
			if markers {
				nvtx.RangePop()
			}
			// Separate consecutive pairs so each lands as its own merged
			// interval. Without a gap the copies and memsets coalesce into one
			// interval and the capture has nothing to attribute.
			time.Sleep(time.Duration(gapUS) * time.Microsecond)
		}

		// A final copy so the last activity in the capture is never a memset.
		if err := dst.MemcpyDtoDAsync(src, bytes, raw); err != nil {
			return fmt.Errorf("trailing copy: %w", err)
		}
		if err := stream.Synchronize(context.Background()); err != nil {
			return err
		}

		// The target states its own counts so the capture can be checked
		// against what ran rather than against what the reader assumes ran.
		// The memset count is what the collecting arm must report and what the
		// non-collecting arm must be missing; the copy count must match in
		// both. Fewer memsets than this in the with arm is a collection
		// failure, more is someone else's work landing in the capture.
		fmt.Printf("memsetgap: iters=%d bytes=%d copies=%d memsets=%d nvtx_ranges=%d\n",
			iters, bytes, iters+3, iters, markerCount(markers, iters))
		return nil
	})
}

func markerCount(on bool, n int) int {
	if !on {
		return 0
	}
	return n
}
