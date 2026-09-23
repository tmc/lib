// Cuda-trace writes a Perfetto trace holding a real device interval measured
// with CUDA events, plus a crash-safe post-mortem flight recording.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// copyBytes is large enough that the device interval clears the calibration
// uncertainty on a current GPU.
const copyBytes = 64 << 20

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(home, "tmp"), 0777); err != nil {
		return err
	}
	stamp := time.Now().UnixNano()
	path := filepath.Join(home, "tmp", fmt.Sprintf("lib-cuda-trace-%d.pftrace", stamp))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	now := uint64(time.Now().UnixNano())
	pw := perfetto.NewWriter(f)
	if err := pw.WriteClockSnapshot(now, now); err != nil {
		return err
	}
	tracer := cudatrace.New(f)
	if _, err := tracer.RegisterStream(nil, "host"); err != nil {
		return err
	}
	if err := tracer.Span(nil, "prepare inputs", "example", func() {}); err != nil {
		return err
	}
	if err := tracer.Instant(nil, "ready", "example"); err != nil {
		return err
	}

	if err := traceDeviceCopy(tracer); err != nil {
		return err
	}

	postmortem := filepath.Join(home, "tmp", fmt.Sprintf("lib-cuda-trace-%d-postmortem.pftrace", stamp))
	pf, err := os.Create(postmortem)
	if err != nil {
		return err
	}
	defer pf.Close()
	if err := tracer.DumpPostMortem(pf); err != nil {
		return err
	}
	fmt.Printf("wrote %s and %s\n", path, postmortem)
	return nil
}

// traceDeviceCopy records a device-to-device copy as a span whose timestamps
// come from CUDA events mapped onto the host clock.
func traceDeviceCopy(tracer *cudatrace.Tracer) error {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		fmt.Printf("no CUDA device (%v); trace holds host spans only\n", err)
		return nil
	}
	defer dev.Close()

	stream, err := dev.NewStream()
	if err != nil {
		return err
	}
	defer stream.Destroy()

	events := make([]*cuda.Event, 3)
	for i := range events {
		e, err := dev.NewEvent()
		if err != nil {
			return err
		}
		defer e.Destroy()
		events[i] = e
	}
	anchor, start, end := events[0], events[1], events[2]

	clock, err := cudatrace.Calibrate(context.Background(), stream, anchor)
	if err != nil {
		return err
	}

	src, err := cudamem.Alloc[byte](dev, copyBytes)
	if err != nil {
		return err
	}
	defer src.Free()
	dst, err := cudamem.Alloc[byte](dev, copyBytes)
	if err != nil {
		return err
	}
	defer dst.Free()

	trackID, err := tracer.RegisterStream(stream, "device")
	if err != nil {
		return err
	}

	if err := start.Record(stream); err != nil {
		return err
	}
	if err := dst.CopyDeviceToDeviceAsync(src, stream); err != nil {
		return err
	}
	if err := end.Record(stream); err != nil {
		return err
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		return err
	}

	startNs, err := clock.HostNs(start)
	if err != nil {
		return err
	}
	endNs, err := clock.HostNs(end)
	if err != nil {
		return err
	}
	elapsed, err := end.ElapsedSince(start)
	if err != nil {
		return err
	}
	if err := tracer.RecordSpan(trackID, startNs, endNs, "CopyDeviceToDevice", "memcpy",
		perfetto.NewIntAnnotation("bytes", copyBytes),
		perfetto.NewIntAnnotation("clock.uncertainty_ns", clock.UncertaintyNs()),
	); err != nil {
		return err
	}
	fmt.Printf("device copy %d bytes in %v (span %d..%d, calibration +/-%dns)\n",
		copyBytes, elapsed, startNs, endNs, clock.UncertaintyNs())
	return nil
}
