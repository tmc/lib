// Cuda-trace writes a Perfetto trace and a crash-safe post-mortem flight recording.
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

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
