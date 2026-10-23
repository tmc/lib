// Command cudarepro replays an explicitly saved isolated CUDA launch.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudarepro"
	"github.com/tmc/lib/nvidia/x/cudasanitizer"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	findings := flag.String("attribute", "", "attribute a sanitizer log to this relocated capture instead of replaying")
	record := flag.String("record", "", "write relocated launch record before execution (required)")
	output := flag.String("output", "", "write output buffer bytes as JSON")
	device := flag.Int("device", 0, "CUDA device ordinal")
	flag.Parse()
	if flag.NArg() != 1 || (*record == "" && *findings == "") {
		return fmt.Errorf("usage: cudarepro -record replay.json [-output output.json] capture.json; or cudarepro -attribute sanitizer.log replay.json")
	}
	f, err := os.Open(flag.Arg(0))
	if err != nil {
		return err
	}
	launch, err := cudarepro.Read(f, 256<<20)
	f.Close()
	if err != nil {
		return err
	}
	if *findings != "" {
		data, err := os.ReadFile(*findings)
		if err != nil {
			return err
		}
		log := cudasanitizer.ParseLog(string(data))
		for _, f := range log.Findings {
			row := struct {
				Finding    cudasanitizer.Finding
				Candidates []cudarepro.Match
			}{f, cudarepro.Attribute(f, launch)}
			if err := json.NewEncoder(os.Stdout).Encode(row); err != nil {
				return err
			}
		}
		return nil
	}
	dev, err := cuda.NewDevice(*device)
	if err != nil {
		return err
	}
	buffers, err := launch.Replay(dev, func(l *cudarepro.Launch) error {
		f, err := os.OpenFile(*record, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			return err
		}
		err = l.Write(f)
		closeErr := f.Close()
		if err != nil {
			return err
		}
		return closeErr
	})
	if err != nil {
		return releaseOwners(err, dev.Close)
	}
	defer dev.Close()
	if *output != "" {
		f, err := os.OpenFile(*output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			return err
		}
		err = json.NewEncoder(f).Encode(buffers)
		closeErr := f.Close()
		if err != nil {
			return err
		}
		if closeErr != nil {
			return closeErr
		}
	}
	fmt.Println("replay complete")
	return nil
}

// releaseOwners closes an io.Closer carried by cause, then each dependent in
// order, and returns cause joined with the first cleanup failure. Nothing after
// a failed cleanup is closed: it still backs the failed resource, and an owner
// that failed to close remains reachable through the returned error.
func releaseOwners(cause error, dependents ...func() error) error {
	var owner io.Closer
	if errors.As(cause, &owner) {
		if err := owner.Close(); err != nil {
			return errors.Join(cause, fmt.Errorf("cudarepro: release replay owner: %w", err))
		}
	}
	for _, release := range dependents {
		if err := release(); err != nil {
			return errors.Join(cause, fmt.Errorf("cudarepro: cleanup: %w", err))
		}
	}
	return cause
}
