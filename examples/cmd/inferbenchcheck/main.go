// Command inferbenchcheck validates a matched inference benchmark campaign.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/examples/internal/inferbench"
)

func main() {
	flag.Parse()
	if flag.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "usage: inferbenchcheck campaign.json raw-directory")
		os.Exit(2)
	}
	f, err := os.Open(flag.Arg(0))
	if err != nil {
		fail(err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var c inferbench.Campaign
	if err := dec.Decode(&c); err != nil {
		fail(fmt.Errorf("decode campaign: %w", err))
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		fail(errors.New("campaign contains trailing data"))
	}
	if err := c.ValidateRawSet(flag.Arg(1)); err != nil {
		fail(fmt.Errorf("invalid campaign: %w", err))
	}
	fmt.Printf("VALID: %d matched arms, %d scheduled samples\n", len(c.Runs), len(c.Order))
}

func fail(err error) { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
