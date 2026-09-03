// Command inferbenchluminal emits the fail-closed Luminal adapter boundary.
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
	capabilitiesPath := flag.String("capabilities", "", "path to measured Luminal capability JSON")
	output := flag.String("output", "", "path for the atomic refusal receipt")
	flag.Parse()
	if *capabilitiesPath == "" || *output == "" {
		fail(errors.New("usage: inferbenchluminal -capabilities capabilities.json -output refusal.json"))
	}
	f, err := os.Open(*capabilitiesPath)
	if err != nil {
		fail(err)
	}
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var capabilities inferbench.AdapterCapabilities
	err = dec.Decode(&capabilities)
	if err == nil {
		err = expectEOF(dec)
	}
	f.Close()
	refusal := inferbench.AdapterRefusal{SchemaVersion: inferbench.SchemaVersion, State: "refused", Arm: "luminal"}
	if err != nil {
		refusal.Code, refusal.Reason = "invalid_capability_receipt", err.Error()
	} else if err := inferbench.CheckAdapterCapabilities("luminal", capabilities); err != nil {
		refusal.Code, refusal.Reason = "unequal_runtime_capability", err.Error()
	} else {
		refusal.Code = "adapter_not_implemented"
		refusal.Reason = "no reviewed Luminal production-serving adapter is available"
	}
	if err := inferbench.WriteJSONAtomic(*output, refusal); err != nil {
		fail(err)
	}
	fmt.Printf("REFUSED: %s\n", refusal.Code)
}

func expectEOF(dec *json.Decoder) error {
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return errors.New("capability receipt contains trailing data")
	}
	return nil
}

func fail(err error) { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
