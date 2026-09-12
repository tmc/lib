// Journalstat decodes a producer journal with the strict wire decoder and
// reports per-kind counts together with the bootstrap, capability, and
// finalized payloads.
//
// It exists so a journal produced on a remote host can be validated by the
// same decoder the controller uses, rather than scraped out of the file.
//
// Usage:
//
//	journalstat [-max-payload n] [-max-records n] journal...
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

func main() {
	maxPayload := flag.Uint("max-payload", 8192, "maximum payload bytes per record")
	maxRecords := flag.Uint("max-records", 1<<24, "maximum records per journal")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: journalstat journal...")
		os.Exit(2)
	}
	limits := wire.Limits{MaxPayload: uint32(*maxPayload), MaxRecords: uint32(*maxRecords)}
	status := 0
	for _, name := range flag.Args() {
		if err := report(name, limits); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", name, err)
			status = 1
		}
	}
	os.Exit(status)
}

type summary struct {
	Journal    string            `json:"journal"`
	Bytes      int64             `json:"bytes"`
	Records    uint64            `json:"records"`
	Kinds      map[string]uint64 `json:"kinds"`
	Dropped    uint64            `json:"dropped"`
	Sequence   uint64            `json:"last_sequence"`
	Process    uint64            `json:"process"`
	Generation uint64            `json:"generation"`
	Bootstrap  json.RawMessage   `json:"bootstrap,omitempty"`
	Capability json.RawMessage   `json:"capability,omitempty"`
	Finalized  json.RawMessage   `json:"finalized,omitempty"`
	Failures   []string          `json:"failures,omitempty"`
	DecodeErr  string            `json:"decode_error,omitempty"`
}

var kindNames = map[uint16]string{
	1: "bootstrap", 2: "activity", 3: "finalized",
	4: "capability", 5: "failure", 6: "external", 7: "api",
}

func report(name string, limits wire.Limits) error {
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	s := summary{Journal: name, Bytes: info.Size(), Kinds: map[string]uint64{}}
	d := wire.NewDecoder(f, limits)
	for {
		r, err := d.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			s.DecodeErr = err.Error()
			break
		}
		s.Records++
		kind := kindNames[r.Kind]
		if kind == "" {
			kind = fmt.Sprintf("unknown-%d", r.Kind)
		}
		s.Kinds[kind]++
		s.Dropped, s.Sequence = r.Dropped, r.Sequence
		s.Process, s.Generation = r.Process, r.Generation
		switch r.Kind {
		case 1:
			s.Bootstrap = clone(r.Payload)
		case 3:
			s.Finalized = clone(r.Payload)
		case 4:
			s.Capability = clone(r.Payload)
		case 5:
			s.Failures = append(s.Failures, string(r.Payload))
		}
	}
	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(s); err != nil {
		return err
	}
	if s.DecodeErr != "" {
		return errors.New(s.DecodeErr)
	}
	return nil
}

// clone copies payload, which the decoder may reuse across records, and
// falls back to a JSON string when the payload is not itself valid JSON.
func clone(p []byte) json.RawMessage {
	if json.Valid(p) {
		return json.RawMessage(append([]byte(nil), p...))
	}
	q, _ := json.Marshal(string(p))
	return json.RawMessage(q)
}
