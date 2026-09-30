package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/counters"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
)

const counterReceiptLimit = 128 << 20

type counterShapeArm struct {
	Path            string         `json:"path"`
	SHA256          string         `json:"sha256"`
	Completeness    string         `json:"completeness"`
	Stationarity    iteration.Gate `json:"stationarity"`
	Markers         int            `json:"inferred_iteration_markers"`
	UnpairedMarkers int            `json:"unpaired_nvtx_markers"`
}
type counterShapeSource struct {
	Version         int               `json:"version"`
	Domain          string            `json:"domain"`
	IterationMarker string            `json:"inferred_iteration_marker"`
	Arms            []counterShapeArm `json:"arms"`
	Shape           shape.Report      `json:"shape_qualification"`
	Limits          string            `json:"limits"`
}

type counterSelectionReport struct {
	Version            int                 `json:"version"`
	ShapeFileSHA256    string              `json:"shape_file_sha256"`
	IdentityFileSHA256 string              `json:"identity_file_sha256"`
	Source             counterShapeSource  `json:"source_shape_comparison"`
	Selection          *counters.Selection `json:"selection,omitempty"`
	Failure            string              `json:"failure,omitempty"`
	Limits             string              `json:"limits"`
}

func runCounterReceiptCommand(args []string) error { return runCounterReceipt(args, os.Stdout) }
func runCounterReceipt(args []string, out io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: cudaprof counter-receipt select -shape FILE -identity FILE | compare BASE TARGET")
	}
	switch args[0] {
	case "-h", "-help", "--help":
		fmt.Fprintln(out, "Usage: cudaprof counter-receipt select -shape FILE -identity FILE | compare BASE TARGET")
		return nil
	case "select":
		return runCounterSelect(args[1:], out)
	case "compare":
		return runCounterCompare(args[1:], out)
	default:
		return fmt.Errorf("counter-receipt: unknown operation %q", args[0])
	}
}

func runCounterSelect(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("counter-receipt select", flag.ContinueOnError)
	fs.SetOutput(out)
	fs.Usage = func() {
		fmt.Fprintln(out, "Usage: cudaprof counter-receipt select -shape FILE -identity FILE")
		fs.PrintDefaults()
	}
	shapePath := fs.String("shape", "", "shape-compare JSON receipt")
	identityPath := fs.String("identity", "", "exact shape.Identity JSON")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 0 || *shapePath == "" || *identityPath == "" {
		return fmt.Errorf("usage: cudaprof counter-receipt select -shape FILE -identity FILE")
	}
	shapeData, shapeSHA, err := readCounterJSONFile(*shapePath)
	if err != nil {
		return err
	}
	identityData, identitySHA, err := readCounterJSONFile(*identityPath)
	if err != nil {
		return err
	}
	report := counterSelectionReport{Version: 1, ShapeFileSHA256: shapeSHA, IdentityFileSHA256: identitySHA, Limits: "offline supplied receipts are structurally checked, not revalidated against their native captures; selection names historical qualified launches, not replay instructions or collected counters; stationarity, correctness and profiler overhead remain separate"}
	if err = decodeCounterJSON(shapeData, &report.Source); err != nil {
		return fmt.Errorf("counter-receipt: shape receipt: %w", err)
	}
	var identity shape.Identity
	if err = decodeCounterIdentity(identityData, &identity); err != nil {
		return fmt.Errorf("counter-receipt: identity: %w", err)
	}
	var selection counters.Selection
	switch {
	case report.Source.Version != 1 || len(report.Source.Arms) != 2 || (report.Source.Domain != "driver" && report.Source.Domain != "runtime"):
		err = fmt.Errorf("counter-receipt: invalid shape-comparison envelope")
	default:
		for _, arm := range report.Source.Arms {
			if arm.Completeness != "passed" || !counterReceiptSHA(arm.SHA256) {
				err = fmt.Errorf("counter-receipt: capture completeness or identity unavailable")
				break
			}
		}
		if err == nil {
			selection, err = counters.Select(report.Source.Shape, identity)
		}
	}
	if err != nil {
		report.Failure = err.Error()
	} else {
		report.Selection = &selection
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return errors.Join(err, enc.Encode(report))
}

func runCounterCompare(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("counter-receipt compare", flag.ContinueOnError)
	fs.SetOutput(out)
	fs.Usage = func() { fmt.Fprintln(out, "Usage: cudaprof counter-receipt compare BASE.json TARGET.json") }
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("usage: cudaprof counter-receipt compare BASE.json TARGET.json")
	}
	var receipts [2]counters.Receipt
	var hashes [2]string
	for i, path := range fs.Args() {
		data, sha, err := readCounterJSONFile(path)
		if err != nil {
			return err
		}
		hashes[i] = sha
		if err = decodeCounterJSON(data, &receipts[i]); err != nil {
			return fmt.Errorf("counter-receipt %s: %w", path, err)
		}
		var fields map[string]json.RawMessage
		if err = json.Unmarshal(data, &fields); err != nil {
			return err
		}
		if err = decodeCounterIdentity(fields["identity"], &receipts[i].Identity); err != nil {
			return fmt.Errorf("counter-receipt %s identity: %w", path, err)
		}
	}
	comparison := counters.Compare(receipts[0], receipts[1])
	report := struct {
		Version      int                 `json:"version"`
		SourceSHA256 [2]string           `json:"receipt_file_sha256"`
		Comparison   counters.Comparison `json:"comparison"`
		Limits       string              `json:"limits"`
	}{1, hashes, comparison, "offline receipt compatibility is not native collection verification, independent workload correctness, timing comparison, or causal attribution"}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	var err error
	if !comparison.Compatible {
		err = fmt.Errorf("counter-receipt: incompatible receipts: %s", comparison.Reason)
	}
	return errors.Join(err, enc.Encode(report))
}

func readCounterJSONFile(path string) ([]byte, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", fmt.Errorf("counter-receipt: read %s: %w", path, err)
	}
	data, readErr := io.ReadAll(io.LimitReader(f, counterReceiptLimit+1))
	err = errors.Join(readErr, f.Close())
	if err != nil {
		return nil, "", err
	}
	if len(data) > counterReceiptLimit {
		return nil, "", fmt.Errorf("counter-receipt: JSON file exceeds %d bytes", counterReceiptLimit)
	}
	sum := sha256.Sum256(data)
	return data, hex.EncodeToString(sum[:]), nil
}
func decodeCounterJSON(data []byte, dst any) error {
	if err := checkCounterJSONKeys(json.NewDecoder(bytes.NewReader(data))); err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		if err != nil {
			return err
		}
		return fmt.Errorf("trailing JSON value")
	}
	return nil
}
func decodeCounterIdentity(data []byte, id *shape.Identity) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for _, key := range []string{"phase", "token", "operation", "dtype", "cache_length", "grid", "block"} {
		if len(fields[key]) == 0 || bytes.Equal(bytes.TrimSpace(fields[key]), []byte("null")) {
			return fmt.Errorf("missing required identity field %q", key)
		}
	}
	return decodeCounterJSON(data, id)
}
func counterReceiptSHA(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// JSON's last-key-wins behavior cannot qualify an identity unambiguously.
func checkCounterJSONKeys(dec *json.Decoder) error {
	token, err := dec.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		seen := map[string]bool{}
		for dec.More() {
			key, err := dec.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			normalized := strings.ToLower(name)
			if !ok || seen[normalized] {
				return fmt.Errorf("duplicate or invalid JSON key %q", name)
			}
			seen[normalized] = true
			if err := checkCounterJSONKeys(dec); err != nil {
				return err
			}
		}
	case '[':
		for dec.More() {
			if err := checkCounterJSONKeys(dec); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unexpected JSON delimiter")
	}
	_, err = dec.Token()
	return err
}
