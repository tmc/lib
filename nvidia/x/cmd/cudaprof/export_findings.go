package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// exportRecordFindings separates retained artifact records from whole collector
// coverage. A retained count alone never establishes a workload count.
func exportRecordFindings(journals map[string][]byte, identity traceExportIdentity, status captureResult, maxRecords, journalBytes int64) ([]tracecontract.Finding, error) {
	records, _, _, err := decodeTraceJournals(journals, maxRecords, journalBytes, false)
	if err != nil {
		return nil, err
	}
	receipt := exportReceipt("trace export findings", identity, records, status)
	if _, err := receipt.JSON(); err != nil {
		return nil, err
	}
	receipt.Gates["journal_decode"] = tracecontract.Gate{Disposition: "passed", Reason: "bounded retained journals finalized without sequence loss", Provenance: "raw journal decoder"}
	var activities, kernels uint64
	projectionOK := true
	names := make([]string, 0, len(journals))
	for name := range journals {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		d := wire.NewDecoder(bytes.NewReader(journals[name]), wire.Limits{})
		for {
			record, err := d.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				return nil, fmt.Errorf("finding journal %s: %w", name, err)
			}
			if record.Kind != wire.KindActivity {
				continue
			}
			activities++
			if err := findingPayloadKeys(record.Payload); err != nil {
				projectionOK = false
				continue
			}
			var p tracePayload
			if err := json.Unmarshal(record.Payload, &p); err != nil {
				return nil, fmt.Errorf("finding payload: %w", err)
			}
			switch p.Subtype {
			case "kernel", "concurrent_kernel":
				if !p.Valid || p.Start == 0 || p.End < p.Start {
					projectionOK = false
					continue
				}
				kernels++
			case "memcpy", "memset", "sync", "graph_trace", "overhead", "marker", "capture_range":
			default:
				// Raw projection defaults to kernel; findings require an explicit subtype.
				projectionOK = false
			}
		}
	}
	if uint64(receipt.ProjectedRows["kernel"]) != kernels {
		projectionOK = false
	}
	receipt.Gates["kernel_projection"] = tracecontract.Gate{Disposition: "failed", Reason: "explicit kernel population does not match decoded projection", Provenance: "independent raw activity count"}
	if projectionOK {
		receipt.Gates["kernel_projection"] = tracecontract.Gate{Disposition: "passed", Reason: "explicit raw kernel records agree with decoded rows", Provenance: "independent raw activity count"}
	}
	receipt.Gates["activity_reconciliation"] = tracecontract.Gate{Disposition: "unavailable", Reason: "supported count or explicit loss assessment is absent", Provenance: "capture-status.json and raw frames"}
	// Nil is unknown. Legacy captures cannot distinguish omitted zero from an
	// unreported producer breakdown; an explicit zero object remains known.
	receipt.LostRecords = map[string]tracecontract.Loss{"activity": {Unknown: true}}
	if status.Protocol == captureProtocolVersion && status.RecordLoss != nil {
		loss := status.RecordLoss
		valid := loss.Offered >= loss.Retained && loss.Retained >= activities
		if valid {
			unknown := loss.Native != 0 || loss.Queue != 0 || loss.QueueOversize != 0 || loss.QueueContention != 0 || loss.QueueCapacity != 0 || status.DroppedRecords != 0
			receipt.LostRecords["activity"] = tracecontract.Loss{Dropped: loss.Offered - loss.Retained, Total: loss.Offered, Unknown: unknown}
		}
		receipt.Gates["activity_reconciliation"] = tracecontract.Gate{Disposition: "failed", Reason: "raw count, loss totals or finalization disagree with collector status", Provenance: "capture-status.json and raw frames"}
		if valid && activities == status.ActivityRecords && runCompletenessOf(status.TransportReady, status.CollectorFinalized, status.UnqualifiedGenerations, status.DroppedRecords, activities) == runComplete {
			receipt.Gates["activity_reconciliation"] = tracecontract.Gate{Disposition: "passed", Reason: "all raw activity frames agree with collector count", Provenance: "capture-status.json and raw frames"}
		}
	}
	var value *float64
	if projectionOK {
		count := float64(kernels) // bounded journals contain far fewer than 2^53 records.
		value = &count
	}
	ev := tracecontract.EvidenceContext{Receipt: receipt, Artifacts: []string{identity.sha256}}
	retained := []tracecontract.Requirement{{Name: "decoded_journals", ReceiptGate: "journal_decode", Backend: "native", ArtifactDigest: identity.sha256}, {Name: "kernel_records", ReceiptGate: "kernel_projection"}}
	whole := append(append([]tracecontract.Requirement(nil), retained...), tracecontract.Requirement{Name: "collector_complete", ReceiptGate: "collector"}, tracecontract.Requirement{Name: "raw_population", ReceiptGate: "activity_reconciliation", Family: "activity"})
	return []tracecontract.Finding{
		tracecontract.Admit("retained_kernel_records", "retained_kernel_records", tracecontract.FindingObserved, "kernel records retained in this artifact; not total workload launches", value, "records", retained, ev),
		tracecontract.Admit("whole_capture_kernel_records", "whole_capture_kernel_records", tracecontract.FindingObserved, "kernel records in the collector capture population; not total workload launches", value, "records", whole, ev),
	}, nil
}

// Canonical tags come from the existing payload contract. Opaque metadata keys
// are not Go field names and retain their own case-sensitive namespaces.
var findingPayloadFields = func() map[string]bool {
	keys := make(map[string]bool)
	typ := reflect.TypeOf(tracePayload{})
	for i := 0; i < typ.NumField(); i++ {
		key := strings.Split(typ.Field(i).Tag.Get("json"), ",")[0]
		if key != "" && key != "-" {
			keys[key] = true
		}
	}
	return keys
}()

// findingPayloadKeys rejects ambiguous JSON without changing raw projection.
// Exact duplicates are refused at every object level. Only top-level known
// payload fields are subject to canonical spelling; opaque nested keys are not.
func findingPayloadKeys(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value func(int) error
	value = func(depth int) error {
		if depth > 64 {
			return fmt.Errorf("finding payload: nesting limit exceeded")
		}
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delim, compound := token.(json.Delim)
		if !compound {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]bool)
			for decoder.More() {
				token, err := decoder.Token()
				if err != nil {
					return err
				}
				key, ok := token.(string)
				if !ok {
					return fmt.Errorf("finding payload: invalid object key")
				}
				if seen[key] {
					return fmt.Errorf("finding payload: duplicate key")
				}
				seen[key] = true
				if depth == 0 && !findingPayloadFields[key] {
					for canonical := range findingPayloadFields {
						if strings.EqualFold(key, canonical) {
							return fmt.Errorf("finding payload: noncanonical field spelling")
						}
					}
				}
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		case '[':
			for decoder.More() {
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("finding payload: unexpected delimiter")
		}
		_, err = decoder.Token()
		return err
	}
	if err := value(0); err != nil {
		return err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return fmt.Errorf("finding payload: trailing JSON")
	}
	return nil
}

func writeRecordFindings(w io.Writer, findings []tracecontract.Finding, identity traceExportIdentity, id uint64) error {
	if w == nil {
		return fmt.Errorf("findings: nil output")
	}
	if len(findings) != 2 {
		return fmt.Errorf("findings: unexpected rule count")
	}
	pw := perfetto.NewWriter(w)
	track := tracecontract.Track{UUID: id, Name: "raw-record diagnostic findings", Role: "findings", Clock: identity.sha256 + ":export-relative", CaptureSHA256: identity.sha256, Provenance: "evaluated raw-record requirements; withheld findings retained", Aggregation: "none"}
	if err := track.Write(pw, 0); err != nil {
		return err
	}
	return tracecontract.WriteFindings(pw, id, 0, findings)
}
