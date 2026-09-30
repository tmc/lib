package tracecontract

import (
	"encoding/json"
	"fmt"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// Gate preserves one independent evidence disposition. Disposition is passed,
// failed, unavailable, unchecked or unsupported. An unavailable gate must not
// be promoted by another gate's success.
type Gate struct {
	Disposition string `json:"disposition"`
	Reason      string `json:"reason"`
	Provenance  string `json:"provenance"`
}

// Backend describes a backend actually used. Version and SHA256 remain empty
// when unavailable; native processing does not claim an external importer ran.
type Backend struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
	SHA256  string `json:"sha256,omitempty"`
}

// Receipt is bounded trace-analysis metadata. Gates must contain collector,
// import, workload, stationarity, shape and correctness independently. InputSHA256
// hashes the exact input bytes parsed; Kind is capture, pftrace or comparison_archive.
// Schema and tool versions are retained separately from evidence dispositions.
// The zero value is not a receipt; Version must be 1 and identities supplied.
type Receipt struct {
	Version                  int               `json:"version"`
	Command                  string            `json:"command"`
	Options                  map[string]string `json:"options"`
	InputPath                string            `json:"input"`
	InputSHA256              string            `json:"input_sha256"`
	Kind                     string            `json:"detected_type"`
	ProducerVersion          string            `json:"producer_version,omitempty"`
	ConsumerVersion          string            `json:"consumer_version,omitempty"`
	ProducerSchema           string            `json:"producer_schema,omitempty"`
	Backends                 []Backend         `json:"backends"`
	Gates                    map[string]Gate   `json:"gates"`
	Warnings                 []string          `json:"warnings"`
	Limits                   map[string]int64  `json:"limits"`
	Partial                  bool              `json:"partial"`
	ProjectedRows            map[string]int64  `json:"projected_rows_by_kind,omitempty"`
	RawDeviceRecords         map[string]int64  `json:"raw_device_records_by_kind,omitempty"`
	CollectorActivityRecords *uint64           `json:"collector_activity_records,omitempty"`
}

// JSON validates the receipt and returns deterministic versioned JSON. It
// refuses metadata exceeding 64 KiB; large evidence belongs in source records.
func (r Receipt) JSON() ([]byte, error) {
	if len(r.Options) > 128 || len(r.Backends) > 64 || len(r.Gates) > 32 || len(r.Warnings) > 256 || len(r.Limits) > 128 {
		return nil, fmt.Errorf("trace receipt: metadata collection limit exceeded")
	}
	if r.Version != 1 || r.Command == "" || r.InputPath == "" || len(r.Backends) == 0 || !validHash(r.InputSHA256) {
		return nil, fmt.Errorf("trace receipt: missing identity or unsupported version")
	}
	switch r.Kind {
	case "capture", "pftrace", "comparison_archive":
	default:
		return nil, fmt.Errorf("trace receipt: unsupported input type %q", r.Kind)
	}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		_, ok := r.Gates[name]
		if !ok {
			return nil, fmt.Errorf("trace receipt: missing %s evidence disposition", name)
		}
	}
	for name, gate := range r.Gates {
		if name == "" || gate.Reason == "" || gate.Provenance == "" {
			return nil, fmt.Errorf("trace receipt: incomplete evidence disposition")
		}
		switch gate.Disposition {
		case "passed", "failed", "unavailable", "unchecked", "unsupported":
		default:
			return nil, fmt.Errorf("trace receipt: invalid %s disposition", name)
		}
	}
	for _, backend := range r.Backends {
		if backend.Name == "" || (backend.SHA256 != "" && !validHash(backend.SHA256)) {
			return nil, fmt.Errorf("trace receipt: invalid backend identity")
		}
	}
	for _, limit := range r.Limits {
		if limit < 0 {
			return nil, fmt.Errorf("trace receipt: negative resource limit")
		}
	}
	data, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if len(data) > 64<<10 {
		return nil, fmt.Errorf("trace receipt: metadata exceeds 65536 bytes")
	}
	return data, nil
}

// Write emits one official TrackEvent instant on a caller-registered receipt
// track. cudaprof.receipt_json retains the complete receipt; individual gate
// dispositions are also directly available through Perfetto's args table.
func (r Receipt) Write(writer *perfetto.Writer, uuid, timestamp uint64) error {
	if writer == nil || uuid == 0 {
		return fmt.Errorf("trace receipt: missing writer or track")
	}
	data, err := r.JSON()
	if err != nil {
		return err
	}
	annotations := []perfetto.DebugAnnotation{text("cudaprof_schema", "cudaprof.receipt.v1"), number("cudaprof_schema_version", 1), text("cudaprof_receipt_json", string(data)), text("cudaprof_input_sha256", r.InputSHA256), text("cudaprof_command", r.Command)}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		annotations = append(annotations, text("cudaprof_gate_"+name, r.Gates[name].Disposition))
	}
	return writer.WriteFlowInstant(uuid, timestamp, "cudaprof.receipt.v1", "cudaprof.semantic", annotations, nil, nil)
}
