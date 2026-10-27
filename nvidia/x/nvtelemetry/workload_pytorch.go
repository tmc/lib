package nvtelemetry

import (
	"encoding/json"
	"errors"
	"io"
	"math"
)

// WorkloadAnnotation preserves a bounded PyTorch user annotation interval.
// Names are not parsed into steps, phases or ranks. Producer timestamp offsets
// retain the caller-declared trace clock; they are not NVML sensor timestamps.
type WorkloadAnnotation struct {
	Name          string
	RawStart      string
	RawDuration   string
	StartNanos    int64
	DurationNanos int64
	PID           json.RawMessage
	TID           json.RawMessage
	Args          json.RawMessage
}

// ReadPyTorchAnnotations reads complete CPU user_annotation events from PyTorch
// 2.7.0 and a fixed CPU-only subset of 2.10.0 Chrome exports. The 2.10.0 profile
// requires schemaVersion 1, present empty deviceProperties, and only
// user_annotation, cpu_op, Trace or uncategorized records. The version is a
// caller declaration checked for consistency, not producer authentication.
// Operator records remain in Export.Raw; this does not establish native GPU
// correlation or compiled member identity. The 2.7.0 raw GPU/flow behavior is
// unchanged.
// A bounded profile is not a continuous training or node-health stream.
func ReadPyTorchAnnotations(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, []WorkloadAnnotation, error) {
	if s.Provider != "pytorch" || s.Version != "2.7.0" && s.Version != "2.10.0" {
		return WorkloadExport{}, nil, errors.New("unsupported PyTorch version")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, nil, err
	}
	var trace struct {
		SchemaVersion    json.RawMessage `json:"schemaVersion"`
		DeviceProperties json.RawMessage `json:"deviceProperties"`
		TraceEvents      []struct {
			Name string             `json:"name"`
			Cat  string             `json:"cat"`
			Ph   string             `json:"ph"`
			TS   workloadJSONNumber `json:"ts"`
			Dur  workloadJSONNumber `json:"dur"`
			PID  json.RawMessage    `json:"pid"`
			TID  json.RawMessage    `json:"tid"`
			Args json.RawMessage    `json:"args"`
		} `json:"traceEvents"`
	}
	if err := decodeWorkloadJSON(e.Raw, &trace); err != nil {
		return e, nil, err
	}
	if s.Version == "2.10.0" && string(trace.SchemaVersion) != "1" {
		return e, nil, errors.New("unsupported PyTorch trace schema")
	}
	if s.Version == "2.10.0" {
		var devices []json.RawMessage
		if err := json.Unmarshal(trace.DeviceProperties, &devices); err != nil || devices == nil || len(devices) != 0 {
			return e, nil, errors.New("unsupported PyTorch CPU device properties")
		}
	}
	if trace.TraceEvents == nil || len(trace.TraceEvents) > l.MaxMetrics {
		return e, nil, errors.New("missing PyTorch trace events or limit exceeded")
	}
	var out []WorkloadAnnotation
	for _, v := range trace.TraceEvents {
		if s.Version == "2.10.0" {
			switch v.Cat {
			case "user_annotation", "cpu_op", "Trace", "":
			default:
				return e, nil, errors.New("unsupported PyTorch CPU trace category")
			}
		}
		if v.Cat != "user_annotation" {
			continue
		}
		if v.Ph != "X" {
			return e, nil, errors.New("unsupported PyTorch annotation phase")
		}
		if v.Name == "" || len(v.Name) > l.MaxStringBytes {
			return e, nil, errors.New("invalid PyTorch annotation name")
		}
		start, err := workloadTime(string(v.TS), 1000)
		if err != nil {
			return e, nil, err
		}
		dur, err := workloadTime(string(v.Dur), 1000)
		if err != nil {
			return e, nil, err
		}
		if dur < 0 || start > math.MaxInt64-dur {
			return e, nil, errors.New("invalid PyTorch annotation interval")
		}
		out = append(out, WorkloadAnnotation{Name: v.Name, RawStart: string(v.TS), RawDuration: string(v.Dur), StartNanos: start, DurationNanos: dur, PID: v.PID, TID: v.TID, Args: v.Args})
	}
	return e, out, nil
}
