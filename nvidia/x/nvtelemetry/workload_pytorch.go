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

// ReadPyTorchAnnotations reads PyTorch 2.7.0 Chrome trace user_annotation
// complete events. CPU/GPU operators and flow records remain in Export.Raw.
// This bounded profile is not a continuous training or node-health stream.
func ReadPyTorchAnnotations(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, []WorkloadAnnotation, error) {
	if s.Provider != "pytorch" || s.Version != "2.7.0" {
		return WorkloadExport{}, nil, errors.New("unsupported PyTorch version")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, nil, err
	}
	var trace struct {
		TraceEvents []struct {
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
	if trace.TraceEvents == nil || len(trace.TraceEvents) > l.MaxMetrics {
		return e, nil, errors.New("missing PyTorch trace events or limit exceeded")
	}
	var out []WorkloadAnnotation
	for _, v := range trace.TraceEvents {
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
