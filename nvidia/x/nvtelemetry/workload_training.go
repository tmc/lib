package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

type workloadJSONNumber string

func (n *workloadJSONNumber) UnmarshalJSON(raw []byte) error {
	if bytes.Equal(raw, []byte("null")) {
		*n = ""
		return nil
	}
	if len(raw) == 0 || (raw[0] != '-' && (raw[0] < '0' || raw[0] > '9')) {
		return errors.New("workload field must be a JSON number")
	}
	var v json.Number
	if err := json.Unmarshal(raw, &v); err != nil {
		return err
	}
	if f, err := v.Float64(); err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
		return errors.New("nonfinite workload JSON number")
	}
	*n = workloadJSONNumber(v)
	return nil
}

func decodeWorkloadJSON(raw []byte, dst any) error {
	// encoding/json normally accepts duplicate object keys. That would erase
	// contradictory producer identities, timestamps or metric values.
	check := json.NewDecoder(bytes.NewReader(raw))
	check.UseNumber()
	if err := checkWorkloadJSON(check, 0); err != nil {
		return err
	}
	if _, err := check.Token(); err != io.EOF {
		return errors.New("trailing workload JSON")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := d.Decode(dst); err != nil {
		return fmt.Errorf("decode workload JSON: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return errors.New("trailing workload JSON")
	}
	return nil
}

func checkWorkloadJSON(d *json.Decoder, depth int) error {
	if depth > 128 {
		return errors.New("workload JSON nesting limit exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			k, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := k.(string)
			if !ok || seen[key] {
				return errors.New("duplicate workload JSON key")
			}
			seen[key] = true
			if err := checkWorkloadJSON(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := checkWorkloadJSON(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("invalid workload JSON delimiter")
	}
	_, err = d.Token()
	return err
}

func trainingMetric(e WorkloadExport, name, value, timestamp string, timeScale int64, step *int64, labels map[string]string, l WorkloadLimits) (WorkloadMetric, error) {
	p := WorkloadMetric{Source: e.Source, Name: name, Semantic: "user_defined_training_metric", Unit: e.Source.Units[name], Kind: "gauge", Temporality: "instantaneous", RawValue: value, RawTime: timestamp, Labels: labels}
	if strings.TrimSpace(name) == "" || len(name) > l.MaxStringBytes || len(value) > l.MaxStringBytes {
		return p, errors.New("invalid training metric name or value length")
	}
	if _, err := workloadNumber(value); err != nil {
		return p, err
	}
	if p.Unit == "" {
		p.Unit = "unknown"
	}
	if timestamp != "" {
		t, err := workloadTime(timestamp, timeScale)
		if err != nil {
			return p, err
		}
		p.TimeNanos = t
		p.HasTime = true
		p.TimeSource = "producer"
		if timeScale == 1e6 {
			p.RawTimeUnit = "milliseconds"
		} else if timeScale == 1e9 {
			p.RawTimeUnit = "seconds"
		}
	}
	if step != nil {
		if *step < 0 {
			return p, errors.New("negative training step")
		}
		p.Step = *step
		p.HasStep = true
		switch e.Source.Provider {
		case "mlflow":
			p.StepKind = "logged_metric_step"
		case "wandb":
			p.StepKind = "history_step"
		case "clearml":
			p.StepKind = "reported_iteration"
		}
	}
	return p, nil
}

// ReadMLflowHistory reads MLflow 2.22.0 /api/2.0/mlflow/metrics/get-history
// response JSON. RunID is required because this response does not contain it.
// Pagination is retained in NextPageToken; one page is not a complete run.
func ReadMLflowHistory(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, error) {
	if s.Provider != "mlflow" || s.Version != "2.22.0" {
		return WorkloadExport{}, errors.New("unsupported MLflow version")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, err
	}
	var body struct {
		Metrics []struct {
			Key       string             `json:"key"`
			Value     workloadJSONNumber `json:"value"`
			Timestamp workloadJSONNumber `json:"timestamp"`
			Step      *int64             `json:"step"`
		} `json:"metrics"`
		NextPageToken string `json:"next_page_token"`
	}
	if err := decodeWorkloadJSON(e.Raw, &body); err != nil {
		return e, err
	}
	if body.Metrics == nil || len(body.Metrics) > l.MaxMetrics || len(body.NextPageToken) > l.MaxStringBytes {
		return e, errors.New("missing MLflow metrics or limit exceeded")
	}
	e.NextPageToken = body.NextPageToken
	for _, v := range body.Metrics {
		// The proto defaults absent step to zero. Preserve absence instead: it
		// cannot prove an explicitly logged training step.
		p, err := trainingMetric(e, v.Key, string(v.Value), string(v.Timestamp), 1e6, v.Step, nil, l)
		if err != nil {
			return e, err
		}
		e.Metrics = append(e.Metrics, p)
	}
	return e, nil
}

// ReadWandBHistory reads W&B 0.19.11 wandb-history.jsonl scalar rows.
// Nested/media records remain in Raw and are not interpreted as scalar metrics.
// Reserved _timestamp (seconds) and _step fields retain their presence.
func ReadWandBHistory(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, error) {
	if s.Provider != "wandb" || s.Version != "0.19.11" {
		return WorkloadExport{}, errors.New("unsupported W&B version")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, err
	}
	for _, line := range bytes.Split(e.Raw, []byte{'\n'}) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var row map[string]json.RawMessage
		if err := decodeWorkloadJSON(line, &row); err != nil {
			return e, err
		}
		var timestamp workloadJSONNumber
		var step *int64
		if raw, ok := row["_timestamp"]; ok {
			if err := json.Unmarshal(raw, &timestamp); err != nil {
				return e, err
			}
		}
		if raw, ok := row["_step"]; ok {
			if err := json.Unmarshal(raw, &step); err != nil {
				return e, err
			}
		}
		var keys []string
		for key := range row {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if strings.HasPrefix(key, "_") {
				continue
			}
			var value workloadJSONNumber
			if err := json.Unmarshal(row[key], &value); err != nil || value == "" {
				continue
			}
			p, err := trainingMetric(e, key, string(value), string(timestamp), 1e9, step, nil, l)
			if err != nil {
				return e, err
			}
			e.Metrics = append(e.Metrics, p)
			if len(e.Metrics) > l.MaxMetrics {
				return e, errors.New("workload metric limit exceeded")
			}
		}
	}
	return e, nil
}

// ReadClearMLScalars reads the response body of ClearML 1.18.0 API v2.9
// events.get_task_events. Only training_stats_scalar events are normalized;
// other event kinds remain in Raw. Timestamp units are milliseconds.
func ReadClearMLScalars(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, error) {
	if s.Provider != "clearml" || s.Version != "1.18.0/api-2.9" {
		return WorkloadExport{}, errors.New("unsupported ClearML version")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, err
	}
	// The SDK exposes response_data directly; an HTTP response wraps it in
	// data and meta. Accept both documented representations, preserving the
	// entire original envelope in Raw.
	var envelope struct {
		Data json.RawMessage `json:"data"`
		Meta struct {
			ResultCode int `json:"result_code"`
		} `json:"meta"`
	}
	if err := decodeWorkloadJSON(e.Raw, &envelope); err != nil {
		return e, err
	}
	payload := e.Raw
	if envelope.Data != nil {
		if envelope.Meta.ResultCode != 200 {
			return e, errors.New("ClearML response is not successful")
		}
		payload = envelope.Data
	}
	var body struct {
		Events []struct {
			Type      string             `json:"type"`
			Task      string             `json:"task"`
			Metric    string             `json:"metric"`
			Variant   string             `json:"variant"`
			Value     workloadJSONNumber `json:"value"`
			Timestamp workloadJSONNumber `json:"timestamp"`
			Iter      *int64             `json:"iter"`
		} `json:"events"`
		ScrollID string `json:"scroll_id"`
	}
	if err := decodeWorkloadJSON(payload, &body); err != nil {
		return e, err
	}
	if body.Events == nil || len(body.Events) > l.MaxMetrics {
		return e, errors.New("missing ClearML events or limit exceeded")
	}
	if len(body.ScrollID) > l.MaxStringBytes {
		return e, errors.New("ClearML scroll token limit exceeded")
	}
	e.NextPageToken = body.ScrollID
	for _, v := range body.Events {
		if v.Type != "training_stats_scalar" {
			continue
		}
		if v.Task != s.RunID || len(v.Variant) > l.MaxStringBytes {
			return e, errors.New("ClearML task mismatch or variant limit exceeded")
		}
		p, err := trainingMetric(e, v.Metric, string(v.Value), string(v.Timestamp), 1e6, v.Iter, map[string]string{"variant": v.Variant}, l)
		if err != nil {
			return e, err
		}
		e.Metrics = append(e.Metrics, p)
	}
	return e, nil
}
