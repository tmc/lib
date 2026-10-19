package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strconv"
)

// TrainingQualityPolicy retains supplied identities, not verified quality or
// convergence. Empty details remain unavailable. Numerical is accepted,
// rejected or unknown; Evidence names the independently owned oracle artifact.
// Tolerance and ConvergenceWindow are opaque policy identities, not numbers
// from which this package invents a numerical or convergence criterion.
type TrainingQualityPolicy struct {
	ID                   string `json:"id"`
	Model                string `json:"model"`
	Data                 string `json:"data"`
	Configuration        string `json:"configuration"`
	Seed                 string `json:"seed"`
	Precision            string `json:"precision"`
	LossScaling          string `json:"loss_scaling"`
	Optimizer            string `json:"optimizer"`
	LearningRateSchedule string `json:"learning_rate_schedule"`
	Batching             string `json:"batching"`
	Oracle               string `json:"oracle"`
	Tolerance            string `json:"tolerance"`
	Metric               string `json:"metric"`
	ConvergenceWindow    string `json:"convergence_window"`
	Numerical            string `json:"numerical"`
	Evidence             string `json:"evidence"`
}

// TrainingOperatorReference identifies an existing supplied operator group and
// its complete declared binding. Equality is agreement, not native observation.
// Missing native/compiler identities remain unknown in the consumer.
type TrainingOperatorReference struct {
	GroupID string          `json:"group_id"`
	Binding OperatorBinding `json:"binding"`
}

// TrainingPhase retains supplied phase identity. OptimizerStep and LoggingStep
// have different meanings. Multiple microbatches may name one optimizer update;
// phase rows are never counted as updates. No duration or additive wall time is
// derived. Evidence retains the raw application phase/clock/loss artifact.
// All integers, including zero, must be explicit on the wire.
type TrainingPhase struct {
	ID                string                      `json:"id"`
	Attempt           string                      `json:"attempt"`
	Restart           string                      `json:"restart"`
	Worker            string                      `json:"worker"`
	OptimizerStep     uint64                      `json:"optimizer_step"`
	LoggingStep       uint64                      `json:"logging_step"`
	Epoch             uint64                      `json:"epoch"`
	Batch             uint64                      `json:"batch"`
	Microbatch        uint64                      `json:"microbatch"`
	AccumulationStep  uint64                      `json:"accumulation_step"`
	AccumulationCount uint64                      `json:"accumulation_count"`
	Rank              uint32                      `json:"rank"`
	WorldSize         uint32                      `json:"world_size"`
	CommunicatorEpoch string                      `json:"communicator_epoch"`
	Phase             string                      `json:"phase"`
	QualityPolicy     string                      `json:"quality_policy"`
	Evidence          string                      `json:"evidence"`
	Operators         []TrainingOperatorReference `json:"operators"`
}

// TrainingPhaseContext is the separate version 2 training detail interchange.
// The existing version 1 optimizer and checkpoint interchanges stay unchanged.
// Policies and phases are bounded by WorkloadLimits; no provider SDK is used.
type TrainingPhaseContext struct {
	Version  int                     `json:"version"`
	RunID    string                  `json:"run_id"`
	Policies []TrainingQualityPolicy `json:"policies"`
	Phases   []TrainingPhase         `json:"phases"`
}

// ReadTrainingPhaseContext reads application training-context-v2 bytes, retaining
// their original digest. Duplicate/unknown fields, future versions, missing/null
// integers and noncanonical integer spellings are refused.
func ReadTrainingPhaseContext(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, TrainingPhaseContext, error) {
	var c TrainingPhaseContext
	if s.Provider != "application" || s.Version != "training-context-v2" {
		return WorkloadExport{}, c, errors.New("training phase context requires application training-context-v2 source")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, c, err
	}
	if err = decodeWorkloadJSON(e.Raw, &c); err != nil {
		return e, c, err
	}
	d := json.NewDecoder(bytes.NewReader(e.Raw))
	d.DisallowUnknownFields()
	if err = d.Decode(&c); err != nil {
		return e, c, fmt.Errorf("training phase context: %w", err)
	}
	if err = trainingIntegerPresence(e.Raw); err != nil {
		return e, c, err
	}
	if err = validateTrainingPhases(c, s.RunID, l); err != nil {
		return e, c, err
	}
	return e, c, nil
}

func trainingIntegerPresence(raw []byte) error {
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return err
	}
	check := func(m map[string]json.RawMessage, names ...string) error {
		for _, name := range names {
			v := m[name]
			n, err := strconv.ParseUint(string(v), 10, 64)
			if err != nil || strconv.FormatUint(n, 10) != string(v) {
				return fmt.Errorf("missing or noncanonical training integer %s", name)
			}
		}
		return nil
	}
	if err := check(body, "version"); err != nil {
		return err
	}
	var rows []map[string]json.RawMessage
	if err := json.Unmarshal(body["phases"], &rows); err != nil {
		return err
	}
	for _, row := range rows {
		if err := check(row, "optimizer_step", "logging_step", "epoch", "batch", "microbatch", "accumulation_step", "accumulation_count", "rank", "world_size"); err != nil {
			return err
		}
	}
	return nil
}

// Check string, collection and encoded-size upper bounds before marshaling a
// caller-created value. These fixed structs have no cycles or arbitrary maps.
func trainingPhaseBudget(c TrainingPhaseContext, l WorkloadLimits) error {
	remaining := l.MaxBytes
	var walk func(reflect.Value) error
	walk = func(v reflect.Value) error {
		var charge int64 = 24
		switch v.Kind() {
		case reflect.Pointer:
			if v.IsNil() {
				break
			}
			return walk(v.Elem())
		case reflect.String:
			if v.Len() > l.MaxStringBytes {
				return errors.New("training phase string limit exceeded")
			}
			charge = int64(v.Len())*6 + 3
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				remaining -= int64(len(v.Type().Field(i).Name))*6 + 4
				if remaining < 0 {
					return errors.New("training phase byte limit exceeded")
				}
				if err := walk(v.Field(i)); err != nil {
					return err
				}
			}
		case reflect.Slice:
			if v.Len() > l.MaxMetrics {
				return errors.New("training phase collection limit exceeded")
			}
			for i := 0; i < v.Len(); i++ {
				if err := walk(v.Index(i)); err != nil {
					return err
				}
			}
		}
		if charge > remaining {
			return errors.New("training phase byte limit exceeded")
		}
		remaining -= charge
		return nil
	}
	return walk(reflect.ValueOf(c))
}

func validateTrainingPhases(c TrainingPhaseContext, run string, l WorkloadLimits) error {
	if c.Version != 2 || c.RunID != run || !checkpointText(c.RunID) || c.Phases == nil || c.Policies == nil {
		return errors.New("invalid training phase version, run or collections")
	}
	if err := trainingPhaseBudget(c, l); err != nil {
		return err
	}
	policies := map[string]bool{}
	for _, p := range c.Policies {
		if !checkpointText(p.ID) || policies[p.ID] {
			return errors.New("invalid or duplicate training policy")
		}
		policies[p.ID] = true
		switch p.Numerical {
		case "accepted", "rejected":
			if p.Oracle == "" || p.Evidence == "" || p.Tolerance == "" {
				return errors.New("missing numerical oracle, tolerance or evidence")
			}
		case "unknown":
		default:
			return errors.New("unsupported training numerical status")
		}
	}
	seen := map[string]bool{}
	type updateKey struct {
		attempt string
		step    uint64
		rank    uint32
	}
	type identity struct {
		restart, worker, comm, policy string
		world                         uint32
		accum                         uint64
	}
	ids := map[updateKey]identity{}
	references := 0
	for _, p := range c.Phases {
		if !checkpointText(p.ID) || seen[p.ID] || !checkpointText(p.Attempt) || !checkpointText(p.Restart) || !checkpointText(p.Worker) || !checkpointText(p.Evidence) || p.WorldSize == 0 || uint64(p.WorldSize) > uint64(l.MaxMetrics) || p.Rank >= p.WorldSize || p.AccumulationCount == 0 || p.AccumulationStep >= p.AccumulationCount {
			return errors.New("invalid training phase identity or accumulation")
		}
		seen[p.ID] = true
		switch p.Phase {
		case "data_wait", "forward", "backward", "optimizer", "communication", "checkpoint_save", "checkpoint_restore", "recompute":
		default:
			return errors.New("unsupported training phase")
		}
		if p.QualityPolicy != "" && !policies[p.QualityPolicy] {
			return errors.New("training phase policy unavailable")
		}
		k := updateKey{p.Attempt, p.OptimizerStep, p.Rank}
		id := identity{p.Restart, p.Worker, p.CommunicatorEpoch, p.QualityPolicy, p.WorldSize, p.AccumulationCount}
		if old, ok := ids[k]; ok && old != id {
			return errors.New("contradictory training update context")
		}
		ids[k] = id
		if len(p.Operators) > l.MaxMetrics-references {
			return errors.New("training operator reference limit exceeded")
		}
		references += len(p.Operators)
		groups := map[string]bool{}
		for _, r := range p.Operators {
			if r.GroupID == "" || groups[r.GroupID] {
				return errors.New("invalid or duplicate training operator reference")
			}
			groups[r.GroupID] = true
			if r.Binding.Code != nil {
				if err := r.Binding.Code.Validate(); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// WriteTrainingPhaseContext validates all bounds before writing version 2. It
// creates no quality evidence or optimizer outcomes and does not mutate c.
func WriteTrainingPhaseContext(w io.Writer, c TrainingPhaseContext, l WorkloadLimits) error {
	l, err := workloadLimits(l)
	if err != nil {
		return err
	}
	if w == nil {
		return errors.New("missing training phase writer")
	}
	if err = validateTrainingPhases(c, c.RunID, l); err != nil {
		return err
	}
	raw, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("training phase byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		return io.ErrShortWrite
	}
	return err
}
