package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

const maxReplayInputBytes = 64 << 20

type observedEnvironmentFile struct {
	Version     int                     `json:"version"`
	Environment cudaregress.Environment `json:"environment"`
	Provenance  string                  `json:"provenance"`
}

type historyBindingFile struct {
	Version               int                           `json:"version"`
	PlanFrozenDigest      string                        `json:"plan_frozen_digest"`
	HistoryTemplateDigest string                        `json:"history_template_digest"`
	PlanWorkload          cudaworkload.WorkloadIdentity `json:"plan_workload"`
	HistoryWorkload       cudacache.WorkloadIdentity    `json:"history_workload"`
	Provenance            string                        `json:"provenance"`
}

type replayCommandReceipt struct {
	Version               int                            `json:"version"`
	PlanID                string                         `json:"plan_id"`
	PlanDigest            string                         `json:"plan_digest"`
	SpecDigest            string                         `json:"spec_digest"`
	Mode                  string                         `json:"mode"`
	ArtifactClosure       []replayArtifactReceipt        `json:"artifact_closure"`
	EnvironmentObserved   cudaregress.Environment        `json:"environment_observed"`
	EnvironmentProvenance string                         `json:"environment_provenance"`
	LockOwnerPID          int                            `json:"lock_owner_pid,omitempty"`
	Execution             *cudaregress.ExecutionReceipt  `json:"execution,omitempty"`
	Comparison            *cudaregress.ComparisonReceipt `json:"comparison,omitempty"`
	Attempts              []cudaregress.CommandAttempt   `json:"attempts"`
	HistoryRecordKey      string                         `json:"history_record_key,omitempty"`
	HistoryTemplateDigest string                         `json:"history_template_digest,omitempty"`
	HistoryBindingDigest  string                         `json:"history_binding_digest,omitempty"`
	HistoryBindingSource  string                         `json:"history_binding_provenance,omitempty"`
	HistoryWithheldReason string                         `json:"history_withheld_reason,omitempty"`
	ReplayError           string                         `json:"replay_error,omitempty"`
}

type replayBoundReceipt struct {
	Version         int                            `json:"version"`
	PlanID          string                         `json:"plan_id"`
	PlanDigest      string                         `json:"plan_digest"`
	SpecDigest      string                         `json:"spec_digest"`
	Mode            string                         `json:"mode"`
	ArtifactClosure []replayArtifactReceipt        `json:"artifact_closure"`
	Environment     cudaregress.Environment        `json:"environment_observed"`
	Provenance      string                         `json:"environment_provenance"`
	Execution       *cudaregress.ExecutionReceipt  `json:"execution,omitempty"`
	Comparison      *cudaregress.ComparisonReceipt `json:"comparison,omitempty"`
}

type replayArtifactReceipt struct {
	Role   string `json:"role"`
	Name   string `json:"name"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

func runRegressReplayCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("regress replay", flag.ContinueOnError)
	planPath := fs.String("plan", "", "path to validated experiment plan JSON")
	specPath := fs.String("spec", "", "path to versioned direct-execution spec JSON")
	envPath := fs.String("environment", "", "path to separately observed environment JSON")
	outputDir := fs.String("o", "", "directory for execution and comparison receipts")
	historyTemplatePath := fs.String("history-template", "", "optional caller-supplied profile history identity record")
	historyBindingPath := fs.String("history-binding", "", "optional asserted plan-to-history identity binding")
	historyOutPath := fs.String("history-out", "", "optional profile history snapshot output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *planPath == "" || *specPath == "" || *envPath == "" || *outputDir == "" {
		return errors.New("usage: cudaprof regress replay -plan plan.json -spec execution-spec.json -environment observed-environment.json -o receipts-dir [-history-template record.json -history-binding binding.json -history-out history.ndjson]")
	}
	historyFlags := 0
	for _, path := range []string{*historyTemplatePath, *historyBindingPath, *historyOutPath} {
		if path != "" {
			historyFlags++
		}
	}
	if historyFlags != 0 && historyFlags != 3 {
		return errors.New("history-template, history-binding, and history-out must be supplied together")
	}

	plan, err := readRegressPlanFile(*planPath)
	if err != nil {
		return fmt.Errorf("read plan: %w", err)
	}
	specFile, err := os.Open(*specPath)
	if err != nil {
		return fmt.Errorf("open execution spec: %w", err)
	}
	spec, err := cudaregress.ReadExecutionSpec(specFile)
	closeErr := specFile.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if err := cudaregress.ValidateExecutionSpec(&plan, &spec); err != nil {
		return err
	}
	var observed observedEnvironmentFile
	if err := readStrictFile(*envPath, &observed); err != nil {
		return fmt.Errorf("read observed environment: %w", err)
	}
	if observed.Version != 1 || strings.TrimSpace(observed.Provenance) == "" {
		return errors.New("observed environment requires version 1 and provenance")
	}
	if err := cudaregress.ValidateEnvironment(&plan, observed.Environment); err != nil {
		return err
	}

	lockOwnerPID := 0
	var reference []float64
	if plan.CorrectnessOracle.Method == "numerical-tolerance" {
		if plan.CorrectnessOracle.ReferencePath == "" || plan.CorrectnessOracle.ReferenceDigest == "" {
			return errors.New("numerical oracle requires a declared reference artifact")
		}
		digest, _, err := cudaregress.ComputeFileDigest(plan.CorrectnessOracle.ReferencePath)
		if err != nil {
			return err
		}
		if digest != plan.CorrectnessOracle.ReferenceDigest {
			return errors.New("reference artifact digest mismatch")
		}
		if reference, err = readReferenceValues(plan.CorrectnessOracle.ReferencePath); err != nil {
			return fmt.Errorf("read reference values: %w", err)
		}
		if len(reference) == 0 {
			return errors.New("reference values are empty")
		}
		for _, v := range reference {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return errors.New("reference values contain non-finite number")
			}
		}
	}
	runner, err := cudaregress.NewCommandReplayRunner(&plan, spec)
	if err != nil {
		return err
	}
	if spec.Mode == "gpu" || plan.DeviceConfig.RequireHardwareLock {
		lockOwnerPID, _ = strconv.Atoi(spec.Environment["GPU_LOCK_HELD"])
	}
	closure := replayArtifactClosure(&plan)
	commandReceipt := replayCommandReceipt{Version: 1, PlanID: plan.PlanID, PlanDigest: plan.PlanDigest, SpecDigest: spec.SpecDigest, Mode: spec.Mode, ArtifactClosure: closure, EnvironmentObserved: observed.Environment, EnvironmentProvenance: observed.Provenance, LockOwnerPID: lockOwnerPID, Attempts: []cudaregress.CommandAttempt{}}
	replayOptions := cudaregress.ReplayOptions{CurrentEnvironment: observed.Environment, ReferenceValues: reference}
	execRec, compRec, replayErr := cudaregress.Replay(context.Background(), &plan, runner, replayOptions)
	if execRec != nil && compRec == nil {
		partialComparison, compareErr := cudaregress.CompareExecution(&plan, execRec, replayOptions)
		if compareErr == nil {
			compRec = partialComparison
		} else if replayErr == nil {
			replayErr = compareErr
		}
	}
	commandReceipt.Execution, commandReceipt.Comparison = execRec, compRec
	commandReceipt.Attempts = runner.Attempts()
	if replayErr != nil {
		commandReceipt.ReplayError = replayErr.Error()
	}
	if replayErr == nil && !replayComplete(&plan, execRec, compRec) {
		replayErr = errors.New("replay did not complete all declared arms, rounds, outputs, and comparison")
		commandReceipt.ReplayError = replayErr.Error()
	}
	if *historyTemplatePath != "" {
		if execRec == nil || compRec == nil || execRec.Partial || !replayComplete(&plan, execRec, compRec) {
			commandReceipt.HistoryWithheldReason = "execution or comparison is incomplete or indeterminate"
		} else {
			key, templateDigest, bindingDigest, bindingSource, historyErr := appendReplayHistory(&plan, execRec, compRec, observed.Environment, *historyTemplatePath, *historyBindingPath, *historyOutPath)
			commandReceipt.HistoryTemplateDigest = templateDigest
			commandReceipt.HistoryBindingDigest = bindingDigest
			commandReceipt.HistoryBindingSource = bindingSource
			if historyErr != nil {
				commandReceipt.HistoryWithheldReason = historyErr.Error()
			} else {
				commandReceipt.HistoryRecordKey = key
			}
		}
	}
	if err := writeReplayReceipts(*outputDir, commandReceipt); err != nil {
		return err
	}
	if replayErr != nil {
		return fmt.Errorf("replay failed (receipts retained): %w", replayErr)
	}
	_, err = fmt.Fprintf(stdout, "Replayed plan %s; execution %s\n", plan.PlanID, execRec.ExecutionID)
	return err
}

func replayComplete(plan *cudaregress.ExperimentPlan, execution *cudaregress.ExecutionReceipt, comparison *cudaregress.ComparisonReceipt) bool {
	if plan == nil || execution == nil || comparison == nil || execution.Partial || !execution.OutputsVerified || execution.RoundsCompleted != plan.Rounds || execution.WarmupRunsCompleted != plan.WarmupRuns {
		return false
	}
	if len(execution.ArmReceipts) != plan.Rounds*2 {
		return false
	}
	seen := make(map[int]map[string]bool, plan.Rounds)
	for _, receipt := range execution.ArmReceipts {
		if !receipt.Complete || receipt.Round < 1 || receipt.Round > plan.Rounds || (receipt.Name != "baseline" && receipt.Name != "candidate") {
			return false
		}
		if seen[receipt.Round] == nil {
			seen[receipt.Round] = make(map[string]bool, 2)
		}
		if seen[receipt.Round][receipt.Name] {
			return false
		}
		seen[receipt.Round][receipt.Name] = true
	}
	for round := 1; round <= plan.Rounds; round++ {
		if !seen[round]["baseline"] || !seen[round]["candidate"] {
			return false
		}
	}
	wantNullControls := 0
	if plan.NullControls.Enabled && plan.NullControls.BasketRuns > 1 {
		wantNullControls = plan.NullControls.BasketRuns
	}
	if len(execution.NullControlReceipts) != wantNullControls {
		return false
	}
	for _, receipt := range execution.NullControlReceipts {
		if !receipt.Complete {
			return false
		}
	}
	return comparison.ExecutionID == execution.ExecutionID && comparison.PlanID == plan.PlanID && (comparison.Verdict == "pass" || comparison.Verdict == "fail")
}

func replayArtifactClosure(plan *cudaregress.ExperimentPlan) []replayArtifactReceipt {
	closure := make([]replayArtifactReceipt, 0, len(plan.BinaryArtifacts)+len(plan.Inputs)+len(plan.LibraryClosure))
	for _, artifact := range plan.BinaryArtifacts {
		role := artifact.Role
		if role == "" {
			role = "binary"
		}
		closure = append(closure, replayArtifactReceipt{Role: role, Name: artifact.Name, Path: artifact.Path, SHA256: artifact.Digest})
	}
	for _, artifact := range plan.Inputs {
		closure = append(closure, replayArtifactReceipt{Role: "input", Name: artifact.Name, Path: artifact.Path, SHA256: artifact.Digest})
	}
	for _, artifact := range plan.LibraryClosure {
		closure = append(closure, replayArtifactReceipt{Role: "library", Name: artifact.Name, Path: artifact.Path, SHA256: artifact.Digest})
	}
	sort.Slice(closure, func(i, j int) bool {
		if closure[i].Role != closure[j].Role {
			return closure[i].Role < closure[j].Role
		}
		if closure[i].Name != closure[j].Name {
			return closure[i].Name < closure[j].Name
		}
		return closure[i].Path < closure[j].Path
	})
	return closure
}

func readReferenceValues(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, maxReplayInputBytes+1))
	if err != nil {
		return nil, err
	}
	if len(b) > maxReplayInputBytes {
		return nil, errors.New("input byte limit exceeded")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var raw []json.RawMessage
	if err := d.Decode(&raw); err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, errors.New("trailing JSON data")
	}
	values := make([]float64, len(raw))
	for i, element := range raw {
		var value any
		elementDecoder := json.NewDecoder(bytes.NewReader(element))
		elementDecoder.UseNumber()
		if err := elementDecoder.Decode(&value); err != nil {
			return nil, err
		}
		number, ok := value.(json.Number)
		if !ok {
			return nil, fmt.Errorf("reference value %d is not a number", i)
		}
		values[i], err = number.Float64()
		if err != nil || math.IsNaN(values[i]) || math.IsInf(values[i], 0) {
			return nil, fmt.Errorf("reference value %d is not finite", i)
		}
	}
	return values, nil
}

func appendReplayHistory(plan *cudaregress.ExperimentPlan, execRec *cudaregress.ExecutionReceipt, compRec *cudaregress.ComparisonReceipt, observed cudaregress.Environment, templatePath, bindingPath, outPath string) (string, string, string, string, error) {
	templateDigest, _, err := cudaregress.ComputeFileDigest(templatePath)
	if err != nil {
		return "", "", "", "", fmt.Errorf("history template: %w", err)
	}
	var template cudaprofilehistory.Record
	if err := readStrictFile(templatePath, &template); err != nil {
		return "", templateDigest, "", "", fmt.Errorf("history template: %w", err)
	}
	if err := template.Validate(); err != nil {
		return "", templateDigest, "", "", fmt.Errorf("history template: %w", err)
	}
	bindingDigest, _, err := cudaregress.ComputeFileDigest(bindingPath)
	if err != nil {
		return "", templateDigest, "", "", fmt.Errorf("history binding: %w", err)
	}
	var binding historyBindingFile
	if err := readStrictFile(bindingPath, &binding); err != nil {
		return "", templateDigest, bindingDigest, "", fmt.Errorf("history binding: %w", err)
	}
	if binding.Version != 1 || binding.PlanFrozenDigest != plan.PlanDigest || binding.HistoryTemplateDigest != templateDigest || !reflect.DeepEqual(binding.PlanWorkload, plan.Workload) || !reflect.DeepEqual(binding.HistoryWorkload, template.Workload) || strings.TrimSpace(binding.Provenance) == "" {
		return "", templateDigest, bindingDigest, binding.Provenance, errors.New("history binding does not match frozen plan, exact template, workload identities, or provenance")
	}
	if template.Environment.Device != observed.Model || template.Environment.Driver != observed.Driver || template.Environment.CUDA != observed.CUDA || template.Environment.Clock != observed.Clock || template.Environment.Domain != observed.Domain {
		return "", templateDigest, bindingDigest, binding.Provenance, errors.New("history environment does not match observed environment")
	}
	if template.Artifact == nil {
		return "", templateDigest, bindingDigest, binding.Provenance, errors.New("history template has no supplied kernel artifact")
	}
	found := false
	for _, b := range plan.BinaryArtifacts {
		if b.Digest == template.Artifact.Digest {
			found = true
			break
		}
	}
	if !found {
		return "", templateDigest, bindingDigest, binding.Provenance, errors.New("history artifact digest is not in plan binary closure")
	}
	if err := template.Artifact.Validate(); err != nil {
		return "", templateDigest, bindingDigest, binding.Provenance, fmt.Errorf("history artifact: %w", err)
	}
	compilerInputKey := template.CompilerInputKey
	storeLimits := cudaprofilehistory.Limits{MaxRecords: 10000, MaxBytes: 64 << 20, MaxRecordBytes: 4 << 20}
	var store *cudaprofilehistory.Store
	if existing, err := os.Open(outPath); err == nil {
		store, err = cudaprofilehistory.ReadSnapshot(existing, storeLimits)
		closeErr := existing.Close()
		if err != nil {
			return "", templateDigest, bindingDigest, binding.Provenance, err
		}
		if closeErr != nil {
			return "", templateDigest, bindingDigest, binding.Provenance, closeErr
		}
	} else if !os.IsNotExist(err) {
		return "", templateDigest, bindingDigest, binding.Provenance, err
	} else {
		store, err = cudaprofilehistory.NewStore(storeLimits)
		if err != nil {
			return "", templateDigest, bindingDigest, binding.Provenance, err
		}
	}
	record, err := cudaregress.LinkToProfileHistory(plan, execRec, compRec, cudaregress.HistoryLinkOptions{CompilerInputKey: compilerInputKey, Compiler: template.Compiler, Workload: template.Workload, Entry: template.Entry, Artifact: template.Artifact, Qualified: false})
	if err != nil {
		return "", templateDigest, bindingDigest, binding.Provenance, err
	}
	key, err := store.Append(record)
	if err != nil {
		return "", templateDigest, bindingDigest, binding.Provenance, err
	}
	var buf bytes.Buffer
	if err := store.WriteSnapshot(&buf); err != nil {
		return "", templateDigest, bindingDigest, binding.Provenance, err
	}
	if err := writeAtomic(outPath, buf.Bytes()); err != nil {
		return "", templateDigest, bindingDigest, binding.Provenance, err
	}
	return key, templateDigest, bindingDigest, binding.Provenance, nil
}

func readStrictFile(path string, out any) error {
	b, err := readBoundedJSONFile(path)
	if err != nil {
		return err
	}
	return decodeStrictJSON(b, out)
}

func readBoundedJSONFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, maxReplayInputBytes+1))
	if err != nil {
		return nil, err
	}
	if len(b) > maxReplayInputBytes {
		return nil, errors.New("input byte limit exceeded")
	}
	return b, nil
}

func decodeStrictJSON(b []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := scanStrictValue(d, reflect.TypeOf(out).Elem(), 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing JSON data")
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return errors.New("trailing JSON data")
	}
	return nil
}

func scanStrictValue(d *json.Decoder, typ reflect.Type, depth int) error {
	if depth > 64 {
		return errors.New("JSON nesting limit exceeded")
	}
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	jsonUnmarshaler := reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()
	if reflect.PointerTo(typ).Implements(jsonUnmarshaler) || typ.Implements(jsonUnmarshaler) {
		var raw json.RawMessage
		if err := d.Decode(&raw); err != nil {
			return err
		}
		if err := scanStrictJSONStructure(raw, depth); err != nil {
			return err
		}
		return json.Unmarshal(raw, reflect.New(typ).Interface())
	}
	tok, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		fields := map[string]reflect.Type{}
		if typ.Kind() == reflect.Struct {
			for i := 0; i < typ.NumField(); i++ {
				f := typ.Field(i)
				if f.PkgPath != "" {
					continue
				}
				name := strings.Split(f.Tag.Get("json"), ",")[0]
				if name == "" {
					name = f.Name
				}
				if name != "-" {
					fields[name] = f.Type
				}
			}
		}
		seen := map[string]bool{}
		for d.More() {
			k, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := k.(string)
			if !ok {
				return errors.New("invalid object key")
			}
			fold := strings.ToLower(key)
			if seen[fold] {
				return errors.New("duplicate or case-alias JSON field")
			}
			seen[fold] = true
			ft, ok := fields[key]
			if !ok {
				if typ.Kind() == reflect.Map {
					ft = typ.Elem()
				} else {
					return fmt.Errorf("unknown or case-alias JSON field %q", key)
				}
			}
			if err := scanStrictValue(d, ft, depth+1); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	case '[':
		et := reflect.TypeOf((*any)(nil)).Elem()
		if typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array {
			et = typ.Elem()
		}
		for d.More() {
			if err := scanStrictValue(d, et, depth+1); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	default:
		return errors.New("unexpected JSON delimiter")
	}
}

func scanStrictJSONStructure(b []byte, depth int) error {
	d := json.NewDecoder(bytes.NewReader(b))
	var walk func(int) error
	walk = func(level int) error {
		if level > 64 {
			return errors.New("JSON nesting limit exceeded")
		}
		tok, err := d.Token()
		if err != nil {
			return err
		}
		delim, ok := tok.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]bool)
			for d.More() {
				token, err := d.Token()
				if err != nil {
					return err
				}
				key, ok := token.(string)
				if !ok {
					return errors.New("invalid object key")
				}
				fold := strings.ToLower(key)
				if seen[fold] {
					return errors.New("duplicate or case-alias JSON field")
				}
				seen[fold] = true
				if err := walk(level + 1); err != nil {
					return err
				}
			}
			_, err = d.Token()
			return err
		case '[':
			for d.More() {
				if err := walk(level + 1); err != nil {
					return err
				}
			}
			_, err = d.Token()
			return err
		default:
			return errors.New("unexpected JSON delimiter")
		}
	}
	if err := walk(depth); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing JSON data")
	}
	return nil
}

func writeReplayReceipts(dir string, receipt replayCommandReceipt) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	output := receipt
	output.Execution = executionReceiptForJSON(receipt.Execution)
	if receipt.Execution != nil {
		bound := replayBoundReceipt{Version: 1, PlanID: receipt.PlanID, PlanDigest: receipt.PlanDigest, SpecDigest: receipt.SpecDigest, Mode: receipt.Mode, ArtifactClosure: receipt.ArtifactClosure, Environment: receipt.EnvironmentObserved, Provenance: receipt.EnvironmentProvenance, Execution: output.Execution}
		b, err := json.MarshalIndent(bound, "", "  ")
		if err != nil {
			return err
		}
		if err := writeAtomic(filepath.Join(dir, "execution.json"), b); err != nil {
			return err
		}
	}
	if receipt.Comparison != nil {
		bound := replayBoundReceipt{Version: 1, PlanID: receipt.PlanID, PlanDigest: receipt.PlanDigest, SpecDigest: receipt.SpecDigest, Mode: receipt.Mode, ArtifactClosure: receipt.ArtifactClosure, Environment: receipt.EnvironmentObserved, Provenance: receipt.EnvironmentProvenance, Comparison: receipt.Comparison}
		b, err := json.MarshalIndent(bound, "", "  ")
		if err != nil {
			return err
		}
		if err := writeAtomic(filepath.Join(dir, "comparison.json"), b); err != nil {
			return err
		}
	}
	b, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(filepath.Join(dir, "command.json"), b)
}

func executionReceiptForJSON(receipt *cudaregress.ExecutionReceipt) *cudaregress.ExecutionReceipt {
	if receipt == nil {
		return nil
	}
	copyReceipt := *receipt
	copyReceipt.ArmReceipts = append(receipt.ArmReceipts[:0:0], receipt.ArmReceipts...)
	copyReceipt.NullControlReceipts = append(receipt.NullControlReceipts[:0:0], receipt.NullControlReceipts...)
	for i := range copyReceipt.ArmReceipts {
		copyReceipt.ArmReceipts[i].Err = nil
	}
	for i := range copyReceipt.NullControlReceipts {
		copyReceipt.NullControlReceipts[i].Err = nil
	}
	return &copyReceipt
}

func writeAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".replay-*.tmp")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if err = f.Chmod(0644); err == nil {
		var n int
		n, err = f.Write(data)
		if err == nil && n != len(data) {
			err = io.ErrShortWrite
		}
	}
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(name, path)
}
