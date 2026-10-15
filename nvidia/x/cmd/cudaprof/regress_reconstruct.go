package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const (
	reconstructInventorySchema = "cudaprof.incident-reconstruct.v1"
	reconstructMaxBytes        = 64 << 20
	reconstructMaxFiles        = 256
	reconstructMaxManifest     = 4 << 20
)

type reconstructProducer struct {
	Name                string   `json:"name"`
	Version             string   `json:"version"`
	Schema              string   `json:"schema"`
	Tool                string   `json:"tool"`
	ToolVersion         string   `json:"tool_version"`
	ToolSHA256          string   `json:"tool_sha256"`
	ModuleSHA256s       []string `json:"module_sha256s"`
	CandidateTreeSHA256 string   `json:"candidate_tree_sha256"`
	CaptureConfigSHA256 string   `json:"capture_config_sha256"`
}

type reconstructTelemetryOptions struct {
	MaxBytes           int64                              `json:"max_bytes"`
	Workload           cudaworkload.WorkloadIdentity      `json:"workload"`
	OpID               cudaworkload.OpID                  `json:"op_id"`
	Assignment         nvtelemetry.EntityAssignment       `json:"assignment"`
	ClockEpoch         string                             `json:"clock_epoch"`
	Observer           nvtelemetry.Observer               `json:"observer"`
	Receipt            *tracecontract.Receipt             `json:"receipt"`
	ExpectedSources    map[string]nvtelemetry.Observation `json:"expected_sources"`
	HistogramBounds    map[string][]string                `json:"histogram_bounds,omitempty"`
	RequireDenominator []string                           `json:"require_denominator,omitempty"`
	WallInterval       tracecontract.Interval             `json:"wall_interval"`
	Thresholds         cudadiag.Thresholds                `json:"thresholds"`
}

type reconstructInventory struct {
	Version          int                         `json:"version"`
	Schema           string                      `json:"schema"`
	Producer         reconstructProducer         `json:"producer"`
	Artifacts        []cudaregress.ArtifactRef   `json:"artifacts"`
	TelemetryStream  string                      `json:"telemetry_stream"`
	RawArtifactPaths map[string]string           `json:"raw_artifact_paths"`
	Telemetry        reconstructTelemetryOptions `json:"telemetry"`
	Observed         cudaregress.Environment     `json:"observed_environment"`
	Plan             cudaregress.PlanOptions     `json:"plan_options"`
}

type reconstructReceipt struct {
	Version         int                         `json:"version"`
	InventorySHA256 string                      `json:"inventory_sha256"`
	Producer        reconstructProducer         `json:"producer"`
	TelemetrySchema int                         `json:"telemetry_schema"`
	Diagnosis       cudadiag.TelemetryDiagnosis `json:"diagnosis"`
	Incident        cudaregress.Incident        `json:"incident"`
	PlanReceipt     *cudaregress.PlanReceipt    `json:"plan_receipt,omitempty"`
	PlanError       string                      `json:"plan_error,omitempty"`
}

type reconstructResult struct {
	Receipt reconstructReceipt          `json:"receipt"`
	Plan    *cudaregress.ExperimentPlan `json:"plan,omitempty"`
}

type reconstructSnapshot struct {
	root          string
	files         map[string][]byte
	manifestBytes []byte
}

func runRegressReconstructCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("regress reconstruct", flag.ContinueOnError)
	inventoryPath := fs.String("inventory", "", "sealed flat incident inventory JSON")
	inventoryDigest := fs.String("inventory-sha256", "", "independently pinned inventory SHA-256")
	outputDir := fs.String("o", "", "new output directory for reconstruction and optional plan")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *inventoryPath == "" || *inventoryDigest == "" || *outputDir == "" {
		return errors.New("usage: cudaprof regress reconstruct -inventory packet.json -inventory-sha256 digest -o new-dir")
	}
	if !reconstructSHA256(*inventoryDigest) {
		return errors.New("reconstruct: inventory SHA-256 must be lowercase hex")
	}

	snapshot, err := readReconstructSnapshot(*inventoryPath, *inventoryDigest)
	if err != nil {
		return err
	}
	var inventory reconstructInventory
	if err := decodeStrictJSON(snapshot.manifestBytes, &inventory); err != nil {
		return fmt.Errorf("reconstruct: decode pinned inventory: %w", err)
	}
	if err := validateReconstructInventory(&inventory, snapshot); err != nil {
		return err
	}

	telemetry := cudadiag.TelemetryOptions{
		MaxBytes:           inventory.Telemetry.MaxBytes,
		Workload:           inventory.Telemetry.Workload,
		OpID:               inventory.Telemetry.OpID,
		Assignment:         inventory.Telemetry.Assignment,
		ClockEpoch:         inventory.Telemetry.ClockEpoch,
		Observer:           inventory.Telemetry.Observer,
		Receipt:            inventory.Telemetry.Receipt,
		RawArtifacts:       make(map[string][]byte, len(inventory.RawArtifactPaths)),
		ExpectedSources:    inventory.Telemetry.ExpectedSources,
		HistogramBounds:    inventory.Telemetry.HistogramBounds,
		RequireDenominator: inventory.Telemetry.RequireDenominator,
		WallInterval:       inventory.Telemetry.WallInterval,
		Thresholds:         inventory.Telemetry.Thresholds,
	}
	for digest, name := range inventory.RawArtifactPaths {
		telemetry.RawArtifacts[digest] = snapshot.files[name]
	}
	stream := snapshot.files[inventory.TelemetryStream]
	streamSchema := reconstructStreamSchema(stream)
	if streamSchema != nvtelemetry.SchemaVersionV3 && streamSchema != nvtelemetry.SchemaVersionV4 {
		return fmt.Errorf("reconstruct: unsupported telemetry diagnosis schema %d", streamSchema)
	}
	diagnosis, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(stream), telemetry)
	if err != nil {
		return fmt.Errorf("reconstruct: diagnose verified telemetry: %w", err)
	}
	output, snapshotDir, cleanupOutput, err := beginReconstructOutput(*outputDir, snapshot)
	if err != nil {
		return err
	}
	completed := false
	defer func() {
		if !completed {
			cleanupOutput()
		}
	}()

	incident := cudaregress.Incident{
		ID:             "reconstructed-" + (*inventoryDigest)[:16],
		Admitted:       diagnosis.Diagnosis.Admitted,
		WithheldReason: diagnosis.Diagnosis.WithheldReason,
		Diagnosis:      diagnosis.Diagnosis,
		PrimaryFinding: diagnosis.Diagnosis.PrimaryFinding,
		Workload:       inventory.Telemetry.Workload,
		CaptureReceiptRef: cudaregress.ArtifactRef{
			Name: "sealed incident inventory", Path: filepath.Join(snapshotDir, "inventory.json"),
			Digest: *inventoryDigest, SizeBytes: int64(len(snapshot.manifestBytes)), Role: "trace",
		},
		ObservedEnvironment: inventory.Observed,
		CreatedAt:           time.Now().UTC(),
	}
	result := reconstructResult{Receipt: reconstructReceipt{
		Version: 1, InventorySHA256: *inventoryDigest,
		Producer:        inventory.Producer,
		TelemetrySchema: streamSchema,
		Diagnosis:       diagnosis, Incident: incident,
	}}
	planOptions, err := reconstructPlanOptions(inventory.Plan, snapshot, snapshotDir)
	if err != nil {
		result.Receipt.PlanError = err.Error()
	} else {
		plan, planReceipt, planErr := cudaregress.NewPlanFromIncident(incident, planOptions)
		if planErr != nil {
			result.Receipt.PlanError = planErr.Error()
		} else {
			result.Plan = plan
			result.Receipt.PlanReceipt = planReceipt
		}
	}
	if !incident.Admitted && result.Plan != nil {
		return errors.New("reconstruct: withheld diagnosis unexpectedly produced a plan")
	}
	if err := writeReconstructResult(output, result); err != nil {
		return err
	}
	completed = true
	return nil
}

func readReconstructSnapshot(inventoryPath, wantDigest string) (*reconstructSnapshot, error) {
	absolute, err := filepath.Abs(inventoryPath)
	if err != nil {
		return nil, err
	}
	root := filepath.Dir(absolute)
	manifestName := filepath.Base(absolute)
	manifestBytes, err := readReconstructFile(absolute, reconstructMaxManifest)
	if err != nil {
		return nil, fmt.Errorf("reconstruct: read inventory snapshot: %w", err)
	}
	if reconstructDigest(manifestBytes) != wantDigest {
		return nil, errors.New("reconstruct: inventory digest mismatch")
	}
	entries, err := readReconstructDirectory(root)
	if err != nil {
		return nil, fmt.Errorf("reconstruct: enumerate inventory directory: %w", err)
	}
	if len(entries) < 2 || len(entries) > reconstructMaxFiles+1 {
		return nil, errors.New("reconstruct: inventory directory file count limit")
	}
	for _, entry := range entries {
		if entry.Type()&os.ModeSymlink != 0 || !entry.Type().IsRegular() {
			return nil, fmt.Errorf("reconstruct: directory entry %q is not a regular file", entry.Name())
		}
	}

	// The root digest pins the manifest before strict JSON decoding. Payload files
	// remain opaque until every entry has been read and matched to that inventory.
	var inventory reconstructInventory
	if err := decodeStrictJSON(manifestBytes, &inventory); err != nil {
		return nil, fmt.Errorf("reconstruct: decode pinned inventory: %w", err)
	}
	if len(inventory.Artifacts) == 0 || len(inventory.Artifacts) > reconstructMaxFiles {
		return nil, errors.New("reconstruct: artifact count limit")
	}
	declared := make(map[string]cudaregress.ArtifactRef, len(inventory.Artifacts))
	for _, ref := range inventory.Artifacts {
		if ref.Name == "" || ref.Role == "" || ref.Path == "" || filepath.Base(ref.Path) != ref.Path || strings.ContainsAny(ref.Path, "/\\\x00") || ref.Path == manifestName || ref.Path == "." || ref.Path == ".." || ref.SizeBytes < 0 || ref.SizeBytes > reconstructMaxBytes || !reconstructSHA256(ref.Digest) {
			return nil, fmt.Errorf("reconstruct: invalid artifact reference %q", ref.Name)
		}
		if _, exists := declared[ref.Path]; exists {
			return nil, fmt.Errorf("reconstruct: duplicate artifact path %q", ref.Path)
		}
		declared[ref.Path] = ref
	}
	if len(entries) != len(declared)+1 {
		return nil, errors.New("reconstruct: inventory directory contains missing or undeclared files")
	}
	files := make(map[string][]byte, len(declared))
	total := int64(len(manifestBytes))
	for _, entry := range entries {
		if entry.Name() == manifestName {
			continue
		}
		ref, ok := declared[entry.Name()]
		if !ok {
			return nil, fmt.Errorf("reconstruct: undeclared file %q", entry.Name())
		}
		remaining := int64(reconstructMaxBytes) - total
		if remaining < 0 || ref.SizeBytes > remaining {
			return nil, errors.New("reconstruct: aggregate byte limit")
		}
		data, err := readReconstructFile(filepath.Join(root, entry.Name()), remaining)
		if err != nil {
			return nil, fmt.Errorf("reconstruct: read artifact %q: %w", entry.Name(), err)
		}
		if int64(len(data)) != ref.SizeBytes || reconstructDigest(data) != ref.Digest {
			return nil, fmt.Errorf("reconstruct: artifact %q size or digest mismatch", ref.Name)
		}
		total += int64(len(data))
		files[entry.Name()] = data
	}
	return &reconstructSnapshot{root: root, files: files, manifestBytes: manifestBytes}, nil
}

func readReconstructDirectory(root string) ([]os.DirEntry, error) {
	before, err := os.Lstat(root)
	if err != nil {
		return nil, err
	}
	if before.Mode()&os.ModeSymlink != 0 || !before.IsDir() {
		return nil, errors.New("inventory root is not a directory")
	}
	dir, err := os.Open(root)
	if err != nil {
		return nil, err
	}
	after, err := dir.Stat()
	if err != nil {
		return nil, errors.Join(err, dir.Close())
	}
	if !after.IsDir() || !os.SameFile(before, after) {
		return nil, errors.Join(errors.New("inventory directory changed during open"), dir.Close())
	}
	entries, readErr := dir.ReadDir(reconstructMaxFiles + 2)
	closeErr := dir.Close()
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		return nil, errors.Join(readErr, closeErr)
	}
	if closeErr != nil {
		return nil, closeErr
	}
	if len(entries) > reconstructMaxFiles+1 {
		return nil, errors.New("inventory directory file count limit")
	}
	return entries, nil
}

func readReconstructFile(path string, limit int64) ([]byte, error) {
	before, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if before.Mode()&os.ModeSymlink != 0 || !before.Mode().IsRegular() || before.Size() < 0 || before.Size() > limit {
		return nil, errors.New("not a bounded regular file")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	after, err := f.Stat()
	if err != nil {
		return nil, errors.Join(err, f.Close())
	}
	if !after.Mode().IsRegular() || !os.SameFile(before, after) || after.Size() != before.Size() {
		return nil, errors.Join(errors.New("file changed during open"), f.Close())
	}
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		return nil, errors.Join(err, closeErr)
	}
	if int64(len(data)) > limit || int64(len(data)) != after.Size() {
		return nil, errors.New("file changed or exceeded byte limit")
	}
	return data, nil
}

func validateReconstructInventory(inventory *reconstructInventory, snapshot *reconstructSnapshot) error {
	if inventory.Version != 1 || inventory.Schema != reconstructInventorySchema {
		return errors.New("reconstruct: unsupported inventory schema")
	}
	if inventory.Producer.Name == "" || inventory.Producer.Version == "" || inventory.Producer.Schema == "" || inventory.Producer.Tool == "" || inventory.Producer.ToolVersion == "" || !reconstructSHA256(inventory.Producer.ToolSHA256) || !reconstructSHA256(inventory.Producer.CandidateTreeSHA256) || !reconstructSHA256(inventory.Producer.CaptureConfigSHA256) || len(inventory.Producer.ModuleSHA256s) == 0 || len(inventory.Producer.ModuleSHA256s) > reconstructMaxFiles {
		return errors.New("reconstruct: incomplete producer or capture provenance")
	}
	if inventory.TelemetryStream == "" || inventory.Telemetry.Receipt == nil || inventory.Telemetry.MaxBytes <= 0 || inventory.Telemetry.MaxBytes > reconstructMaxBytes {
		return errors.New("reconstruct: missing telemetry stream, receipt or byte bound")
	}
	if _, ok := snapshot.files[inventory.TelemetryStream]; !ok {
		return errors.New("reconstruct: telemetry stream is outside the verified inventory")
	}
	if len(inventory.RawArtifactPaths) == 0 || len(inventory.RawArtifactPaths) > reconstructMaxFiles || len(inventory.Telemetry.ExpectedSources) == 0 || len(inventory.Telemetry.ExpectedSources) > reconstructMaxFiles {
		return errors.New("reconstruct: raw source reference count limit")
	}
	refs := make(map[string]cudaregress.ArtifactRef, len(inventory.Artifacts))
	byDigest := make(map[string]string, len(inventory.Artifacts))
	names := make(map[string]bool, len(inventory.Artifacts))
	allowedRoles := map[string]bool{
		"telemetry_stream": true, "raw": true, "trace": true, "transform": true,
		"source": true, "binary": true, "module": true, "library": true,
		"tool": true, "input": true, "oracle": true, "candidate_tree": true,
		"capture_config": true,
	}
	for _, ref := range inventory.Artifacts {
		if !allowedRoles[ref.Role] || names[ref.Name] {
			return fmt.Errorf("reconstruct: unsupported role or duplicate artifact name %q", ref.Name)
		}
		names[ref.Name] = true
		refs[ref.Path] = ref
		if prior, exists := byDigest[ref.Digest]; exists && prior != ref.Path {
			return fmt.Errorf("reconstruct: duplicate artifact digest at %q and %q", prior, ref.Path)
		}
		byDigest[ref.Digest] = ref.Path
	}
	if byDigest[inventory.Producer.ToolSHA256] == "" || refs[byDigest[inventory.Producer.ToolSHA256]].Role != "tool" || byDigest[inventory.Producer.CandidateTreeSHA256] == "" || refs[byDigest[inventory.Producer.CandidateTreeSHA256]].Role != "candidate_tree" || byDigest[inventory.Producer.CaptureConfigSHA256] == "" || refs[byDigest[inventory.Producer.CaptureConfigSHA256]].Role != "capture_config" {
		return errors.New("reconstruct: tool, candidate tree or capture configuration is not in the typed closure")
	}
	for _, digest := range inventory.Producer.ModuleSHA256s {
		if !reconstructSHA256(digest) || byDigest[digest] == "" || refs[byDigest[digest]].Role != "module" {
			return errors.New("reconstruct: module closure does not match a retained module artifact")
		}
	}
	streamRef, ok := refs[inventory.TelemetryStream]
	if !ok || streamRef.Role != "telemetry_stream" {
		return errors.New("reconstruct: telemetry stream role mismatch")
	}
	for _, role := range []string{"source", "binary", "module", "library", "tool", "input", "oracle", "candidate_tree", "capture_config"} {
		found := false
		for _, ref := range inventory.Artifacts {
			if ref.Role == role {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("reconstruct: inventory lacks required %q closure", role)
		}
	}
	if _, ok := inventory.RawArtifactPaths[inventory.Telemetry.Receipt.InputSHA256]; !ok {
		return errors.New("reconstruct: independent trace receipt bytes are not inventoried")
	}
	if inventory.Telemetry.ExpectedSources == nil || len(inventory.Telemetry.ExpectedSources) == 0 {
		return errors.New("reconstruct: expected source provenance is empty")
	}
	for digest := range inventory.RawArtifactPaths {
		path, ok := inventory.RawArtifactPaths[digest]
		if !ok || byDigest[digest] != path || refs[path].Role != "raw" && refs[path].Role != "trace" && refs[path].Role != "transform" {
			return fmt.Errorf("reconstruct: raw artifact %q is not in verified closure", digest)
		}
	}
	for digest := range inventory.Telemetry.ExpectedSources {
		if _, ok := inventory.RawArtifactPaths[digest]; !ok {
			return fmt.Errorf("reconstruct: expected source %q has no retained raw bytes", digest)
		}
	}
	for _, source := range inventory.Telemetry.ExpectedSources {
		if source.Producer != inventory.Producer.Name || source.ProducerVersion != inventory.Producer.Version || source.ProducerSchema != inventory.Producer.Schema {
			return errors.New("reconstruct: expected producer metadata differs from the pinned inventory")
		}
		if _, ok := inventory.RawArtifactPaths[source.TransformSHA256]; !ok {
			return fmt.Errorf("reconstruct: transform %q has no retained raw bytes", source.TransformSHA256)
		}
	}
	device := inventory.Telemetry.Assignment.Device
	if inventory.Telemetry.Workload.Process.PID == 0 || inventory.Telemetry.Workload.Process.Generation == 0 || inventory.Telemetry.OpID == 0 || device.UUID == "" || inventory.Observed.DeviceUUID != device.UUID || inventory.Observed.DeviceIndex != device.Index || inventory.Observed.Model != device.Model || device.DriverVersion != "" && inventory.Observed.Driver != device.DriverVersion {
		return errors.New("reconstruct: workload or observed device identity is incomplete")
	}
	if inventory.Observed.Model == "" || inventory.Observed.Driver == "" || inventory.Observed.CUDA == "" || inventory.Observed.Clock == "" || inventory.Observed.Domain == "" {
		return errors.New("reconstruct: observed environment identity is incomplete")
	}
	return validateReconstructPlanRefs(inventory.Plan, refs, snapshot.root)
}

func validateReconstructPlanRefs(plan cudaregress.PlanOptions, refs map[string]cudaregress.ArtifactRef, root string) error {
	resolve := func(name, path, digest, role string) (string, error) {
		if path == "" || path != filepath.Base(path) || path == "." || path == ".." {
			return "", fmt.Errorf("reconstruct: invalid plan artifact path %q", path)
		}
		clean := filepath.Clean(path)
		if filepath.IsAbs(clean) || strings.Contains(clean, string(filepath.Separator)) {
			return "", fmt.Errorf("reconstruct: plan artifact %q must name an inventoried flat file", name)
		}
		ref, ok := refs[clean]
		if !ok || digest == "" || ref.Digest != digest || ref.Role != role {
			return "", fmt.Errorf("reconstruct: plan artifact %q is outside the pinned inventory", name)
		}
		return filepath.Join(root, clean), nil
	}
	for _, ref := range plan.Sources {
		if _, err := resolve(ref.Name, ref.Path, ref.Digest, "source"); err != nil {
			return err
		}
	}
	for _, ref := range plan.Binaries {
		if _, err := resolve(ref.Name, ref.Path, ref.Digest, "binary"); err != nil {
			return err
		}
	}
	for _, lib := range plan.Libraries {
		if _, err := resolve(lib.Name, lib.Path, lib.Digest, "library"); err != nil {
			return err
		}
	}
	for _, input := range plan.Inputs {
		if input.Path == "" || input.Digest == "" {
			return fmt.Errorf("reconstruct: input %q has no retained bytes", input.Name)
		}
		if _, err := resolve(input.Name, input.Path, input.Digest, "input"); err != nil {
			return err
		}
	}
	if plan.Oracle.ReferencePath == "" || plan.Oracle.ReferenceDigest == "" {
		return errors.New("reconstruct: independent oracle reference is not inventoried")
	}
	if _, err := resolve("oracle reference", plan.Oracle.ReferencePath, plan.Oracle.ReferenceDigest, "oracle"); err != nil {
		return err
	}
	if len(plan.Sources) == 0 || len(plan.Binaries) == 0 || len(plan.Libraries) == 0 || len(plan.Inputs) == 0 || strings.TrimSpace(plan.Oracle.Method) == "" {
		return errors.New("reconstruct: plan requires source, binary, library, input and oracle closure")
	}
	return nil
}

func reconstructPlanOptions(plan cudaregress.PlanOptions, snapshot *reconstructSnapshot, snapshotDir string) (cudaregress.PlanOptions, error) {
	resolve := func(ref *cudaregress.ArtifactRef) {
		ref.Path = filepath.Join(snapshotDir, ref.Path)
	}
	for i := range plan.Sources {
		resolve(&plan.Sources[i])
	}
	for i := range plan.Binaries {
		resolve(&plan.Binaries[i])
	}
	for i := range plan.Libraries {
		plan.Libraries[i].Path = filepath.Join(snapshotDir, plan.Libraries[i].Path)
	}
	for i := range plan.Inputs {
		if plan.Inputs[i].Path != "" {
			plan.Inputs[i].Path = filepath.Join(snapshotDir, plan.Inputs[i].Path)
		}
	}
	if plan.Oracle.ReferencePath != "" {
		plan.Oracle.ReferencePath = filepath.Join(snapshotDir, plan.Oracle.ReferencePath)
	}
	if plan.DeviceOverrides != nil {
		return plan, errors.New("reconstruct: device overrides are not admitted by the observed incident")
	}
	return plan, nil
}

func reconstructStreamSchema(data []byte) int {
	var first struct {
		Manifest *nvtelemetry.Manifest `json:"manifest"`
	}
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		_ = json.Unmarshal(data[:i], &first)
	}
	if first.Manifest == nil {
		return 0
	}
	return first.Manifest.SchemaVersion
}

func reconstructSHA256(s string) bool {
	if len(s) != sha256.Size*2 || strings.ToLower(s) != s {
		return false
	}
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size
}

func reconstructDigest(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func beginReconstructOutput(output string, snapshot *reconstructSnapshot) (string, string, func(), error) {
	absolute, err := filepath.Abs(output)
	if err != nil {
		return "", "", nil, err
	}
	inputRoot, err := filepath.EvalSymlinks(snapshot.root)
	if err != nil {
		return "", "", nil, fmt.Errorf("reconstruct: resolve input directory: %w", err)
	}
	outputParent, err := filepath.EvalSymlinks(filepath.Dir(absolute))
	if err != nil {
		return "", "", nil, fmt.Errorf("reconstruct: resolve output parent: %w", err)
	}
	absolute = filepath.Join(outputParent, filepath.Base(absolute))
	rel, err := filepath.Rel(inputRoot, absolute)
	if err != nil {
		return "", "", nil, fmt.Errorf("reconstruct: compare resolved output path: %w", err)
	}
	if rel == "." || rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", "", nil, errors.New("reconstruct: output directory resolves inside the input inventory")
	}
	if _, err := os.Lstat(absolute); err == nil {
		return "", "", nil, errors.New("reconstruct: output directory already exists")
	} else if !os.IsNotExist(err) {
		return "", "", nil, err
	}
	if err := os.Mkdir(absolute, 0700); err != nil {
		return "", "", nil, err
	}
	cleanup := func() { _ = os.RemoveAll(absolute) }
	snapshotDir := filepath.Join(absolute, "snapshot")
	if err := os.Mkdir(snapshotDir, 0700); err != nil {
		cleanup()
		return "", "", nil, err
	}
	if err := writeNewFile(filepath.Join(snapshotDir, "inventory.json"), snapshot.manifestBytes); err != nil {
		cleanup()
		return "", "", nil, err
	}
	for name, data := range snapshot.files {
		if err := writeNewFile(filepath.Join(snapshotDir, name), data); err != nil {
			cleanup()
			return "", "", nil, err
		}
	}
	return absolute, snapshotDir, cleanup, nil
}

func writeReconstructResult(output string, result reconstructResult) error {
	receiptBytes, err := json.MarshalIndent(result.Receipt, "", "  ")
	if err != nil {
		return err
	}
	if err := writeNewFile(filepath.Join(output, "reconstruction.json"), append(receiptBytes, '\n')); err != nil {
		return err
	}
	if result.Plan != nil {
		planBytes, err := json.MarshalIndent(struct {
			Plan    *cudaregress.ExperimentPlan `json:"plan"`
			Receipt *cudaregress.PlanReceipt    `json:"receipt"`
		}{result.Plan, result.Receipt.PlanReceipt}, "", "  ")
		if err != nil {
			return err
		}
		if err := writeNewFile(filepath.Join(output, "plan.json"), append(planBytes, '\n')); err != nil {
			return err
		}
	}
	return nil
}

func writeNewFile(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	n, writeErr := f.Write(data)
	if writeErr == nil && n != len(data) {
		writeErr = io.ErrShortWrite
	}
	closeErr := f.Close()
	return errors.Join(writeErr, closeErr)
}
