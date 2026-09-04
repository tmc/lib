package inferbench

import (
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"time"
)

// LinkedReceiptVersion is the frozen version of the linked receipt contract.
// A new version requires a retained receipt that this version accepts falsely.
const LinkedReceiptVersion = 1

// LinkedReceipts separates the proofs needed for a performance result. Resource
// receipts gate publication, not collection or validation of timing samples.
// Diagnostics are optional and support only claims about the quantities named
// by them.
type LinkedReceipts struct {
	Version       int                    `json:"version"`
	Manifest      ExperimentManifest     `json:"manifest"`
	Qualification QualificationReceipt   `json:"qualification"`
	Correctness   []CorrectnessReceipt   `json:"correctness"`
	Performance   []PerformanceReceipt   `json:"performance"`
	Resources     []ResourceReceipt      `json:"resources,omitempty"`
	Diagnostics   []DiagnosticReceiptRef `json:"diagnostics,omitempty"`
}

// ExperimentManifest is the immutable, preregistered experiment contract.
type ExperimentManifest struct {
	ID               string            `json:"id"`
	Arms             []ExperimentArm   `json:"arms"`
	Workload         Workload          `json:"workload"`
	RandomSeed       uint64            `json:"random_seed"`
	Order            []ScheduledSample `json:"order"`
	TimingBoundary   string            `json:"timing_boundary"`
	RunnerLease      string            `json:"runner_lease"`
	PrimaryMetric    string            `json:"primary_metric"`
	PrimaryThreshold float64           `json:"primary_threshold"`
	SecondaryLimit   float64           `json:"secondary_regression_limit"`
}

// ExperimentArm names one exact executable and source identity.
type ExperimentArm struct {
	Name      string   `json:"name"`
	Identity  Identity `json:"identity"`
	SampleIDs []string `json:"sample_ids"`
}

// QualificationReceipt proves the execution environment used by a manifest.
type QualificationReceipt struct {
	ManifestSHA256   string          `json:"manifest_sha256"`
	Backend          string          `json:"backend"`
	DeviceID         string          `json:"device_id"`
	DeviceSHA256     string          `json:"device_sha256"`
	NoCPUFallback    bool            `json:"no_cpu_fallback"`
	DeviceResidentKV bool            `json:"device_resident_kv"`
	LeaseHeld        bool            `json:"lease_held"`
	Sentinels        []time.Duration `json:"sentinel_ns"`
	SentinelLimit    time.Duration   `json:"sentinel_limit_ns"`
	Periodic         []Sentinel      `json:"periodic_sentinels"`
}

// Sentinel records a stability probe after a scheduled interval.
type Sentinel struct {
	AfterIndex int           `json:"after_index"`
	Duration   time.Duration `json:"duration_ns"`
}

// CorrectnessReceipt proves one arm and shape against a strong semantic oracle.
// It is collected outside the timed workload.
type CorrectnessReceipt struct {
	ManifestSHA256    string    `json:"manifest_sha256"`
	Arm               string    `json:"arm"`
	Generated         [][]int32 `json:"generated_token_ids"`
	FullLogitsSHA256  string    `json:"full_logits_sha256"`
	PopulatedKVSHA256 string    `json:"populated_kv_sha256"`
	Synchronized      bool      `json:"synchronized"`
}

// PerformanceReceipt contains one raw, synchronized serving interval.
type PerformanceReceipt struct {
	ManifestSHA256 string          `json:"manifest_sha256"`
	Index          int             `json:"index"`
	Arm            string          `json:"arm"`
	ID             string          `json:"id"`
	StartedAt      time.Time       `json:"started_at"`
	EndedAt        time.Time       `json:"ended_at"`
	Requests       []RequestTiming `json:"requests"`
	Generated      [][]int32       `json:"generated_token_ids"`
	Synchronized   bool            `json:"synchronized"`
	Status         string          `json:"status"`
	RawFile        string          `json:"raw_file"`
	RawSHA256      string          `json:"raw_sha256"`
}

// ResourceReceipt records a separately collected resource envelope.
type ResourceReceipt struct {
	ManifestSHA256    string `json:"manifest_sha256"`
	Arm               string `json:"arm"`
	PersistentBytes   uint64 `json:"persistent_bytes"`
	PeakBytes         uint64 `json:"peak_bytes"`
	RequestBytes      uint64 `json:"request_bytes"`
	AfterReleaseBytes uint64 `json:"after_release_bytes"`
	AfterCloseBytes   uint64 `json:"after_close_bytes"`
	Coverage          string `json:"coverage"`
}

// DiagnosticReceiptRef links optional architecture evidence to the experiment.
type DiagnosticReceiptRef struct {
	ManifestSHA256 string `json:"manifest_sha256"`
	Kind           string `json:"kind"`
	SHA256         string `json:"sha256"`
}

// ValidatePerformance validates the proofs needed to interpret timing. It does
// not require resource or architecture-diagnostic receipts.
func (r *LinkedReceipts) ValidatePerformance() error {
	sum, arms, err := r.validateManifest()
	if err != nil {
		return err
	}
	q := r.Qualification
	first := r.Manifest.Arms[0].Identity
	if q.ManifestSHA256 != sum || q.Backend != first.Backend || q.DeviceID != first.DeviceID || !validSHA256(q.DeviceSHA256) || !q.NoCPUFallback || !q.DeviceResidentKV || !q.LeaseHeld || len(q.Sentinels) < 20 || q.SentinelLimit <= 0 {
		return errors.New("qualification receipt incomplete")
	}
	for _, d := range q.Sentinels {
		if d <= 0 || d > q.SentinelLimit {
			return errors.New("qualification sentinel outside limit")
		}
	}
	lastSentinel := -1
	for _, sentinel := range q.Periodic {
		if sentinel.AfterIndex <= lastSentinel || sentinel.AfterIndex >= len(r.Manifest.Order) || sentinel.AfterIndex-lastSentinel > 10 || sentinel.Duration <= 0 || sentinel.Duration > q.SentinelLimit {
			return errors.New("periodic qualification sentinel incomplete")
		}
		lastSentinel = sentinel.AfterIndex
	}
	if lastSentinel != len(r.Manifest.Order)-1 {
		return errors.New("periodic qualification sentinels do not cover campaign")
	}
	correct := make(map[string][][]int32)
	correctState := make(map[string][2]string)
	for _, c := range r.Correctness {
		if c.ManifestSHA256 != sum || !arms[c.Arm] || !c.Synchronized || !validSHA256(c.FullLogitsSHA256) || !validSHA256(c.PopulatedKVSHA256) || len(c.Generated) != r.Manifest.Workload.Concurrency {
			return fmt.Errorf("arm %q: correctness receipt incomplete", c.Arm)
		}
		if _, dup := correct[c.Arm]; dup {
			return fmt.Errorf("arm %q: duplicate correctness receipt", c.Arm)
		}
		correct[c.Arm] = c.Generated
		correctState[c.Arm] = [2]string{c.FullLogitsSHA256, c.PopulatedKVSHA256}
	}
	if len(correct) != len(arms) {
		return errors.New("correctness receipts do not cover every arm")
	}
	var oracle [][]int32
	var oracleState [2]string
	for _, a := range r.Manifest.Arms {
		got := correct[a.Name]
		if oracle == nil {
			oracle = got
			oracleState = correctState[a.Name]
		} else if !reflect.DeepEqual(oracle, got) || oracleState != correctState[a.Name] {
			return errors.New("correctness outputs differ")
		}
	}
	if len(r.Performance) != len(r.Manifest.Order) {
		return errors.New("performance receipts do not cover schedule")
	}
	counts := make(map[string]int)
	files := make(map[string]bool)
	var last time.Time
	for i, p := range r.Performance {
		want := r.Manifest.Order[i]
		if p.ManifestSHA256 != sum || p.Index != i || p.Arm != want.Arm || p.ID != want.ID || !arms[p.Arm] || !p.Synchronized || p.Status != "completed" || p.RawFile == "" || filepath.Base(p.RawFile) != p.RawFile || files[p.RawFile] || !validSHA256(p.RawSHA256) || p.StartedAt.IsZero() || !p.EndedAt.After(p.StartedAt) || len(p.Requests) != r.Manifest.Workload.Concurrency || len(p.Generated) != r.Manifest.Workload.Concurrency {
			return fmt.Errorf("performance receipt %d incomplete", i)
		}
		files[p.RawFile] = true
		if !last.IsZero() && p.StartedAt.Before(last) {
			return fmt.Errorf("performance receipt %d overlaps or violates schedule", i)
		}
		last = p.EndedAt
		for j := range p.Requests {
			request := p.Requests[j]
			if request.StartedAt.Before(p.StartedAt) || request.EndedAt.After(p.EndedAt) || request.EndedAt.Before(request.StartedAt) || len(request.TokenAt) != r.Manifest.Workload.GeneratedTokens || len(p.Generated[j]) != r.Manifest.Workload.GeneratedTokens {
				return fmt.Errorf("performance receipt %d has invalid request", i)
			}
			at := request.StartedAt
			for _, tokenAt := range request.TokenAt {
				if tokenAt.Before(at) || tokenAt.After(request.EndedAt) {
					return fmt.Errorf("performance receipt %d has invalid token time", i)
				}
				at = tokenAt
			}
		}
		if !reflect.DeepEqual(p.Generated, oracle) {
			return fmt.Errorf("performance receipt %d differs from correctness output", i)
		}
		counts[p.Arm]++
	}
	for arm := range arms {
		if counts[arm] < 20 {
			return fmt.Errorf("arm %q: only %d performance samples", arm, counts[arm])
		}
	}
	for _, diagnostic := range r.Diagnostics {
		if diagnostic.ManifestSHA256 != sum || diagnostic.Kind == "" || !validSHA256(diagnostic.SHA256) {
			return errors.New("invalid optional diagnostic receipt")
		}
	}
	return nil
}

// ValidatePerformanceRaw also verifies that every retained raw artifact has
// the hash declared by its performance receipt.
func (r *LinkedReceipts) ValidatePerformanceRaw(dir string) error {
	if err := r.ValidatePerformance(); err != nil {
		return err
	}
	for i, receipt := range r.Performance {
		sum, err := HashFile(filepath.Join(dir, receipt.RawFile))
		if err != nil || sum != receipt.RawSHA256 {
			return fmt.Errorf("performance receipt %d raw artifact differs", i)
		}
	}
	return nil
}

// ValidatePublication verifies retained timing artifacts and additionally
// requires a complete, matched resource envelope. Optional diagnostics never
// become a publication prerequisite.
func (r *LinkedReceipts) ValidatePublication(rawDir string) error {
	if err := r.ValidatePerformanceRaw(rawDir); err != nil {
		return err
	}
	sum, arms, _ := r.validateManifest()
	seen := make(map[string]bool)
	for _, resource := range r.Resources {
		if resource.ManifestSHA256 != sum || !arms[resource.Arm] || seen[resource.Arm] || resource.PersistentBytes == 0 || resource.PeakBytes < resource.PersistentBytes || resource.AfterReleaseBytes != resource.PersistentBytes || resource.AfterCloseBytes != 0 || resource.Coverage == "" {
			return fmt.Errorf("arm %q: resource receipt incomplete", resource.Arm)
		}
		seen[resource.Arm] = true
	}
	if len(seen) != len(arms) {
		return errors.New("resource receipts do not cover every arm")
	}
	return nil
}

func (r *LinkedReceipts) validateManifest() (string, map[string]bool, error) {
	m := r.Manifest
	if r.Version != LinkedReceiptVersion || m.ID == "" || m.RandomSeed == 0 || len(m.Arms) < 2 || len(m.Order) < 40 || m.TimingBoundary != "submit-to-final-device-synchronize" || m.RunnerLease == "" || m.PrimaryMetric == "" || m.PrimaryThreshold <= 0 || m.SecondaryLimit <= 0 {
		return "", nil, errors.New("experiment manifest incomplete")
	}
	if err := validateLinkedWorkload(m.Workload); err != nil {
		return "", nil, err
	}
	arms := make(map[string]bool)
	var plans []Run
	var common Identity
	for i, arm := range m.Arms {
		if arm.Name == "" || arms[arm.Name] {
			return "", nil, errors.New("invalid experiment arm")
		}
		arms[arm.Name] = true
		if len(arm.SampleIDs) < 20 {
			return "", nil, fmt.Errorf("arm %q: only %d scheduled samples", arm.Name, len(arm.SampleIDs))
		}
		ids := make(map[string]bool)
		plan := Run{Arm: arm.Name}
		for _, id := range arm.SampleIDs {
			if id == "" || ids[id] {
				return "", nil, fmt.Errorf("arm %q: invalid sample id", arm.Name)
			}
			ids[id] = true
			plan.Samples = append(plan.Samples, Sample{ID: id})
		}
		plans = append(plans, plan)
		if err := validateLinkedIdentity(arm.Identity); err != nil {
			return "", nil, fmt.Errorf("arm %q: %w", arm.Name, err)
		}
		if i == 0 {
			common = arm.Identity
		} else if arm.Identity.ModelSHA256 != common.ModelSHA256 || arm.Identity.TokenizerSHA256 != common.TokenizerSHA256 || arm.Identity.ConfigSHA256 != common.ConfigSHA256 || arm.Identity.Backend != common.Backend || arm.Identity.Precision != common.Precision || arm.Identity.DeviceID != common.DeviceID {
			return "", nil, fmt.Errorf("arm %q: unequal semantic or device identity", arm.Name)
		}
	}
	if want := Schedule(plans, m.RandomSeed); !reflect.DeepEqual(want, m.Order) {
		return "", nil, errors.New("experiment schedule is not the seeded randomized order")
	}
	sum, err := HashJSON(m)
	return sum, arms, err
}

func validateLinkedIdentity(id Identity) error {
	for _, sum := range []string{id.BinarySHA256, id.ModelSHA256, id.TokenizerSHA256, id.ConfigSHA256, id.SourceSHA256} {
		if !validSHA256(sum) {
			return errors.New("invalid identity hash")
		}
	}
	if id.Backend != "metal" && id.Backend != "cuda" || id.Precision == "" || id.DeviceID == "" {
		return errors.New("incomplete identity")
	}
	return nil
}

func validateLinkedWorkload(w Workload) error {
	if len(w.PromptTokenIDs) == 0 || len(w.PromptTokenIDs) != w.Concurrency || w.ContextTokens <= 0 || w.GeneratedTokens < 2 || w.Concurrency <= 0 || w.WarmupRuns <= 0 || w.Sampler != "greedy-argmax" || w.CacheState == "" || w.OutputMode != ThroughputOutputMode {
		return errors.New("incomplete performance workload")
	}
	for _, prompt := range w.PromptTokenIDs {
		if len(prompt) != w.ContextTokens {
			return errors.New("prompt token count differs from context tokens")
		}
	}
	return nil
}
