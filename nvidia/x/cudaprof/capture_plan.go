package cudaprof

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

// CaptureProvider identifies an existing collection owner.
type CaptureProvider string

const (
	CaptureActivity    CaptureProvider = "activity"
	CaptureRange       CaptureProvider = "range"
	CapturePM          CaptureProvider = "pm"
	CapturePC          CaptureProvider = "pc"
	CaptureSASS        CaptureProvider = "sass"
	CapturePlanVersion                 = 1
)

// CaptureBudget bounds every capture dimension. Zero never means unlimited.
// MaxPasses reserves one execution per declared pass, plus each range result
// reports additional successful native starts beyond its first reserved start.
type CaptureBudget struct {
	MaxRecords   uint64 `json:"max_records"`
	MaxBytes     int    `json:"max_bytes"`
	MaxPasses    int    `json:"max_passes"`
	MaxWallNanos int64  `json:"max_wall_nanos"`
}

// CaptureEnvironment identifies the expected or observed collection context.
type CaptureEnvironment struct {
	Device            string `json:"device"`
	Driver            string `json:"driver"`
	CUDA              string `json:"cuda"`
	CUPTI             string `json:"cupti"`
	ClockMode         string `json:"clock_mode"`
	CorrelationDomain string `json:"correlation_domain"`
}

// CaptureProviderConfig identifies requested owner configuration. ConfigDigest
// is the exact configuration-image SHA-256, not a new workload identity.
// CaptureSASSMetric retains an actual owner metric ID and output granularity.
type CaptureSASSMetric struct {
	MetricID    uint64 `json:"metric_id"`
	Granularity uint8  `json:"granularity"`
}
type CaptureProviderConfig struct {
	SamplingInterval uint64              `json:"sampling_interval"`
	TriggerMode      int32               `json:"trigger_mode"`
	AppendMode       int32               `json:"append_mode"`
	SASSMetrics      []CaptureSASSMetric `json:"sass_metrics"`
	LazyPatching     bool                `json:"lazy_patching"`
	Metrics          []string            `json:"metrics"`
	ConfigDigest     string              `json:"config_digest"`
	Period           uint32              `json:"period"`
}

// CaptureNativePasses retains actual owner-reported native pass accounting.
type CaptureNativePasses struct {
	Started      uint64 `json:"started"`
	Stopped      uint64 `json:"stopped"`
	AllSubmitted bool   `json:"all_submitted"`
}

// CapturePass groups providers admitted within one declared replay.
type CapturePass struct {
	ID        string                                    `json:"id"`
	Replay    string                                    `json:"replay"`
	Providers []CaptureProvider                         `json:"providers"`
	Configs   map[CaptureProvider]CaptureProviderConfig `json:"configs"`
}

// CapturePlan is the opt-in request. Callers must not mutate it during use.
type CapturePlan struct {
	Version     int                   `json:"version"`
	Identity    cudaevidence.Identity `json:"identity"`
	Environment CaptureEnvironment    `json:"environment"`
	Budget      CaptureBudget         `json:"budget"`
	Passes      []CapturePass         `json:"passes"`
}

// CaptureCounts retains exact producer counters. Nil means unavailable.
type CaptureCounts struct {
	Retained *uint64 `json:"retained"`
	Offered  *uint64 `json:"offered"`
	Dropped  *uint64 `json:"dropped"`
	NonUser  *uint64 `json:"non_user"`
	Unnamed  *uint64 `json:"unnamed"`
}

// CaptureProviderResult retains one owner's result and byte-exact raw payload.
// A complete provider result does not establish cross-pass causality.
type CaptureProviderResult struct {
	Config          CaptureProviderConfig `json:"config"`
	NativePasses    *CaptureNativePasses  `json:"native_passes"`
	Provider        CaptureProvider       `json:"provider"`
	PassID          string                `json:"pass_id"`
	Replay          string                `json:"replay"`
	Scope           string                `json:"scope"`
	Process         uint64                `json:"process"`
	Generation      uint64                `json:"generation"`
	Context         string                `json:"context"`
	Identity        cudaevidence.Identity `json:"identity"`
	Environment     CaptureEnvironment    `json:"environment"`
	Start           uint64                `json:"start"`
	Stop            uint64                `json:"stop"`
	Status          string                `json:"status"`
	Reason          string                `json:"reason"`
	EvidenceClass   string                `json:"evidence_class"`
	Counts          CaptureCounts         `json:"counts"`
	Raw             []byte                `json:"raw"`
	Values          []cudaevidence.Value  `json:"values"`
	Decoded         bool                  `json:"decoded"`
	CleanupComplete bool                  `json:"cleanup_complete"`
}

// CaptureManifest records independent provider results against their request.
type CaptureManifest struct {
	Version      int                     `json:"version"`
	Plan         CapturePlan             `json:"plan"`
	Results      []CaptureProviderResult `json:"results"`
	ElapsedNanos int64                   `json:"elapsed_nanos"`
}

func captureProviderValid(p CaptureProvider) bool {
	switch p {
	case CaptureActivity, CaptureRange, CapturePM, CapturePC, CaptureSASS:
		return true
	}
	return false
}

func (e CaptureEnvironment) validate() error {
	for _, s := range []string{e.Device, e.Driver, e.CUDA, e.CUPTI, e.ClockMode, e.CorrelationDomain} {
		if strings.TrimSpace(s) == "" {
			return fmt.Errorf("capture: incomplete environment")
		}
	}
	return nil
}

// Validate checks request isolation, identity and finite admission budgets.
func (p CapturePlan) Validate() error {
	if p.Version != CapturePlanVersion {
		return fmt.Errorf("capture: unsupported plan version")
	}
	if err := p.Identity.Validate(); err != nil {
		return err
	}
	if err := p.Environment.validate(); err != nil {
		return err
	}
	b := p.Budget
	if b.MaxRecords == 0 || b.MaxRecords > 1<<32 || b.MaxBytes <= 0 || b.MaxBytes > 64<<20 || b.MaxPasses <= 0 || b.MaxPasses > 64 || b.MaxWallNanos <= 0 || b.MaxWallNanos > int64(24*time.Hour) {
		return fmt.Errorf("capture: invalid budget")
	}
	if len(p.Passes) == 0 || len(p.Passes) > b.MaxPasses {
		return fmt.Errorf("capture: pass budget exceeded")
	}
	ids := make(map[string]bool)
	for _, pass := range p.Passes {
		if strings.TrimSpace(pass.ID) == "" || strings.TrimSpace(pass.Replay) == "" || ids[pass.ID] || len(pass.Providers) == 0 {
			return fmt.Errorf("capture: invalid or duplicate pass")
		}
		ids[pass.ID] = true
		seen := make(map[CaptureProvider]bool)
		for _, provider := range pass.Providers {
			if !captureProviderValid(provider) || seen[provider] {
				return fmt.Errorf("capture: unknown or duplicate provider")
			}
			seen[provider] = true
			if err := validateCaptureConfig(provider, pass.Configs[provider], true); err != nil {
				return err
			}
			if provider != CaptureActivity && provider != CapturePC && len(pass.Providers) != 1 {
				return fmt.Errorf("capture: provider requires isolated pass")
			}
		}
		for provider := range pass.Configs {
			if !seen[provider] {
				return fmt.Errorf("capture: config for unrequested provider")
			}
		}
	}
	return nil
}

func validateCaptureConfig(provider CaptureProvider, c CaptureProviderConfig, required bool) error {
	if len(c.Metrics) > 128 {
		return fmt.Errorf("capture: too many metrics")
	}
	seen := make(map[string]bool)
	for _, metric := range c.Metrics {
		if strings.TrimSpace(metric) == "" || len(metric) > 256 || seen[metric] {
			return fmt.Errorf("capture: invalid or duplicate metric")
		}
		seen[metric] = true
	}
	if c.ConfigDigest != "" {
		if err := captureDigest(c.ConfigDigest); err != nil {
			return err
		}
	}
	if provider != CapturePM && (c.SamplingInterval != 0 || c.TriggerMode != 0 || c.AppendMode != 0) {
		return fmt.Errorf("capture: PM policy on another provider")
	}
	if provider != CaptureSASS && (len(c.SASSMetrics) > 0 || c.LazyPatching) {
		return fmt.Errorf("capture: SASS policy on another provider")
	}
	if len(c.SASSMetrics) > 128 {
		return fmt.Errorf("capture: too many SASS metrics")
	}
	seenIDs := make(map[uint64]bool)
	for _, metric := range c.SASSMetrics {
		if seenIDs[metric.MetricID] || metric.Granularity > 2 {
			return fmt.Errorf("capture: invalid SASS metric")
		}
		seenIDs[metric.MetricID] = true
	}
	if !required {
		return nil
	}
	if provider == CapturePM {
		if c.SamplingInterval == 0 || c.SamplingInterval > uint64(24*time.Hour) {
			return fmt.Errorf("capture: invalid PM sampling interval")
		}
		if c.TriggerMode != int32(cuptipmsampling.TriggerModeSysclkInterval) && c.TriggerMode != int32(cuptipmsampling.TriggerModeTimeInterval) {
			return fmt.Errorf("capture: invalid PM trigger mode")
		}
		if c.AppendMode != int32(cuptipmsampling.AppendModeKeepOldest) && c.AppendMode != int32(cuptipmsampling.AppendModeKeepLatest) {
			return fmt.Errorf("capture: invalid PM append mode")
		}
	}
	switch provider {
	case CaptureActivity:
		if len(c.Metrics) > 0 || c.ConfigDigest != "" || c.Period != 0 {
			return fmt.Errorf("capture: activity configuration must be empty")
		}
	case CapturePC:
		if len(c.Metrics) != 0 || c.ConfigDigest != "" {
			return fmt.Errorf("capture: unexpected PC metric configuration")
		}
		if c.Period < 5 || c.Period > 31 {
			return fmt.Errorf("capture: invalid PC period")
		}
	case CaptureSASS:
		if c.Period != 0 {
			return fmt.Errorf("capture: PC period on another provider")
		}
		if len(c.SASSMetrics) == 0 || c.ConfigDigest != "" {
			return fmt.Errorf("capture: SASS requires actual metric configuration without image digest")
		}
	case CaptureRange, CapturePM:
		if c.Period != 0 {
			return fmt.Errorf("capture: PC period on another provider")
		}
		if len(c.Metrics) == 0 || c.ConfigDigest == "" {
			return fmt.Errorf("capture: metrics and configuration image required")
		}
	}
	return nil
}
func captureDigest(d string) error {
	b, err := hex.DecodeString(d)
	if err != nil || len(b) != 32 || d != strings.ToLower(d) {
		return fmt.Errorf("capture: invalid supplied digest")
	}
	return nil
}
func captureConfigsEqual(a, b CaptureProviderConfig) bool {
	return a.ConfigDigest == b.ConfigDigest && a.Period == b.Period && a.SamplingInterval == b.SamplingInterval && a.TriggerMode == b.TriggerMode && a.AppendMode == b.AppendMode && a.LazyPatching == b.LazyPatching && slices.Equal(a.SASSMetrics, b.SASSMetrics) && slices.Equal(a.Metrics, b.Metrics)
}

func validateCaptureResult(r CaptureProviderResult) error {
	if !captureProviderValid(r.Provider) || strings.TrimSpace(r.PassID) == "" || strings.TrimSpace(r.Replay) == "" {
		return fmt.Errorf("capture: unknown provider or missing pass/replay")
	}
	if r.Status == "complete" {
		if err := r.Identity.Validate(); err != nil {
			return err
		}
		if err := r.Environment.validate(); err != nil {
			return err
		}
	} else {
		for _, d := range []string{r.Identity.ArtifactDigest, r.Identity.CompilerKey, r.Identity.WorkloadKey} {
			if d != "" {
				if err := captureDigest(d); err != nil {
					return err
				}
			}
		}
	}
	if err := validateCaptureConfig(r.Provider, r.Config, r.Status == "complete"); err != nil {
		return err
	}
	if n := r.NativePasses; n != nil {
		if n.Stopped > n.Started {
			return fmt.Errorf("capture: stopped native passes exceed started")
		}
	}
	if r.Stop < r.Start {
		return fmt.Errorf("capture: invalid provider boundaries")
	}
	for _, v := range r.Values {
		record := v.Record()
		if record.Source.Identity != r.Identity {
			return fmt.Errorf("capture: value identity mismatch")
		}
		if _, err := cudaevidence.New(record); err != nil {
			return err
		}
	}
	c := r.Counts
	if c.Offered != nil {
		for _, n := range []*uint64{c.Retained, c.Dropped, c.NonUser, c.Unnamed} {
			if n != nil && *n > *c.Offered {
				return fmt.Errorf("capture: count exceeds offered records")
			}
		}
		if c.Retained != nil && c.Dropped != nil && *c.Retained > *c.Offered-*c.Dropped {
			return fmt.Errorf("capture: retained and dropped exceed offered records")
		}
	}
	switch r.Status {
	case "complete":
		if r.Reason != "" || !r.Decoded || !r.CleanupComplete || r.Start == 0 || r.Stop <= r.Start || len(r.Raw) == 0 || strings.TrimSpace(r.Scope) == "" {
			return fmt.Errorf("capture: incomplete complete result")
		}
		switch r.Provider {
		case CapturePM:
			if r.Scope != "device-wide" {
				return fmt.Errorf("capture: PM scope is not device-wide")
			}
		case CaptureActivity:
			if r.Scope != "process" || r.Process == 0 || r.Generation == 0 {
				return fmt.Errorf("capture: activity ownership unavailable")
			}
		case CapturePC:
			if r.Scope != "target-context-first" || r.Process == 0 || r.Generation == 0 || r.Context == "" {
				return fmt.Errorf("capture: first target context ownership unavailable")
			}
		default:
			if r.Scope != "target-context" || r.Process == 0 || r.Generation == 0 || r.Context == "" {
				return fmt.Errorf("capture: target context ownership unavailable")
			}
		}
		if r.Provider == CaptureRange {
			n := r.NativePasses
			if n == nil || n.Started == 0 || n.Stopped != n.Started || !n.AllSubmitted {
				return fmt.Errorf("capture: range native passes incomplete")
			}
		}
		if r.EvidenceClass != "measured" && r.EvidenceClass != "derived" {
			return fmt.Errorf("capture: invalid complete evidence class")
		}
		if c.Retained == nil || *c.Retained == 0 || c.Offered == nil || *c.Retained != *c.Offered || c.Dropped == nil || *c.Dropped != 0 || c.NonUser == nil || *c.NonUser != 0 || c.Unnamed == nil || *c.Unnamed != 0 {
			return fmt.Errorf("capture: complete result lacks loss-free counters")
		}
	case "partial", "unavailable", "unsupported", "failed":
		if strings.TrimSpace(r.Reason) == "" {
			return fmt.Errorf("capture: noncomplete result lacks reason")
		}
		if r.EvidenceClass != "" && r.EvidenceClass != "unread" && r.EvidenceClass != "measured" && r.EvidenceClass != "derived" && r.EvidenceClass != "inferred" && r.EvidenceClass != "modeled" {
			return fmt.Errorf("capture: unknown evidence class")
		}
	default:
		return fmt.Errorf("capture: unknown result status")
	}
	return nil
}

// Validate checks final receipts, not runtime budget enforcement. It requires
// exactly one result per requested provider and pass.
func (m CaptureManifest) Validate() error {
	if m.Version != CapturePlanVersion {
		return fmt.Errorf("capture: unsupported manifest version")
	}
	if err := m.Plan.Validate(); err != nil {
		return err
	}
	if m.ElapsedNanos < 0 {
		return fmt.Errorf("capture: wall budget exceeded")
	}
	type key struct {
		pass     string
		provider CaptureProvider
	}
	expected := make(map[key]string)
	for _, p := range m.Plan.Passes {
		for _, provider := range p.Providers {
			expected[key{p.ID, provider}] = p.Replay
		}
	}
	if len(m.Results) != len(expected) {
		return fmt.Errorf("capture: missing provider results")
	}
	var offered uint64
	nativePasses := uint64(len(m.Plan.Passes))
	overWall := m.ElapsedNanos > m.Plan.Budget.MaxWallNanos
	for _, r := range m.Results {
		k := key{r.PassID, r.Provider}
		replay, ok := expected[k]
		if !ok || r.Replay != replay {
			return fmt.Errorf("capture: duplicate or mismatched pass/provider/replay")
		}
		delete(expected, k)
		if r.Status == "complete" && (r.Identity != m.Plan.Identity || r.Environment != m.Plan.Environment) {
			return fmt.Errorf("capture: result identity or environment mismatch")
		}
		if overWall && (r.Status == "complete" || !strings.Contains(strings.ToLower(r.Reason), "budget")) {
			return fmt.Errorf("capture: wall budget exceeded without explicit incomplete results")
		}
		var expectedConfig CaptureProviderConfig
		for _, pass := range m.Plan.Passes {
			if pass.ID == r.PassID {
				expectedConfig = pass.Configs[r.Provider]
			}
		}
		if r.Status == "complete" && !captureConfigsEqual(expectedConfig, r.Config) {
			return fmt.Errorf("capture: provider configuration mismatch")
		}
		if n := r.NativePasses; n != nil {
			extra := uint64(0)
			if n.Started > 1 {
				extra = n.Started - 1
			}
			if extra > uint64(m.Plan.Budget.MaxPasses)-nativePasses {
				return fmt.Errorf("capture: native pass budget exceeded")
			}
			nativePasses += extra
		}
		if err := validateCaptureResult(r); err != nil {
			return err
		}
		var records uint64
		for _, count := range []*uint64{r.Counts.Offered, r.Counts.Retained, r.Counts.Dropped, r.Counts.NonUser, r.Counts.Unnamed} {
			if count != nil {
				records = max(records, *count)
			}
		}
		if records > 0 {
			if records > m.Plan.Budget.MaxRecords-offered {
				return fmt.Errorf("capture: record budget exceeded")
			}
			offered += records
		}
	}
	return nil
}

// Complete means every requested result is individually complete. It does not
// assert simultaneous observations or causal relationships between passes.
func (m CaptureManifest) Complete() bool {
	if m.Validate() != nil {
		return false
	}
	for _, r := range m.Results {
		if r.Status != "complete" {
			return false
		}
	}
	return true
}

// CheckCaptureJoin refuses mismatched identities and unexplained cross-pass joins.
// A supplied basis records the interpretation; it does not prove causality.
func CheckCaptureJoin(a, b CaptureProviderResult, basis string) error {
	if err := validateCaptureResult(a); err != nil {
		return err
	}
	if err := validateCaptureResult(b); err != nil {
		return err
	}
	if a.Status != "complete" || b.Status != "complete" {
		return fmt.Errorf("capture join: incomplete ownership or observations")
	}
	if a.Identity != b.Identity || a.Environment != b.Environment {
		return fmt.Errorf("capture join: identity or environment mismatch")
	}
	if a.Scope == "device-wide" || b.Scope == "device-wide" {
		if strings.TrimSpace(basis) == "" {
			return fmt.Errorf("capture join: device-wide basis required")
		}
	} else {
		if a.Process != b.Process || a.Generation != b.Generation || (a.Context != "" && b.Context != "" && a.Context != b.Context) {
			return fmt.Errorf("capture join: owner mismatch")
		}
	}
	if (a.Context == "" || b.Context == "") && strings.TrimSpace(basis) == "" {
		return fmt.Errorf("capture join: context attribution basis required")
	}
	if a.PassID == b.PassID && (a.Stop <= b.Start || b.Stop <= a.Start) && strings.TrimSpace(basis) == "" {
		return fmt.Errorf("capture join: disjoint intervals require basis")
	}
	if a.PassID == b.PassID {
		if a.Replay != b.Replay {
			return fmt.Errorf("capture join: replay mismatch")
		}
	} else if strings.TrimSpace(basis) == "" {
		return fmt.Errorf("capture join: cross-pass basis required")
	}
	return nil
}

// WriteCaptureManifest validates and bounds the complete encoding before writing.
func WriteCaptureManifest(w io.Writer, m CaptureManifest) error {
	if err := m.Plan.Validate(); err != nil {
		return err
	}
	if err := captureEncodingPreflight(m, m.Plan.Budget.MaxBytes); err != nil {
		return err
	}
	if err := m.Validate(); err != nil {
		return err
	}
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if len(data) > m.Plan.Budget.MaxBytes {
		return fmt.Errorf("capture: byte budget exceeded")
	}
	n, err := w.Write(data)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	return err
}

// ReadCapturePlan reads a bounded, strict plan into independent owned slices.
func ReadCapturePlan(r io.Reader, maxBytes int) (CapturePlan, error) {
	var p CapturePlan
	err := readCaptureJSON(r, maxBytes, &p, nil)
	if err == nil {
		err = p.Validate()
	}
	return p, err
}

// ReadCaptureManifest reads a bounded, strict manifest and validates its semantics.
func ReadCaptureManifest(r io.Reader, maxBytes int) (CaptureManifest, error) {
	var m CaptureManifest
	var encodedBytes int
	err := readCaptureJSON(r, maxBytes, &m, &encodedBytes)
	if err == nil {
		err = m.Validate()
	}
	if err == nil && encodedBytes > m.Plan.Budget.MaxBytes {
		err = fmt.Errorf("capture: original byte budget exceeded")
	}
	if err == nil {
		data, e := json.Marshal(m)
		err = e
		if err == nil && len(data) > m.Plan.Budget.MaxBytes {
			err = fmt.Errorf("capture: byte budget exceeded")
		}
	}
	return m, err
}

func readCaptureJSON(r io.Reader, maxBytes int, out any, encodedBytes *int) error {
	if maxBytes <= 0 || maxBytes > 64<<20 {
		return fmt.Errorf("capture: invalid read byte limit")
	}
	data, err := io.ReadAll(io.LimitReader(r, int64(maxBytes)+1))
	if err != nil {
		return err
	}
	if len(data) > maxBytes {
		return fmt.Errorf("capture: read byte budget exceeded")
	}
	if encodedBytes != nil {
		*encodedBytes = len(data)
	}
	if err := captureJSONKeys(json.NewDecoder(bytes.NewReader(data))); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	var tail any
	if d.Decode(&tail) != io.EOF {
		return fmt.Errorf("capture: trailing JSON")
	}
	return nil
}

func captureJSONKeys(d *json.Decoder) error {
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			t, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := t.(string)
			if !ok || seen[key] {
				return fmt.Errorf("capture: duplicate JSON key")
			}
			seen[key] = true
			for _, c := range key {
				if !(c >= 'a' && c <= 'z' || c == '_') {
					return fmt.Errorf("capture: noncanonical JSON key")
				}
			}
			if err := captureJSONKeys(d); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := captureJSONKeys(d); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("capture: invalid JSON delimiter")
	}
	_, err = d.Token()
	return err
}

// captureEncodingPreflight bounds caller-owned variable-sized data before JSON
// allocation. Final encoding still receives the exact byte-budget check.
func captureEncodingPreflight(m CaptureManifest, limit int) error {
	remaining := limit
	var visit func(reflect.Value) bool
	visit = func(v reflect.Value) bool {
		switch v.Kind() {
		case reflect.String:
			if v.Len() > remaining {
				return false
			}
			remaining -= v.Len()
		case reflect.Pointer:
			if !v.IsNil() {
				return visit(v.Elem())
			}
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				if !visit(v.Field(i)) {
					return false
				}
			}
		case reflect.Slice:
			if v.Type().Elem().Kind() == reflect.Uint8 {
				if v.Len() > remaining {
					return false
				}
				remaining -= v.Len()
				return true
			}
			if v.Len() > remaining {
				return false
			}
			remaining -= v.Len()
			for i := 0; i < v.Len(); i++ {
				if !visit(v.Index(i)) {
					return false
				}
			}
		case reflect.Map:
			if v.Len() > remaining {
				return false
			}
			remaining -= v.Len()
			iter := v.MapRange()
			for iter.Next() {
				if !visit(iter.Key()) || !visit(iter.Value()) {
					return false
				}
			}
		}
		return true
	}
	if !visit(reflect.ValueOf(m)) {
		return fmt.Errorf("capture: encoded data exceeds preflight byte budget")
	}
	return nil
}
