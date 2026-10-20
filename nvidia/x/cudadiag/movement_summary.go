package cudadiag

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// MovementInput names the movement context and the existing evidence it is
// checked against. Activities, Batch, Annotations and Receipt feed
// AnalyzeActivityJSONL. Health, when present, is a health snapshot whose
// topology edges are checked against declared NVLink and PCIe paths.
// Telemetry, when present, is a canonical stream analyzed by
// DiagnoseTelemetry; its admitted observations become supporting symptoms.
// Artifacts holds raw path evidence by SHA-256.
type MovementInput struct {
	Movement    io.Reader
	Source      nvtelemetry.WorkloadSource
	Limits      nvtelemetry.WorkloadLimits
	Activities  io.Reader
	Batch       cudaworkload.ActivityBatch
	Annotations []cudaworkload.HostSubmission
	Receipt     tracecontract.Receipt

	Health       io.Reader
	HealthLimits nvtelemetry.HealthLimits

	Telemetry        io.Reader
	TelemetryOptions TelemetryOptions

	Artifacts map[string][]byte
	// MaxUncertaintyNS bounds transfer clock uncertainty for duration and
	// symptom claims. Zero means 50.
	MaxUncertaintyNS int64
}

// MovementTransferSummary is the evaluated evidence for one supplied
// transfer. Bytes is the exact supplied count, present only when the bytes
// claim is admitted. ActivityIndex indexes the exactly linked activity join.
// Symptoms lists admitted telemetry finding IDs on a transfer endpoint during
// the transfer; a symptom is not a cause.
type MovementTransferSummary struct {
	ID            string                  `json:"id"`
	Family        string                  `json:"family"`
	RawSHA256     string                  `json:"raw_sha256"`
	RequestID     string                  `json:"request_id,omitempty"`
	Attempt       string                  `json:"attempt,omitempty"`
	OpID          cudaworkload.OpID       `json:"op_id,omitempty"`
	Bytes         *uint64                 `json:"bytes,omitempty"`
	ActivityIndex *int                    `json:"activity_index,omitempty"`
	Symptoms      []string                `json:"symptoms"`
	Findings      []tracecontract.Finding `json:"findings"`
}

// MovementFamily counts supplied transfer states in one loss family.
// Qualified reports known zero loss; otherwise absent transfers and
// completions in the family are unassessed rather than known not to exist.
type MovementFamily struct {
	Family    string `json:"family"`
	Total     uint64 `json:"total"`
	Completed uint64 `json:"completed"`
	Failed    uint64 `json:"failed"`
	Cancelled uint64 `json:"cancelled"`
	Pending   uint64 `json:"pending"`
	Qualified bool   `json:"qualified"`
	Reason    string `json:"reason"`
}

// MovementSummary is a versioned movement incident summary. It retains the
// activity analysis and optional telemetry diagnosis it consumed. Each
// transfer claim is observed, associated or withheld independently, so
// missing evidence for one claim does not discard the others.
type MovementSummary struct {
	Version       int                         `json:"version"`
	ContextSHA256 string                      `json:"context_sha256"`
	Source        nvtelemetry.WorkloadSource  `json:"source"`
	Context       nvtelemetry.MovementContext `json:"context"`
	Activities    ActivityAnalysis            `json:"activities"`
	Telemetry     *TelemetryDiagnosis         `json:"telemetry,omitempty"`
	TopologyEpoch string                      `json:"topology_epoch,omitempty"`
	Transfers     []MovementTransferSummary   `json:"transfers"`
	Families      []MovementFamily            `json:"families"`
	Limits        []string                    `json:"limits"`
}

// Movement claim rules.
const (
	RuleMovementBytes      = "movement_bytes"
	RuleMovementDirection  = "movement_direction"
	RuleMovementCompletion = "movement_completion"
	RuleMovementDuration   = "movement_duration"
	RuleMovementPath       = "movement_path"
	RuleMovementActivity   = "movement_activity_link"
	RuleMovementAttempt    = "movement_attempt_link"
	RuleMovementCongestion = "movement_congestion"
)

// SummarizeMovement reads a movement context, runs the existing activity
// analysis and, when supplied, the health topology reader and telemetry
// diagnosis, and evaluates each transfer's claims. Path claims require
// retained evidence. An NVLink claim also requires an available
// nvlink_remote_pci edge from source to destination, valid over the
// transfer widened by its clock uncertainty, in the same topology epoch and
// clock; PCIe and NVSwitch topology is not supported and stays withheld.
// Storage, staged and RDMA modes require an independent mechanism artifact.
// Activity links require the same run, one copy and matching bytes, kinds
// and direction. CUDA activity never establishes a transfer path.
// No throughput, rate or reset-crossing delta is computed, and no congestion
// rule is defined: latency, bytes and symptoms never admit congestion.
func SummarizeMovement(ctx context.Context, in MovementInput) (MovementSummary, error) {
	var out MovementSummary
	if ctx == nil || in.Movement == nil || in.Activities == nil {
		return out, fmt.Errorf("movement summary: missing context or reader")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	export, c, err := nvtelemetry.ReadMovementContext(telemetryReader{ctx, in.Movement}, in.Source, in.Limits)
	if err != nil {
		return out, fmt.Errorf("movement summary context: %w", err)
	}
	activities, err := AnalyzeActivityJSONL(telemetryReader{ctx, in.Activities}, in.Batch, in.Annotations, in.Receipt)
	if err != nil {
		return out, fmt.Errorf("movement summary activities: %w", err)
	}
	if len(in.Artifacts) > 256 {
		return out, fmt.Errorf("movement summary: too many artifacts")
	}
	var artifactBytes int
	for hash, data := range in.Artifacts {
		artifactBytes += len(data)
		sum := sha256.Sum256(data)
		if artifactBytes > 64<<20 || hex.EncodeToString(sum[:]) != hash {
			return out, fmt.Errorf("movement summary: artifact %q exceeds bounds or digest", hash)
		}
	}
	var health *nvtelemetry.HealthSnapshot
	if in.Health != nil {
		health, err = nvtelemetry.ReadHealthSnapshot(telemetryReader{ctx, in.Health}, in.HealthLimits)
		if err != nil {
			return out, fmt.Errorf("movement summary health: %w", err)
		}
	}
	if in.Telemetry != nil {
		d, err := DiagnoseTelemetry(ctx, in.Telemetry, in.TelemetryOptions)
		if err != nil {
			return out, fmt.Errorf("movement summary telemetry: %w", err)
		}
		out.Telemetry = &d
	}
	maxUnc := in.MaxUncertaintyNS
	if maxUnc == 0 {
		maxUnc = 50
	}
	if maxUnc < 0 {
		return out, fmt.Errorf("movement summary: negative uncertainty bound")
	}
	out.Version = 1
	out.ContextSHA256 = export.RawSHA256
	out.Source = export.Source
	out.Context = c
	out.Activities = activities
	if health != nil {
		out.TopologyEpoch = health.TopologyEpoch
	}
	out.Limits = []string{
		"caller-supplied movement context; producer authenticity unverified",
		"no throughput, rate or reset-crossing delta is computed",
		"topology edges describe permitted links, not executed routes",
		"CUDA memcpy activity cannot establish a storage, direct, fallback or fabric path",
		"no congestion rule is defined; latency, bytes and symptoms do not establish congestion",
		"activity links use the existing external correlation reader, which does not match peer-to-peer memcpy2 records",
	}
	e := movementEvaluator{in: in, c: c, activities: activities, health: health, telemetry: out.Telemetry, maxUnc: maxUnc}
	for _, t := range c.Transfers {
		if err := ctx.Err(); err != nil {
			return MovementSummary{}, err
		}
		out.Transfers = append(out.Transfers, e.transfer(t))
	}
	out.Families = movementFamilies(c)
	if c.SourceClass == "modeled" {
		out.Limits = append(out.Limits, "modeled source: no transfer claim admitted")
	}
	return out, nil
}

type movementEvaluator struct {
	in         MovementInput
	c          nvtelemetry.MovementContext
	activities ActivityAnalysis
	health     *nvtelemetry.HealthSnapshot
	telemetry  *TelemetryDiagnosis
	maxUnc     int64
}

func (e movementEvaluator) transfer(t nvtelemetry.MovementTransfer) MovementTransferSummary {
	s := MovementTransferSummary{ID: t.ID, Family: t.Family, RawSHA256: t.RawSHA256, RequestID: t.RequestID, Attempt: t.Attempt, OpID: t.OpID, Symptoms: []string{}}
	class := tracecontract.FindingAssociated
	if e.c.SourceClass == "native" {
		class = tracecontract.FindingObserved
	}
	modeled := e.c.SourceClass == "modeled"
	add := func(rule string, class tracecontract.FindingClass, msg string, value *float64, unit string, intervals []tracecontract.Interval, code, reason string) {
		if modeled {
			code, reason = tracecontract.ReasonBackendUnsupported, "modeled movement source"
		}
		s.Findings = append(s.Findings, movementFinding(t, rule, class, msg, value, unit, intervals, code, reason))
	}

	join, candidate, linkCode, linkReason := e.activityLink(t)
	if join >= 0 && !modeled {
		s.ActivityIndex = &join
	}

	code, reason := "", ""
	if t.Bytes == nil {
		code, reason = tracecontract.ReasonEvidenceUnavailable, "bytes not reported"
	} else if candidate >= 0 && e.activities.Joins[candidate].Record.Bytes != *t.Bytes {
		code, reason = tracecontract.ReasonIdentityMismatch, "linked activity bytes contradict supplied bytes"
	}
	if code == "" && !modeled {
		b := *t.Bytes
		s.Bytes = &b
	}
	add(RuleMovementBytes, class, "supplied exact transfer bytes; not interval throughput", nil, "bytes", nil, code, reason)

	direction := movementDirection(t)
	code, reason = "", ""
	if !direction {
		code, reason = tracecontract.ReasonIdentityMismatch, "direction contradicts endpoint kinds or identities"
	}
	add(RuleMovementDirection, class, "supplied direction consistent with endpoint kinds", nil, "", nil, code, reason)

	completed := t.State == "completed"
	code, reason = "", ""
	switch {
	case !completed:
		code, reason = tracecontract.ReasonEvidenceUnavailable, "no completion record; enqueue or async return is not completion"
	case join >= 0 && !e.activities.Joins[join].Record.Complete:
		code, reason = tracecontract.ReasonIdentityMismatch, "linked activity is incomplete"
	}
	completed = code == ""
	add(RuleMovementCompletion, class, "producer-reported transfer completion", nil, "", nil, code, reason)

	start := t.StartNanos
	msg := "start-to-completion duration in the transfer clock"
	if start == nil {
		start = t.EnqueueNanos
		msg = "enqueue-to-completion duration in the transfer clock; includes queueing"
	}
	var value *float64
	var intervals []tracecontract.Interval
	code, reason = "", ""
	switch {
	case !completed:
		code, reason = tracecontract.ReasonEvidenceUnavailable, "completion not admitted"
	case start == nil:
		code, reason = tracecontract.ReasonEvidenceUnavailable, "no start or enqueue time"
	case t.UncertaintyNS == nil:
		code, reason = tracecontract.ReasonClockUncertaintyExceeded, "clock uncertainty unavailable"
	case *t.UncertaintyNS > e.maxUnc:
		code, reason = tracecontract.ReasonClockUncertaintyExceeded, "clock uncertainty exceeds bound"
	default:
		v := float64(*t.CompleteNanos - *start)
		value = &v
		intervals = []tracecontract.Interval{{Start: *start, End: *t.CompleteNanos, Clock: t.ClockDomain}}
	}
	add(RuleMovementDuration, class, msg, value, "ns", intervals, code, reason)

	code, reason = e.path(t, direction)
	add(RuleMovementPath, tracecontract.FindingAssociated, "declared "+t.Path.Mechanism+" path consistent with retained evidence; not an observed route", nil, "", nil, code, reason)

	add(RuleMovementActivity, class, "transfer operation exactly linked to a captured memcpy activity", nil, "", nil, linkCode, linkReason)

	code, reason = "", ""
	if t.RequestID == "" || t.Attempt == "" {
		code, reason = tracecontract.ReasonEvidenceUnavailable, "request attempt not supplied"
	}
	add(RuleMovementAttempt, tracecontract.FindingAssociated, "supplied request attempt annotation; not a native identity", nil, "", nil, code, reason)

	code, reason = e.symptoms(t, &s)
	if modeled {
		s.Symptoms = []string{}
	}
	if code == "" {
		code, reason = tracecontract.ReasonControlMissing, "no congestion rule; latency, bytes and symptoms do not establish congestion"
	}
	add(RuleMovementCongestion, class, "transport congestion", nil, "", nil, code, reason)
	return s
}

func movementFinding(t nvtelemetry.MovementTransfer, rule string, class tracecontract.FindingClass, msg string, value *float64, unit string, intervals []tracecontract.Interval, code, reason string) tracecontract.Finding {
	f := tracecontract.Finding{ID: "movement-" + t.ID + "-" + strings.TrimPrefix(rule, "movement_"), Rule: rule, Class: class, Artifacts: []string{t.RawSHA256}, Intervals: intervals, Requirements: []tracecontract.Requirement{}, Message: msg, Value: value, Unit: unit}
	if t.Path.Evidence != "" && rule == RuleMovementPath {
		f.Artifacts = append(f.Artifacts, t.Path.Evidence)
	}
	if code != "" {
		f.Class, f.Value, f.Intervals = tracecontract.FindingUnavailable, nil, nil
		f.Admission = tracecontract.Admission{Status: "withheld", ReasonCode: code, Reason: reason}
		return f
	}
	f.Admission = tracecontract.Admission{Admitted: true, Status: "admitted", ReasonCode: tracecontract.ReasonAdmitted, Reason: "supplied movement evidence consistent"}
	return f
}

// movementDirection reports whether the declared direction agrees with the
// endpoint kinds. Device-to-device is the same device; peer-to-peer is not.
func movementDirection(t nvtelemetry.MovementTransfer) bool {
	src, dst := t.Source.Kind, t.Destination.Kind
	host := func(k string) bool { return k == "host_pinned" || k == "host_pageable" }
	switch t.Direction {
	case "host_to_device":
		return host(src) && dst == "device"
	case "device_to_host":
		return src == "device" && host(dst)
	case "device_to_device":
		return src == "device" && dst == "device" && t.Source.Entity == t.Destination.Entity
	case "peer_to_peer":
		return src == "device" && dst == "device" && t.Source.Entity != t.Destination.Entity
	case "storage_to_device":
		return src == "storage" && dst == "device"
	case "device_to_storage":
		return src == "device" && dst == "storage"
	case "network_send":
		return (src == "device" || host(src)) && dst == "network"
	case "network_receive":
		return src == "network" && (dst == "device" || host(dst))
	}
	return false
}

// activityLink finds the single exact memcpy join of the same run for the
// transfer's supplied operation. Timestamp proximity and native IDs alone
// never link a transfer. It returns the admitted join index, or -1, and the
// candidate memcpy index even when the link is contradicted.
func (e movementEvaluator) activityLink(t nvtelemetry.MovementTransfer) (join, candidate int, code, reason string) {
	if t.OpID == 0 {
		return -1, -1, tracecontract.ReasonCorrelationUnmatched, "operation identity not supplied"
	}
	found := -1
	for i, j := range e.activities.Joins {
		sub := j.Join.HostSubmission
		if j.Join.OpID != t.OpID && (sub == nil || sub.OpID != t.OpID) {
			continue
		}
		if j.Join.Outcome != cudaworkload.OutcomeExact {
			return -1, -1, tracecontract.ReasonCorrelationUnmatched, "operation activity join is not exact"
		}
		if found >= 0 {
			return -1, -1, tracecontract.ReasonCorrelationUnmatched, "operation links several activities"
		}
		found = i
	}
	if found < 0 {
		return -1, -1, tracecontract.ReasonCorrelationUnmatched, "no exact activity for operation"
	}
	j := e.activities.Joins[found]
	r := j.Record
	if j.Run != e.c.RunID || j.Join.Workload.RunID != "" && j.Join.Workload.RunID != e.c.RunID {
		return -1, -1, tracecontract.ReasonIdentityMismatch, "activity belongs to another run"
	}
	if r.Kind != cupti.ActivityKindMemcpy {
		return -1, -1, tracecontract.ReasonIdentityMismatch, "linked activity is not a memcpy"
	}
	if r.CopyCount > 1 {
		return -1, found, tracecontract.ReasonIdentityMismatch, "memcpy record represents several copies"
	}
	want, ok := map[string]cupti.ActivityMemcpyKind{"host_to_device": cupti.ActivityMemcpyKindHtod, "device_to_host": cupti.ActivityMemcpyKindDtoh, "device_to_device": cupti.ActivityMemcpyKindDtod, "peer_to_peer": cupti.ActivityMemcpyKindPtop}[t.Direction]
	if !ok || r.CopyKind != want || !memoryKind(t.Source.Kind, r.SourceMemoryKind) || !memoryKind(t.Destination.Kind, r.DestinationMemoryKind) {
		return -1, found, tracecontract.ReasonIdentityMismatch, "memcpy direction or memory kinds contradict transfer"
	}
	if t.Bytes != nil && r.Bytes != *t.Bytes {
		return -1, found, tracecontract.ReasonIdentityMismatch, "memcpy bytes contradict transfer"
	}
	return found, found, "", ""
}

func memoryKind(kind string, native cupti.ActivityMemoryKind) bool {
	switch native {
	case cupti.ActivityMemoryKindUnknown:
		return true
	case cupti.ActivityMemoryKindDevice, cupti.ActivityMemoryKindDeviceStatic:
		return kind == "device"
	case cupti.ActivityMemoryKindPinned:
		return kind == "host_pinned"
	case cupti.ActivityMemoryKindPageable:
		return kind == "host_pageable"
	case cupti.ActivityMemoryKindManaged, cupti.ActivityMemoryKindManagedStatic:
		return kind == "managed"
	}
	return false
}

// path checks a declared mechanism against retained evidence and, for
// NVLink, the topology snapshot. See SummarizeMovement.
func (e movementEvaluator) path(t nvtelemetry.MovementTransfer, direction bool) (string, string) {
	p := t.Path
	switch {
	case p.Mechanism == "unknown":
		return tracecontract.ReasonEvidenceUnavailable, "path unknown"
	case !direction:
		return tracecontract.ReasonIdentityMismatch, "direction contradicts endpoints"
	case p.Evidence == e.in.Receipt.InputSHA256:
		return tracecontract.ReasonEvidenceUnavailable, "CUDA activity export cannot establish a transfer path"
	}
	if _, ok := e.in.Artifacts[p.Evidence]; !ok {
		return tracecontract.ReasonEvidenceUnavailable, "path evidence artifact not retained"
	}
	switch p.Mechanism {
	case "gds_direct", "gds_fallback", "host_staged", "network_rdma":
		if p.Basis != "independent_artifact" {
			return tracecontract.ReasonEvidenceUnavailable, "direct, fallback, staged or RDMA mode requires an independent mechanism artifact"
		}
		storage := t.Source.Kind == "storage" || t.Destination.Kind == "storage"
		network := t.Source.Kind == "network" || t.Destination.Kind == "network"
		if strings.HasPrefix(p.Mechanism, "gds_") && !storage || p.Mechanism == "network_rdma" && !network {
			return tracecontract.ReasonIdentityMismatch, "path mechanism contradicts endpoint kinds"
		}
		return "", ""
	}
	if e.health == nil {
		return tracecontract.ReasonEvidenceUnavailable, "no topology snapshot"
	}
	if t.TopologyEpoch == "" || t.TopologyEpoch != e.health.TopologyEpoch {
		return tracecontract.ReasonIdentityMismatch, "transfer and topology epochs differ"
	}
	if t.ClockDomain != e.health.ClockDomain || t.ClockEpoch != e.health.ClockEpoch {
		return tracecontract.ReasonClockDomainMismatch, "topology validity cannot be mapped to transfer time"
	}
	start, end, ok := transferInterval(t)
	if !ok {
		return tracecontract.ReasonEvidenceUnavailable, "transfer interval unavailable for topology validity"
	}
	if t.UncertaintyNS == nil || *t.UncertaintyNS > e.maxUnc || end > math.MaxInt64-*t.UncertaintyNS {
		return tracecontract.ReasonClockUncertaintyExceeded, "transfer clock uncertainty unavailable or exceeds bound"
	}
	start -= *t.UncertaintyNS
	end += *t.UncertaintyNS
	kinds := map[string]string{"nvlink": "nvlink_remote_pci"}
	kind, ok := kinds[p.Mechanism]
	if !ok {
		return tracecontract.ReasonBackendUnsupported, "no supported topology edge kind for mechanism"
	}
	for _, edge := range e.health.Edges {
		if edge.From == t.Source.Entity && edge.To == t.Destination.Entity && edge.Kind == kind && edge.State == "available" && edge.ValidFromNanos <= start && end < edge.ValidUntilNanos {
			return "", ""
		}
	}
	return tracecontract.ReasonIdentityMismatch, "no available topology edge between endpoints"
}

// symptoms attaches admitted telemetry observations on a transfer endpoint
// whose interval overlaps the transfer in the same clock domain and epoch.
func (e movementEvaluator) symptoms(t nvtelemetry.MovementTransfer, s *MovementTransferSummary) (string, string) {
	if e.telemetry == nil {
		return "", ""
	}
	start, end, ok := transferInterval(t)
	if !ok || t.UncertaintyNS == nil || *t.UncertaintyNS > e.maxUnc {
		return tracecontract.ReasonClockUncertaintyExceeded, "transfer interval or clock bound unavailable; symptoms not mapped"
	}
	mapped := false
	for _, o := range e.telemetry.Observations {
		obs := o.Reading.Observation
		if !o.Finding.Admission.Admitted || obs == nil || obs.Interval == nil {
			continue
		}
		if obs.Interval.Clock != t.ClockDomain || obs.ClockEpoch != t.ClockEpoch {
			continue
		}
		mapped = true
		dev := o.Reading.DeviceUUID
		if (dev == t.Source.Entity || dev == t.Destination.Entity) && obs.Interval.Start <= end && start <= obs.Interval.End {
			s.Symptoms = append(s.Symptoms, o.Finding.ID)
		}
	}
	sort.Strings(s.Symptoms)
	if !mapped {
		return tracecontract.ReasonClockDomainMismatch, "no telemetry observation shares the transfer clock"
	}
	return "", ""
}

// transferInterval returns the earliest start or enqueue time and the
// completion or latest progress time.
func transferInterval(t nvtelemetry.MovementTransfer) (start, end int64, ok bool) {
	s, e := t.StartNanos, t.CompleteNanos
	if s == nil {
		s = t.EnqueueNanos
	}
	if e == nil {
		e = t.ProgressNanos
	}
	if s == nil || e == nil {
		return 0, 0, false
	}
	return *s, *e, true
}

func movementFamilies(c nvtelemetry.MovementContext) []MovementFamily {
	byFamily := make(map[string]*MovementFamily)
	for family := range c.LossByFamily {
		byFamily[family] = &MovementFamily{Family: family}
	}
	for _, t := range c.Transfers {
		f := byFamily[t.Family]
		if f == nil {
			f = &MovementFamily{Family: t.Family}
			byFamily[t.Family] = f
		}
		f.Total++
		switch t.State {
		case "completed":
			f.Completed++
		case "failed":
			f.Failed++
		case "cancelled":
			f.Cancelled++
		default:
			f.Pending++
		}
	}
	out := make([]MovementFamily, 0, len(byFamily))
	for family, f := range byFamily {
		loss, ok := c.LossByFamily[family]
		switch {
		case c.SourceClass == "modeled":
			f.Reason = "modeled movement source"
		case !ok:
			f.Reason = "family loss unassessed"
		case loss.Unknown:
			f.Reason = "family loss unknown"
		case loss.Dropped != 0:
			f.Reason = fmt.Sprintf("family dropped %d of %d records", loss.Dropped, loss.Total)
		default:
			f.Qualified = true
			f.Reason = "known zero family loss"
		}
		out = append(out, *f)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Family < out[j].Family })
	return out
}
