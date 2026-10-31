package nvtelemetry

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// suppliedLogEvent names supplied occurrences, never changes inferred from
// counters, device disappearance, XIDs, or assignment snapshots.
func suppliedLogEvent(name string) bool {
	switch name {
	case "ecc_error_event", "job_start_event", "job_end_event", "device_reset_event":
		return true
	}
	return false
}

func validateSuppliedLogEvent(r Reading) error {
	o := r.Observation
	if !suppliedLogEvent(r.Name) {
		if o != nil && o.Kind == KindEvent {
			return fmt.Errorf("event logs: unsupported event category")
		}
		return nil
	}
	if o == nil || o.Kind != KindEvent || r.Unit != "code" {
		return fmt.Errorf("event logs: category requires event observation and code unit")
	}
	if r.Value != 0 || r.IntValue != 0 || r.IsInt || o.Histogram != nil || o.Denominator != nil || o.DenominatorUnit != "" || o.CounterEpoch != "" || o.CounterReset || o.HasReturn || o.OTLPScopeName != nil || o.OTLPClock != nil || o.OTLPSourceTime != nil {
		return fmt.Errorf("event logs: conflicting value or provenance")
	}
	if r.Evidence == EvidenceUnavailable {
		if r.Number != nil || r.UnavailableReason == "" {
			return fmt.Errorf("event logs: invalid unavailable occurrence")
		}
	} else if r.Evidence != EvidenceMeasured || r.UnavailableReason != "" || o.SourceClass == "modeled" || r.Number == nil || r.Number.Uint == nil || r.Number.Int != nil || r.Number.Float != nil || *r.Number.Uint != 1 {
		return fmt.Errorf("event logs: occurrence requires measured unsigned marker one")
	}
	if o.UncertaintyNS != nil && *o.UncertaintyNS < 0 {
		return fmt.Errorf("event logs: negative source uncertainty")
	}
	if p := o.Interval; p != nil {
		if p.Start < 0 || p.Start != p.End || !exportText(p.Clock) {
			return fmt.Errorf("event logs: invalid source instant")
		}
	}
	if !exportText(o.ClockEpoch) {
		return fmt.Errorf("event logs: invalid source epoch")
	}
	if r.HasSourceTimestamp && (!eventSourceTimeKnown(r) || r.SourceTimestamp != o.Interval.Start || r.SourceTimestampDomain != o.Interval.Clock) {
		return fmt.Errorf("event logs: conflicting source timestamp")
	}
	return validateEntityExportStructure(r, true)
}

func eventSourceTimeKnown(r Reading) bool {
	o := r.Observation
	return o != nil && o.Interval != nil && o.Interval.Clock != "" && o.ClockEpoch != ""
}

// eventLogTime keeps collector receipt time separate from the supplied instant.
// A mapping is a caller assertion, not an authenticated calibration.
func eventLogTime(r Reading, options ExportOptions, record map[string]any, attrs []any) ([]any, error) {
	o := r.Observation
	attr := func(key string, value any) {
		attrs = append(attrs, map[string]any{"key": "nvtelemetry.event." + key, "value": value})
	}
	text := func(s string) any { return map[string]string{"stringValue": s} }
	integer := func(n int64) any { return map[string]string{"intValue": strconv.FormatInt(n, 10)} }
	known := eventSourceTimeKnown(r)
	attr("source_time_known", map[string]bool{"boolValue": known})
	attr("source_uncertainty_known", map[string]bool{"boolValue": o.UncertaintyNS != nil})
	if o.UncertaintyNS != nil {
		attr("source_uncertainty_nanos", integer(*o.UncertaintyNS))
	}
	if known {
		attr("source_clock_domain", text(o.Interval.Clock))
		attr("source_clock_epoch", text(o.ClockEpoch))
		attr("source_timestamp_nanos", integer(o.Interval.Start))
	}
	if r.TimestampUnix > 0 {
		record["observedTimeUnixNano"] = strconv.FormatInt(r.TimestampUnix, 10)
	}
	if m := options.UnixTimeMapping; m != nil {
		if !known || m.SourceClockDomain == "" || m.SourceClockEpoch == "" || m.Provenance == "" || !exportText(m.SourceClockDomain) || !exportText(m.SourceClockEpoch) || !exportText(m.Provenance) || m.SourceValidFromNanos < 0 || m.SourceValidUntilNanos < m.SourceValidFromNanos || o.Interval.Clock != m.SourceClockDomain || o.ClockEpoch != m.SourceClockEpoch || o.Interval.Start < m.SourceValidFromNanos || o.Interval.Start > m.SourceValidUntilNanos {
			return nil, fmt.Errorf("event logs: invalid source time mapping")
		}
		timestamp, err := checkedUnixTime(o.Interval.Start, m.UnixOffsetNanos)
		if err != nil {
			return nil, err
		}
		record["timeUnixNano"] = strconv.FormatInt(timestamp, 10)
	}
	return attrs, nil
}

// eventEntityResource never projects a historical relationship as current
// ownership. Missing evidence remains withheld; supplied contradictions refuse.
func eventEntityResource(r Reading, options ExportOptions) ([]any, string, error) {
	if err := validateEntityExportStructure(r, true); err != nil {
		return nil, "", err
	}
	o := r.Observation
	a := o.Assignment
	if a != nil && (a.State == "contradictory" || options.ClockDomain != "" && options.ClockDomain != a.ClockDomain) {
		return nil, "", fmt.Errorf("event logs: contradictory assignment")
	}
	known := eventSourceTimeKnown(r)
	if a != nil && known && (o.Interval.Clock != a.ClockDomain || o.ClockEpoch != a.ClockEpoch || options.ClockDomain != "" && options.ClockDomain != a.ClockDomain) {
		return nil, "", fmt.Errorf("event logs: assignment clock or epoch mismatch")
	}
	boundary := a != nil && known && o.Interval.Start == a.EndNanos
	if a != nil && known && !boundary && r.IsAvailable() && r.Name != "job_end_event" {
		return exportEntityResource(r, options)
	}
	bound := o.Workload != nil && !o.Workload.Process.IsZero() || o.Host != nil && !o.Host.Process.IsZero() || o.Device != nil && !o.Device.Process.IsZero()
	ended := r.IsAvailable() && r.Name == "job_end_event" && boundary && o.Scope == "process" && a.State == "assigned" && a.Authoritative && a.Device.UUID == r.DeviceUUID && completeEntityProcess(a.Process) && a.Owner != (EntityOwner{}) && len(a.Provenance) > 0 && bound && o.Producer != "" && o.ProducerVersion != "" && o.ProducerSchema != "" && o.SourceClass != "" && o.UncertaintyNS != nil && *o.UncertaintyNS == 0 && (!options.HasClockCalibration || options.ClockCalibrationBoundsNanos == 0)
	if boundary && r.Name != "job_end_event" && r.IsAvailable() {
		return nil, "", fmt.Errorf("event logs: assignment is not current for observation")
	}
	attrs := []any{}
	text := func(k, v string) {
		attrs = append(attrs, map[string]any{"key": "nvtelemetry." + k, "value": map[string]string{"stringValue": v}})
	}
	integer := func(k string, v int64) {
		attrs = append(attrs, map[string]any{"key": "nvtelemetry." + k, "value": map[string]string{"intValue": strconv.FormatInt(v, 10)}})
	}
	text("device.uuid", r.DeviceUUID)
	observationAttrs, err := exportObservationAttributes(o)
	if err != nil {
		return nil, "", err
	}
	attrs = append(attrs, observationAttrs...)
	status := "absent"
	if a != nil {
		status = "unknown"
		if !a.Authoritative {
			status = "unverified"
		}
		if o.Scope != "process" {
			status = "device_scope"
		}
	}
	if ended {
		status = "ended_boundary"
		for _, p := range [][2]string{{"assignment.state", a.State}, {"assignment.source", a.Source}, {"assignment.source_revision", a.SourceRevision}, {"assignment.clock_domain", a.ClockDomain}, {"assignment.clock_epoch", a.ClockEpoch}, {"subject.owner.job_id", a.Owner.JobID}, {"subject.owner.allocation_id", a.Owner.AllocationID}, {"subject.owner.attempt", a.Owner.Attempt}, {"subject.owner.pod_uid", a.Owner.PodUID}, {"subject.owner.container_id", a.Owner.ContainerID}} {
			text(p[0], p[1])
		}
		boolean := func(k string, v bool) {
			attrs = append(attrs, map[string]any{"key": "nvtelemetry." + k, "value": map[string]bool{"boolValue": v}})
		}
		process := func(prefix string, p EntityProcess) {
			for _, v := range [][2]string{{"node", p.NodeID}, {"boot", p.BootID}, {"pid_namespace", p.PIDNamespace}, {"generation", strconv.FormatUint(p.Process.Generation, 10)}, {"exec_generation", strconv.FormatUint(p.ExecGeneration, 10)}} {
				text(prefix+v[0], v[1])
			}
			integer(prefix+"pid", int64(p.Process.PID))
			boolean(prefix+"exec_generation_known", p.HasExecGeneration)
			values := make([]any, 0, len(p.NamespacePIDs))
			for _, pid := range p.NamespacePIDs {
				values = append(values, map[string]string{"intValue": strconv.FormatUint(uint64(pid), 10)})
			}
			attrs = append(attrs, map[string]any{"key": "nvtelemetry." + prefix + "namespace_pids", "value": map[string]any{"arrayValue": map[string]any{"values": values}}})
		}
		process("process.", a.Process)
		if a.MPSServer != nil {
			process("mps.server.", *a.MPSServer)
		}
		for _, v := range [][2]string{{"assignment.sharing", a.Sharing}, {"mps.role", a.MPSRole}, {"mig.configuration_epoch", a.MIGConfigurationEpoch}, {"mig.scope", a.Device.MIGScope}, {"mig.parent_uuid", a.Device.MIGParentUUID}} {
			text(v[0], v[1])
		}
		boolean("assignment.authoritative_assertion", a.Authoritative)
		boolean("observation.uncertainty_known", true)
		integer("observation.uncertainty_nanos", *o.UncertaintyNS)
		boolean("mig.instance_known", a.Device.HasMIGInstanceID)
		boolean("mig.compute_instance_known", a.Device.HasMIGComputeInstanceID)
		if a.Device.HasMIGInstanceID {
			integer("mig.instance", int64(a.Device.MIGInstanceID))
		}
		if a.Device.HasMIGComputeInstanceID {
			integer("mig.compute_instance", int64(a.Device.MIGComputeInstanceID))
		}
		integer("assignment.start_nanos", a.StartNanos)
		integer("assignment.end_nanos", a.EndNanos)
		provenance := make([]any, 0, len(a.Provenance))
		for _, p := range a.Provenance {
			provenance = append(provenance, map[string]any{"kvlistValue": map[string]any{"values": []any{
				map[string]any{"key": "source", "value": map[string]string{"stringValue": p.Source}}, map[string]any{"key": "revision", "value": map[string]string{"stringValue": p.Revision}}, map[string]any{"key": "start_nanos", "value": map[string]string{"intValue": strconv.FormatInt(p.StartNanos, 10)}}, map[string]any{"key": "end_nanos", "value": map[string]string{"intValue": strconv.FormatInt(p.EndNanos, 10)}},
			}}})
		}
		attrs = append(attrs, map[string]any{"key": "nvtelemetry.assignment.provenance", "value": map[string]any{"arrayValue": map[string]any{"values": provenance}}})
	}

	text("assignment.status", status)
	if len(attrs) > 64 {
		return nil, "", fmt.Errorf("event logs: entity attribute limit")
	}
	key, err := json.Marshal(attrs)
	return attrs, string(key), err
}
