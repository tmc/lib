package nvtelemetry

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
)

// exportEntityResource transports a bounded caller-asserted relationship, never
// inferring device consumption from allocation. Only explicit process-scoped
// readings with a current complete assigned identity receive owner attributes.
// Unknown/incomplete/unverified relationships remain visibly withheld. These
// private attributes do not authenticate the supplying authority.
func exportEntityResource(r Reading, options ExportOptions) ([]any, string, error) {
	attrs := []any{}
	text := func(key, value string) error {
		if !exportText(value) {
			return fmt.Errorf("export: invalid entity attribute")
		}
		attrs = append(attrs, map[string]any{"key": "nvtelemetry." + key, "value": map[string]string{"stringValue": value}})
		return nil
	}
	boolean := func(key string, value bool) {
		attrs = append(attrs, map[string]any{"key": "nvtelemetry." + key, "value": map[string]bool{"boolValue": value}})
	}
	integer := func(key string, value int64) {
		attrs = append(attrs, map[string]any{"key": "nvtelemetry." + key, "value": map[string]string{"intValue": strconv.FormatInt(value, 10)}})
	}
	if err := text("device.uuid", r.DeviceUUID); err != nil {
		return nil, "", err
	}
	finish := func() ([]any, string, error) {
		if len(attrs) > 64 {
			return nil, "", fmt.Errorf("export: entity attribute limit")
		}
		key, err := json.Marshal(attrs)
		return attrs, string(key), err
	}
	o := r.Observation
	if o != nil {
		if o.Scope == "process" && o.Assignment == nil {
			return nil, "", fmt.Errorf("export: process scope requires assignment evidence")
		}
		if o.SourceClass != "" && o.SourceClass != "native" && o.SourceClass != "managed" && o.SourceClass != "imported" && o.SourceClass != "synthetic" && o.SourceClass != "modeled" {
			return nil, "", fmt.Errorf("export: invalid observation source class")
		}
		for _, pair := range [][2]string{
			{"observation.scope", o.Scope}, {"observation.source_class", o.SourceClass},
			{"observation.producer", o.Producer}, {"observation.producer_version", o.ProducerVersion}, {"observation.producer_schema", o.ProducerSchema},
			{"observation.population", o.Observer.Population}, {"observation.mode", o.Observer.Mode}, {"observation.pass", o.Observer.Pass}, {"observation.filter", o.Observer.Filter}, {"observation.replay", o.Observer.Replay},
		} {
			if err := text(pair[0], pair[1]); err != nil {
				return nil, "", err
			}
		}
	}
	if o == nil || o.Assignment == nil {
		if err := text("assignment.status", "absent"); err != nil {
			return nil, "", err
		}
		return finish()
	}
	a := o.Assignment
	if !exportText(a.State) || !exportText(a.Sharing) || !exportText(a.MPSRole) {
		return nil, "", fmt.Errorf("export: invalid entity assignment")
	}
	if err := a.Validate(); err != nil {
		// Structural enum errors can quote caller input. Keep that text out of
		// collector diagnostics as well as the encoded resource.
		return nil, "", fmt.Errorf("export: invalid entity assignment")
	}
	if a.Device.UUID != "" && a.Device.UUID != r.DeviceUUID {
		return nil, "", fmt.Errorf("export: assignment device mismatch")
	}
	if len(a.Process.NamespacePIDs) > 32 || a.MPSServer != nil && len(a.MPSServer.NamespacePIDs) > 32 {
		return nil, "", fmt.Errorf("export: namespace pid limit")
	}
	if o.Interval == nil || o.Interval.Start < 0 || o.Interval.End < o.Interval.Start || o.Interval.Clock != a.ClockDomain || o.ClockEpoch != a.ClockEpoch || options.ClockDomain != "" && options.ClockDomain != a.ClockDomain {
		return nil, "", fmt.Errorf("export: assignment clock or epoch mismatch")
	}
	uncertainty := int64(0)
	if o.UncertaintyNS != nil {
		uncertainty = *o.UncertaintyNS
	}
	if uncertainty < 0 || o.Interval.Start < uncertainty || o.Interval.End > math.MaxInt64-uncertainty || o.Interval.Start-uncertainty < a.StartNanos || o.Interval.End+uncertainty >= a.EndNanos {
		return nil, "", fmt.Errorf("export: assignment is not current for observation")
	}
	if o.Workload != nil && !o.Workload.Process.IsZero() && o.Workload.Process != a.Process.Process || o.Host != nil && !o.Host.Process.IsZero() && o.Host.Process != a.Process.Process || o.Device != nil && !o.Device.Process.IsZero() && o.Device.Process != a.Process.Process {
		return nil, "", fmt.Errorf("export: assignment process generation mismatch")
	}
	status := a.State
	if status == "assigned" {
		switch {
		case o.Scope != "process":
			status = "device_scope"
		case !a.Authoritative:
			status = "unverified"
		case o.UncertaintyNS == nil || !completeEntityProcess(a.Process):
			status = "unknown"
		}
	}
	for _, pair := range [][2]string{
		{"assignment.status", status}, {"assignment.state", a.State}, {"assignment.source", a.Source}, {"assignment.source_revision", a.SourceRevision},
		{"assignment.clock_domain", a.ClockDomain}, {"assignment.clock_epoch", a.ClockEpoch},
	} {
		if err := text(pair[0], pair[1]); err != nil {
			return nil, "", err
		}
	}
	boolean("assignment.authoritative_assertion", a.Authoritative)
	boolean("observation.uncertainty_known", o.UncertaintyNS != nil)
	if o.UncertaintyNS != nil {
		integer("observation.uncertainty_nanos", *o.UncertaintyNS)
	}
	integer("assignment.start_nanos", a.StartNanos)
	integer("assignment.end_nanos", a.EndNanos)
	if o.Scope == "process" {
		process := func(prefix string, p EntityProcess) error {
			for _, pair := range [][2]string{{prefix + "node", p.NodeID}, {prefix + "boot", p.BootID}, {prefix + "pid_namespace", p.PIDNamespace}, {prefix + "generation", strconv.FormatUint(p.Process.Generation, 10)}, {prefix + "exec_generation", strconv.FormatUint(p.ExecGeneration, 10)}} {
				if err := text(pair[0], pair[1]); err != nil {
					return err
				}
			}
			integer(prefix+"pid", int64(p.Process.PID))
			boolean(prefix+"exec_generation_known", p.HasExecGeneration)
			values := make([]any, 0, len(p.NamespacePIDs))
			for _, pid := range p.NamespacePIDs {
				values = append(values, map[string]string{"intValue": strconv.FormatUint(uint64(pid), 10)})
			}
			attrs = append(attrs, map[string]any{"key": "nvtelemetry." + prefix + "namespace_pids", "value": map[string]any{"arrayValue": map[string]any{"values": values}}})
			return nil
		}
		if err := process("process.", a.Process); err != nil {
			return nil, "", err
		}
		if a.MPSServer != nil {
			if !completeEntityProcess(*a.MPSServer) || a.MPSServer.BootID != a.Process.BootID || a.MPSServer.NodeID != a.Process.NodeID {
				return nil, "", fmt.Errorf("export: incomplete or conflicting mps server")
			}
			if err := process("mps.server.", *a.MPSServer); err != nil {
				return nil, "", err
			}
		}
		for _, pair := range [][2]string{{"assignment.sharing", a.Sharing}, {"mps.role", a.MPSRole}, {"mig.configuration_epoch", a.MIGConfigurationEpoch}, {"mig.scope", a.Device.MIGScope}, {"mig.parent_uuid", a.Device.MIGParentUUID}} {
			if err := text(pair[0], pair[1]); err != nil {
				return nil, "", err
			}
		}
		if !a.Device.HasMIGInstanceID && a.Device.MIGInstanceID != 0 || !a.Device.HasMIGComputeInstanceID && a.Device.MIGComputeInstanceID != 0 || a.Device.HasMIGInstanceID && a.Device.MIGInstanceID < 0 || a.Device.HasMIGComputeInstanceID && a.Device.MIGComputeInstanceID < 0 {
			return nil, "", fmt.Errorf("export: contradictory mig identity")
		}
		boolean("mig.instance_known", a.Device.HasMIGInstanceID)
		boolean("mig.compute_instance_known", a.Device.HasMIGComputeInstanceID)
		if a.Device.HasMIGInstanceID {
			integer("mig.instance", int64(a.Device.MIGInstanceID))
		}
		if a.Device.HasMIGComputeInstanceID {
			integer("mig.compute_instance", int64(a.Device.MIGComputeInstanceID))
		}
	}
	if status == "assigned" {
		for _, pair := range [][2]string{{"owner.pod_uid", a.Owner.PodUID}, {"owner.container_id", a.Owner.ContainerID}, {"owner.job_id", a.Owner.JobID}, {"owner.allocation_id", a.Owner.AllocationID}, {"owner.attempt", a.Owner.Attempt}} {
			if err := text(pair[0], pair[1]); err != nil {
				return nil, "", err
			}
		}
	}
	provenance := make([]any, 0, len(a.Provenance))
	for _, p := range a.Provenance {
		if !exportText(p.Source) || !exportText(p.Revision) {
			return nil, "", fmt.Errorf("export: invalid entity provenance")
		}
		provenance = append(provenance, map[string]any{"kvlistValue": map[string]any{"values": []any{
			map[string]any{"key": "source", "value": map[string]string{"stringValue": p.Source}}, map[string]any{"key": "revision", "value": map[string]string{"stringValue": p.Revision}},
			map[string]any{"key": "start_nanos", "value": map[string]string{"intValue": strconv.FormatInt(p.StartNanos, 10)}}, map[string]any{"key": "end_nanos", "value": map[string]string{"intValue": strconv.FormatInt(p.EndNanos, 10)}},
		}}})
	}
	attrs = append(attrs, map[string]any{"key": "nvtelemetry.assignment.provenance", "value": map[string]any{"arrayValue": map[string]any{"values": provenance}}})
	return finish()
}

// exportSeriesKey keeps relationship revisions from manufacturing independent
// whole-device measurements. Observer fields identify asserted populations;
// changing an allocation or its provenance does not create a new population.
func exportSeriesKey(r Reading, resourceKey string) string {
	if r.Observation != nil && r.Observation.Scope == "process" {
		return resourceKey + "\x00" + r.Name
	}
	parts := []string{r.DeviceUUID, r.Name}
	if o := r.Observation; o != nil {
		parts = append(parts, o.Observer.Population, o.Observer.Mode, o.Observer.Pass, o.Observer.Filter, o.Observer.Replay)
	} else {
		parts = append(parts, "", "", "", "", "")
	}
	key, _ := json.Marshal(parts)
	return string(key)
}

// exportScalar preserves canonical V3 and legacy values without coercing a
// uint64 through float64. OTLP metrics apply their signed-int64 wire limit later;
// the private diagnostic log body can retain the full unsigned decimal value.
func exportScalar(r Reading) (kind, value string, err error) {
	if r.Number != nil {
		if r.IsInt || r.IntValue != 0 || r.Value != 0 {
			return "", "", fmt.Errorf("export: conflicting scalar representations")
		}
		if err := validNumber(r.Number); err != nil {
			return "", "", fmt.Errorf("export: number: %w", err)
		}
		switch {
		case r.Number.Uint != nil:
			return "uint", strconv.FormatUint(*r.Number.Uint, 10), nil
		case r.Number.Int != nil:
			return "int", strconv.FormatInt(*r.Number.Int, 10), nil
		default:
			return "float", strconv.FormatFloat(*r.Number.Float, 'g', -1, 64), nil
		}
	}
	if r.Observation != nil {
		return "", "", fmt.Errorf("export: missing canonical number")
	}
	if r.IsInt {
		return "uint", strconv.FormatUint(r.IntValue, 10), nil
	}
	if math.IsNaN(r.Value) || math.IsInf(r.Value, 0) {
		return "", "", fmt.Errorf("export: non-finite value")
	}
	return "float", strconv.FormatFloat(r.Value, 'g', -1, 64), nil
}

type otlpEntityGroup struct {
	attrs []any
	items []any
}
type otlpEntityExport struct {
	signal      string
	limit, size int
	groups      []otlpEntityGroup
	index       map[string]int
}

func newOTLPEntityExport(signal string, limit int) *otlpEntityExport {
	b, _ := json.Marshal(map[string]any{"resource" + signal: []any{}})
	return &otlpEntityExport{signal: signal, limit: limit, size: len(b), index: make(map[string]int)}
}
func (e *otlpEntityExport) object(g otlpEntityGroup) map[string]any {
	scope, items := "scopeMetrics", "metrics"
	if e.signal == "Logs" {
		scope, items = "scopeLogs", "logRecords"
	}
	return map[string]any{"resource": map[string]any{"attributes": g.attrs}, scope: []any{map[string]any{"scope": map[string]string{"name": otlpScopeName}, items: g.items}}}
}
func (e *otlpEntityExport) add(key string, attrs []any, item any) error {
	encoded, err := json.Marshal(item)
	if err != nil {
		return err
	}
	i, found := e.index[key]
	extra := len(encoded)
	if found {
		if len(e.groups[i].items) > 0 {
			extra++
		}
	} else {
		base, err := json.Marshal(e.object(otlpEntityGroup{attrs: attrs, items: []any{}}))
		if err != nil {
			return err
		}
		extra += len(base)
		if len(e.groups) > 0 {
			extra++
		}
	}
	if extra > e.limit-e.size {
		return fmt.Errorf("export: byte limit")
	}
	if !found {
		i = len(e.groups)
		e.index[key] = i
		e.groups = append(e.groups, otlpEntityGroup{attrs: attrs, items: []any{}})
	}
	e.groups[i].items = append(e.groups[i].items, item)
	e.size += extra
	return nil
}
func (e *otlpEntityExport) write(w io.Writer) error {
	resources := make([]any, 0, len(e.groups))
	for _, g := range e.groups {
		resources = append(resources, e.object(g))
	}
	data, err := json.Marshal(map[string]any{"resource" + e.signal: resources})
	if err != nil {
		return err
	}
	if len(data) != e.size {
		return fmt.Errorf("export: internal resource size mismatch")
	}
	return exportWrite(w, data, e.limit)
}
