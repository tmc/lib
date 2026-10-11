package hostmetrics

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
	"time"
)

// RedfishSensor is the bounded Sensor.v1_9_0 power/temperature projection.
// Vendor health is a reported judgment, not an independently measured result.
// ReadingTime is retained verbatim; no host-clock calibration is inferred.
// Empty State means absent or null vendor state. A numeric Reading can be
// retained while the sensor is inactive; it is not then an available reading.
type RedfishSensor struct {
	Context                                                                    Context
	ID, URI, ReadingType, Unit, ImplementationType, State, Health, ReadingTime string
	Reading                                                                    *float64
}

// ReadRedfishSensor reads a supplied DMTF Sensor.v1_9_0 artifact. URI and ID
// must equal caller-selected identities. Other schema fields are ignored after
// duplicate/depth checks. No requests, discovery, or sensor controls occur.
func ReadRedfishSensor(r io.Reader, c Context, uri, id string, maxBytes int64) (RedfishSensor, error) {
	b, e := read(r, c, maxBytes)
	if e != nil {
		return RedfishSensor{}, e
	}
	d := json.NewDecoder(strings.NewReader(b))
	d.UseNumber()
	if e = checkJSON(d, 0); e != nil {
		return RedfishSensor{}, e
	}
	if _, e = d.Token(); e != io.EOF {
		return RedfishSensor{}, fmt.Errorf("trailing Redfish JSON")
	}
	var raw map[string]json.RawMessage
	if e = json.Unmarshal([]byte(b), &raw); e != nil {
		return RedfishSensor{}, e
	}
	s := RedfishSensor{Context: c}
	get := func(k string, dest *string) error {
		v, ok := raw[k]
		if !ok || string(v) == "null" {
			return fmt.Errorf("missing Redfish field %s", k)
		}
		return json.Unmarshal(v, dest)
	}
	var typ string
	for _, x := range []struct {
		k   string
		dst *string
	}{{"@odata.type", &typ}, {"@odata.id", &s.URI}, {"Id", &s.ID}, {"ReadingType", &s.ReadingType}, {"ReadingUnits", &s.Unit}, {"Implementation", &s.ImplementationType}} {
		if e = get(x.k, x.dst); e != nil {
			return RedfishSensor{}, e
		}
	}
	if typ != "#Sensor.v1_9_0.Sensor" || s.URI != uri || s.ID != id {
		return RedfishSensor{}, fmt.Errorf("Redfish identity or schema mismatch")
	}
	if v, ok := raw["ReadingBasis"]; ok {
		var basis string
		if e = json.Unmarshal(v, &basis); e != nil || basis != "Zero" {
			return RedfishSensor{}, fmt.Errorf("unsupported Redfish reading basis")
		}
	}
	var status map[string]json.RawMessage
	if v, ok := raw["Status"]; ok {
		if e = json.Unmarshal(v, &status); e != nil {
			return RedfishSensor{}, fmt.Errorf("invalid Redfish status")
		}
	}
	if v, ok := status["State"]; ok {
		if e = json.Unmarshal(v, &s.State); e != nil {
			return RedfishSensor{}, fmt.Errorf("invalid Redfish state")
		}
	}
	if v, ok := status["Health"]; ok && string(v) != "null" {
		if e = json.Unmarshal(v, &s.Health); e != nil {
			return RedfishSensor{}, e
		}
	}
	if v, ok := raw["ReadingTime"]; ok && string(v) != "null" {
		if e = json.Unmarshal(v, &s.ReadingTime); e != nil {
			return RedfishSensor{}, e
		}
	}
	v, ok := raw["Reading"]
	if !ok {
		return RedfishSensor{}, fmt.Errorf("missing Redfish reading")
	}
	if e = json.Unmarshal(v, &s.Reading); e != nil {
		return RedfishSensor{}, e
	}
	return s, s.Validate()
}

// Validate checks the supported Redfish projection without interpreting health.
func (s RedfishSensor) Validate() error {
	if e := s.Context.validate(); e != nil {
		return e
	}
	if !identity(s.ID) || !identity(s.URI) || !strings.HasPrefix(s.URI, "/redfish/v1/") {
		return fmt.Errorf("invalid Redfish identity")
	}
	if (s.ReadingType != "Power" || s.Unit != "W") && (s.ReadingType != "Temperature" || s.Unit != "Cel") {
		return fmt.Errorf("unsupported Redfish type or unit")
	}
	if s.ImplementationType != "PhysicalSensor" && s.ImplementationType != "Reported" && s.ImplementationType != "Synthesized" {
		return fmt.Errorf("unknown Redfish implementation")
	}
	switch s.State {
	case "", "Enabled", "Disabled", "Absent", "UnavailableOffline", "StandbyOffline", "StandbySpare", "Starting", "Updating":
	default:
		return fmt.Errorf("unsupported Redfish state")
	}
	if s.Health != "" && s.Health != "OK" && s.Health != "Warning" && s.Health != "Critical" {
		return fmt.Errorf("unknown Redfish health")
	}
	if s.ReadingTime != "" {
		if _, e := time.Parse(time.RFC3339Nano, s.ReadingTime); e != nil {
			return fmt.Errorf("invalid Redfish reading time")
		}
	}
	if s.Reading != nil {
		v := *s.Reading
		if math.IsNaN(v) || math.IsInf(v, 0) || (s.ReadingType == "Power" && v < 0) {
			return fmt.Errorf("Redfish reading contradicts units")
		}
	}
	return nil
}

// checkJSON rejects duplicate keys, excessive nesting and JSON name aliases
// before projecting selected fields. Exact DMTF spelling is required.
func checkJSON(d *json.Decoder, depth int) error {
	if depth > 16 {
		return fmt.Errorf("Redfish JSON exceeds depth limit")
	}
	t, e := d.Token()
	if e != nil {
		return e
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			k, e := d.Token()
			if e != nil {
				return e
			}
			key, ok := k.(string)
			if !ok {
				return fmt.Errorf("invalid JSON key")
			}
			lower := strings.ToLower(key)
			if seen[lower] {
				return fmt.Errorf("duplicate Redfish JSON key")
			}
			seen[lower] = true
			if e = checkJSON(d, depth+1); e != nil {
				return e
			}
		}
	case '[':
		for d.More() {
			if e = checkJSON(d, depth+1); e != nil {
				return e
			}
		}
	default:
		return fmt.Errorf("invalid Redfish JSON delimiter")
	}
	_, e = d.Token()
	return e
}
