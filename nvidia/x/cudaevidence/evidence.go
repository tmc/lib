package cudaevidence

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
)

// Class describes how a value was established. Classes have no ranking.
type Class string

const (
	// Unread means no usable value is available. It is the zero class.
	Unread Class = ""
	// Measured is an observation supplied by an identified producer.
	Measured Class = "measured"
	// Derived is computed from stated inputs by an explicit method.
	Derived Class = "derived"
	// Inferred is an interpretation that is not directly observed.
	Inferred Class = "inferred"
	// Modeled is a prediction under stated assumptions.
	Modeled Class = "modeled"
)

// Identity retains existing artifact and cache keys without rehashing them.
// Entry and target scope a fact within an artifact; compiler and workload keys
// preserve the corresponding lane 40 identities.
type Identity struct {
	ArtifactDigest string `json:"artifact_digest"`
	Entry          string `json:"entry"`
	Target         string `json:"target"`
	CompilerKey    string `json:"compiler_key"`
	WorkloadKey    string `json:"workload_key"`
}

// Validate rejects incomplete identities. It does not verify source bytes.
func (id Identity) Validate() error {
	for _, key := range []string{id.ArtifactDigest, id.CompilerKey, id.WorkloadKey} {
		b, err := hex.DecodeString(key)
		if err != nil || len(b) != 32 || key != strings.ToLower(key) {
			return fmt.Errorf("evidence identity: invalid digest")
		}
	}
	if strings.TrimSpace(id.Entry) == "" || strings.TrimSpace(id.Target) == "" {
		return fmt.Errorf("evidence identity: missing entry or target")
	}
	return nil
}

// Source identifies a producer and its record within an existing identity.
type Source struct {
	Identity Identity `json:"identity"`
	Producer string   `json:"producer"`
	Locator  string   `json:"locator"`
}

// Loss describes omitted records out of Total attempted records.
type Loss struct {
	Dropped uint64 `json:"dropped"`
	Total   uint64 `json:"total"`
}

// Record is the JSON representation and constructor input for a Value.
// Number is nil for Unread; zero is a real numeric value. Uncertainty is an
// optional nonnegative absolute uncertainty in Unit. Reason explains Unread.
type Record struct {
	Class       Class    `json:"class"`
	Number      *float64 `json:"number"`
	Unit        string   `json:"unit,omitempty"`
	Basis       string   `json:"basis,omitempty"`
	Source      Source   `json:"source"`
	Reason      string   `json:"reason,omitempty"`
	Uncertainty *float64 `json:"uncertainty,omitempty"`
	Loss        *Loss    `json:"loss,omitempty"`
}

// Value is an immutable classified numeric value. Its zero value is Unread.
// Do not concurrently replace a Value variable while other goroutines read it.
type Value struct{ record Record }

// New validates and copies a record.
func New(r Record) (Value, error) {
	if r.Class == "unread" {
		r.Class = Unread
	}
	switch r.Class {
	case Unread:
		if r.Number != nil || r.Uncertainty != nil {
			return Value{}, fmt.Errorf("evidence: unread carries numeric value")
		}
		if strings.TrimSpace(r.Reason) == "" {
			return Value{}, fmt.Errorf("evidence: unread needs reason")
		}
	case Measured, Derived, Inferred, Modeled:
		if r.Number == nil || math.IsNaN(*r.Number) || math.IsInf(*r.Number, 0) {
			return Value{}, fmt.Errorf("evidence: missing or nonfinite number")
		}
		if strings.TrimSpace(r.Basis) == "" || strings.TrimSpace(r.Source.Producer) == "" || strings.TrimSpace(r.Source.Locator) == "" {
			return Value{}, fmt.Errorf("evidence: missing basis or source")
		}
		if err := r.Source.Identity.Validate(); err != nil {
			return Value{}, err
		}
		if r.Reason != "" {
			return Value{}, fmt.Errorf("evidence: available value carries unread reason")
		}
	default:
		return Value{}, fmt.Errorf("evidence: unknown class %q", r.Class)
	}
	if r.Uncertainty != nil && (*r.Uncertainty < 0 || math.IsNaN(*r.Uncertainty) || math.IsInf(*r.Uncertainty, 0)) {
		return Value{}, fmt.Errorf("evidence: invalid uncertainty")
	}
	if r.Loss != nil && r.Loss.Dropped > r.Loss.Total {
		return Value{}, fmt.Errorf("evidence: losses exceed total")
	}
	return Value{copyRecord(r)}, nil
}

func copyRecord(r Record) Record {
	if r.Number != nil {
		n := *r.Number
		r.Number = &n
	}
	if r.Uncertainty != nil {
		n := *r.Uncertainty
		r.Uncertainty = &n
	}
	if r.Loss != nil {
		loss := *r.Loss
		r.Loss = &loss
	}
	return r
}

// Record returns a copy of the classification and provenance.
func (v Value) Record() Record {
	r := copyRecord(v.record)
	if r.Class == Unread && r.Reason == "" {
		r.Reason = "not read"
	}
	return r
}

// MarshalJSON retains classification and explicit null for an unread number.
func (v Value) MarshalJSON() ([]byte, error) {
	r := v.Record()
	if r.Class == Unread {
		r.Class = "unread"
	}
	return json.Marshal(r)
}

// Decode reads one JSON record, rejecting unknown fields and trailing values.
// It returns a new Value; existing Values are never modified by decoding.
func Decode(data []byte) (Value, error) {
	if err := uniqueFields(json.NewDecoder(bytes.NewReader(data))); err != nil {
		return Value{}, fmt.Errorf("decode evidence: %w", err)
	}
	var r Record
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&r); err != nil {
		return Value{}, fmt.Errorf("decode evidence: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return Value{}, fmt.Errorf("decode evidence: trailing data")
	}
	return New(r)
}

// UnmarshalJSON replaces the destination with a validated independent Value.
// Failed decoding leaves it unchanged. As with assigning a Value variable,
// callers must not decode into a destination being read concurrently.
func (v *Value) UnmarshalJSON(data []byte) error {
	if v == nil {
		return fmt.Errorf("decode evidence: nil destination")
	}
	next, err := Decode(data)
	if err != nil {
		return err
	}
	*v = next
	return nil
}

// uniqueFields rejects ambiguous JSON object members at every nesting level.
func uniqueFields(d *json.Decoder) error {
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
			token, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := token.(string)
			if !ok || seen[key] || key != strings.ToLower(key) || !asciiField(key) {
				return fmt.Errorf("duplicate or invalid object field %q", token)
			}
			seen[key] = true
			if err := uniqueFields(d); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := uniqueFields(d); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unexpected delimiter")
	}
	_, err = d.Token()
	return err
}

func asciiField(key string) bool {
	for _, r := range key {
		if r > 127 {
			return false
		}
	}
	return true
}

// CheckClaim verifies exact classification and identity. It never promotes a
// conclusion to an observation. Unread cannot satisfy a numeric claim.
func (v Value) CheckClaim(class Class, id Identity) error {
	if err := id.Validate(); err != nil {
		return err
	}
	if v.record.Class == Unread {
		return fmt.Errorf("evidence claim: unread")
	}
	if v.record.Source.Identity != id {
		return fmt.Errorf("evidence claim: identity mismatch")
	}
	if v.record.Class != class {
		return fmt.Errorf("evidence claim: class mismatch")
	}
	return nil
}
