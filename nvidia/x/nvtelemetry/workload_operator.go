package nvtelemetry

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// OperatorReference identifies retained metadata bytes and a producer record.
// Locator is opaque: readers never open it as a path or infer member identity.
type OperatorReference struct {
	SHA256  string `json:"sha256"`
	Locator string `json:"locator"`
}

// OperatorBinding retains caller-supplied execution and compiler identities.
// Zero/absent code, module and pass fields remain unknown. These declarations
// do not prove that a compiler or device produced the referenced bytes.
type OperatorBinding struct {
	Process          cudaworkload.ProcessGenID          `json:"process"`
	OpID             cudaworkload.OpID                  `json:"op_id"`
	Context          cudaworkload.ContextStreamID       `json:"context"`
	GraphReplay      cudaworkload.GraphReplayGeneration `json:"graph_replay"`
	GraphNodeID      uint64                             `json:"graph_node_id,omitempty"`
	Code             *cudaevidence.Identity             `json:"code,omitempty"`
	ModuleID         string                             `json:"module_id,omitempty"`
	ModuleGeneration uint64                             `json:"module_generation,omitempty"`
	Pass             string                             `json:"pass,omitempty"`
	Replay           string                             `json:"replay,omitempty"`
	CompilerBuild    string                             `json:"compiler_build,omitempty"`
	CompilerOptions  []string                           `json:"compiler_options,omitempty"`
}

// OperatorMember is a supplied framework/compiler record. ID is native to its
// producer and scoped by Source.SHA256; Name is display text. Shapes preserve
// supplied dimensions (-1 means unknown), without computing tensor extents.
type OperatorMember struct {
	ID     string            `json:"id"`
	Name   string            `json:"name"`
	Schema string            `json:"schema,omitempty"`
	Shapes [][]int64         `json:"shapes,omitempty"`
	Source OperatorReference `json:"source"`
}

// OperatorInventory is an application-owned, separately retained expectation.
// Version1 binds one group to its expected producer-native member IDs. Agreement
// establishes completeness relative to these bytes, not producer authenticity.
type OperatorInventory struct {
	Version int              `json:"version"`
	RunID   string           `json:"run_id"`
	Binding OperatorBinding  `json:"binding"`
	Members []OperatorMember `json:"members"`
}

// OperatorGroup supplies contributing members for one annotated submission.
// Inventory is populated only by ReadOperatorContext from InventorySource.
// Its metadata is never inferred from names, kernel geometry or launch order.
type OperatorGroup struct {
	ID              string             `json:"id"`
	Iteration       string             `json:"iteration,omitempty"`
	Binding         OperatorBinding    `json:"binding"`
	Members         []OperatorMember   `json:"members"`
	InventorySource *OperatorReference `json:"inventory_source,omitempty"`
	Inventory       *OperatorInventory `json:"-"`
	InventoryReason string             `json:"-"`
}

// OperatorContext is version1 application interchange, not a framework export.
// Artifacts closes metadata references with byte-exact raw payloads keyed by
// SHA256. Batch contains metadata only; its Records must be empty. Family names
// select existing Receipt loss entries; empty names retain unknown coverage.
// The zero value is not a valid interchange.
type OperatorContext struct {
	Version           int                           `json:"version"`
	RunID             string                        `json:"run_id"`
	Producer          string                        `json:"producer"`
	ProducerVersion   string                        `json:"producer_version"`
	ProducerSchema    string                        `json:"producer_schema"`
	SourceClass       string                        `json:"source_class"`
	Batch             cudaworkload.ActivityBatch    `json:"batch"`
	Receipt           tracecontract.Receipt         `json:"receipt"`
	Annotations       []cudaworkload.HostSubmission `json:"annotations"`
	Artifacts         map[string][]byte             `json:"artifacts"`
	Groups            []OperatorGroup               `json:"groups"`
	ActivityFamily    string                        `json:"activity_family,omitempty"`
	CorrelationFamily string                        `json:"correlation_family,omitempty"`
	MetadataFamily    string                        `json:"metadata_family,omitempty"`
}

// ReadOperatorContext reads bounded application operator-context-v1 metadata.
// It preserves original bytes/hash, rejects duplicate/unknown fields and checks
// raw artifact hashes. Missing or unsupported inventory remains explicitly
// unknown. It does not authenticate supplied compiler or framework metadata.
func ReadOperatorContext(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, OperatorContext, error) {
	var c OperatorContext
	if s.Provider != "application" || s.Version != "operator-context-v1" {
		return WorkloadExport{}, c, errors.New("operator context requires application operator-context-v1 source")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, c, err
	}
	if err := strictOperatorJSON(e.Raw, &c); err != nil {
		return e, c, err
	}
	if err := validateOperatorContext(c, s.RunID, l); err != nil {
		return e, c, err
	}
	inventories := make(map[string]*OperatorInventory)
	reasons := make(map[string]string)
	for i := range c.Groups {
		g := &c.Groups[i]
		g.InventoryReason = "expected inventory unavailable"
		if g.InventorySource == nil {
			continue
		}
		if reason, seen := reasons[g.InventorySource.SHA256]; seen {
			g.Inventory, g.InventoryReason = inventories[g.InventorySource.SHA256], reason
			continue
		}
		reasons[g.InventorySource.SHA256] = g.InventoryReason
		raw, ok := c.Artifacts[g.InventorySource.SHA256]
		if !ok {
			continue
		}
		var inventory OperatorInventory
		if err := strictOperatorJSON(raw, &inventory); err != nil || inventory.Version != 1 || inventory.RunID != c.RunID {
			g.InventoryReason = "expected inventory schema or run unavailable"
			reasons[g.InventorySource.SHA256] = g.InventoryReason
			continue
		}
		if err := validateOperatorInventory(inventory, l); err != nil {
			g.InventoryReason = "expected inventory invalid or exceeds bounds"
			reasons[g.InventorySource.SHA256] = g.InventoryReason
			continue
		}
		g.Inventory = &inventory
		g.InventoryReason = ""
		inventories[g.InventorySource.SHA256] = &inventory
		reasons[g.InventorySource.SHA256] = ""
	}
	return e, c, nil
}

// WriteOperatorContext writes validated, bounded version1 application metadata.
// Derived inventory fields are omitted; readers recover them from retained raw
// bytes. A successful write is serialization, not framework/native evidence.
func WriteOperatorContext(w io.Writer, c OperatorContext, l WorkloadLimits) error {
	l, err := workloadLimits(l)
	if err != nil {
		return err
	}
	if w == nil {
		return errors.New("missing operator context writer")
	}
	if err := validateOperatorContext(c, c.RunID, l); err != nil {
		return err
	}
	// Validation bounds aggregate encoded metadata before this allocation.
	raw, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("operator context byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		return io.ErrShortWrite
	}
	return err
}

func strictOperatorJSON(raw []byte, dst any) error {
	if err := decodeWorkloadJSON(raw, dst); err != nil {
		return fmt.Errorf("operator metadata: %w", err)
	}
	if err := operatorPresence(raw); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return fmt.Errorf("operator metadata: %w", err)
	}
	return nil
}

// A null integer decodes to zero in encoding/json. Inspect only the new wire
// fields where that would turn unknown shapes/loss into measured zero. JSON
// syntax, duplicate keys and numeric range remain owned by existing decoders.
func operatorPresence(raw []byte) error {
	type member struct {
		Shapes []json.RawMessage `json:"shapes"`
	}
	var presence struct {
		Members []member `json:"members"`
		Groups  []struct {
			Members []member `json:"members"`
		} `json:"groups"`
		Receipt struct {
			Losses map[string]json.RawMessage `json:"lost_records_by_family"`
		} `json:"receipt"`
	}
	if err := json.Unmarshal(raw, &presence); err != nil {
		return err
	}
	check := func(members []member) error {
		for _, m := range members {
			for _, shape := range m.Shapes {
				if bytes.Equal(bytes.TrimSpace(shape), []byte("null")) {
					return errors.New("operator shape is null; omit unavailable shapes")
				}
				var dimensions []json.RawMessage
				if err := json.Unmarshal(shape, &dimensions); err != nil {
					return err
				}
				for _, dimension := range dimensions {
					if bytes.Equal(bytes.TrimSpace(dimension), []byte("null")) {
						return errors.New("operator shape dimension is null; use -1 for unknown")
					}
				}
			}
		}
		return nil
	}
	if err := check(presence.Members); err != nil {
		return err
	}
	for _, g := range presence.Groups {
		if err := check(g.Members); err != nil {
			return err
		}
	}
	for _, rawLoss := range presence.Receipt.Losses {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(rawLoss, &fields); err != nil {
			return err
		}
		for _, name := range []string{"dropped", "total"} {
			value, ok := fields[name]
			if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return errors.New("operator receipt loss requires explicit dropped and total counts")
			}
		}
	}
	return nil
}

func operatorHash(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size && s == strings.ToLower(s)
}

func validateOperatorInventory(c OperatorInventory, l WorkloadLimits) error {
	if err := operatorMetadataBudget(OperatorContext{RunID: c.RunID, Groups: []OperatorGroup{{Binding: c.Binding, Members: c.Members}}}, l); err != nil {
		return err
	}
	if len(c.Members) > l.MaxMetrics || c.Members == nil {
		return errors.New("invalid operator inventory member count")
	}
	seen := map[string]bool{}
	for _, m := range c.Members {
		key := m.Source.SHA256 + ":" + m.ID
		if m.ID == "" || len(m.ID) > l.MaxStringBytes || !operatorHash(m.Source.SHA256) || m.Source.Locator == "" || seen[key] {
			return errors.New("invalid or duplicate inventory member")
		}
		seen[key] = true
	}
	return nil
}

// Account for encoded string escaping and structural overhead before encoding
// a caller-created value. The fixed interchange has no recursive types. Derived
// fields (json:"-") are not input and do not contribute to encoded size.
func operatorMetadataBudget(c OperatorContext, l WorkloadLimits) error {
	remaining := l.MaxBytes
	charge := func(n int64) error {
		if n > remaining {
			return errors.New("operator metadata byte limit exceeded")
		}
		remaining -= n
		return nil
	}
	var walk func(reflect.Value) error
	walk = func(v reflect.Value) error {
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				return charge(5)
			}
			return walk(v.Elem())
		}
		switch v.Kind() {
		case reflect.String:
			if v.Len() > l.MaxStringBytes {
				return errors.New("operator metadata string limit exceeded")
			}
			return charge(int64(v.Len())*6 + 3)
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				field := v.Type().Field(i)
				if field.Tag.Get("json") == "-" {
					continue
				}
				if err := charge(int64(len(field.Name))*6 + 4); err != nil {
					return err
				}
				if err := walk(v.Field(i)); err != nil {
					return err
				}
			}
		case reflect.Slice:
			if v.Type().Elem().Kind() == reflect.Uint8 {
				return charge((int64(v.Len())+2)/3*4 + 3)
			}
			if v.Len() > l.MaxMetrics {
				return errors.New("operator metadata collection limit exceeded")
			}
			for i := 0; i < v.Len(); i++ {
				if err := walk(v.Index(i)); err != nil {
					return err
				}
			}
		case reflect.Map:
			if v.Len() > l.MaxMetrics {
				return errors.New("operator metadata collection limit exceeded")
			}
			it := v.MapRange()
			for it.Next() {
				if err := walk(it.Key()); err != nil {
					return err
				}
				if err := walk(it.Value()); err != nil {
					return err
				}
			}
		}
		return charge(24)
	}
	return walk(reflect.ValueOf(c))
}

func validateOperatorContext(c OperatorContext, run string, l WorkloadLimits) error {
	if c.Version != 1 || c.RunID == "" || c.RunID != run || c.Producer == "" || c.ProducerVersion == "" || c.ProducerSchema == "" || c.Groups == nil || len(c.Groups) > l.MaxMetrics || len(c.Annotations) > l.MaxMetrics || len(c.Artifacts) > l.MaxMetrics || len(c.Batch.Records) != 0 {
		return errors.New("invalid operator context version, identity or count")
	}
	if c.SourceClass != "application" && c.SourceClass != "synthetic" && c.SourceClass != "imported" {
		return errors.New("unsupported operator metadata source class")
	}
	if c.Batch.Run != c.RunID {
		return errors.New("operator context batch run mismatch")
	}
	if c.Batch.MaxRecords < 0 || c.Batch.MaxRecords > 65536 {
		return errors.New("invalid operator activity record limit")
	}
	if _, err := c.Receipt.JSON(); err != nil {
		return fmt.Errorf("operator context receipt: %w", err)
	}
	// Bound all caller metadata before JSON allocates its encoded form. This
	// counts every string and raw byte, including nested existing identities.
	if err := operatorMetadataBudget(c, l); err != nil {
		return err
	}
	for hash, raw := range c.Artifacts {
		sum := sha256.Sum256(raw)
		if !operatorHash(hash) || hex.EncodeToString(sum[:]) != hash {
			return errors.New("operator raw artifact hash mismatch")
		}
	}
	groups := map[string]bool{}
	members := 0
	for _, g := range c.Groups {
		if g.ID == "" || groups[g.ID] || g.Binding.OpID == 0 || g.Binding.Process.PID == 0 || g.Binding.Process.Generation == 0 {
			return errors.New("invalid or duplicate operator group identity")
		}
		groups[g.ID] = true
		if len(g.Members) > l.MaxMetrics-members {
			return errors.New("operator member count limit exceeded")
		}
		members += len(g.Members)
		seen := map[string]bool{}
		for _, m := range g.Members {
			key := m.Source.SHA256 + ":" + m.ID
			if m.ID == "" || seen[key] || !operatorHash(m.Source.SHA256) || m.Source.Locator == "" || len(m.Shapes) > 32 {
				return errors.New("invalid or duplicate operator member")
			}
			seen[key] = true
			for _, shape := range m.Shapes {
				if len(shape) > 32 {
					return errors.New("operator shape rank limit exceeded")
				}
				for _, dim := range shape {
					if dim < -1 {
						return errors.New("invalid operator shape dimension")
					}
				}
			}
		}
		if g.InventorySource != nil && (!operatorHash(g.InventorySource.SHA256) || g.InventorySource.Locator == "") {
			return errors.New("invalid operator inventory reference")
		}
		if g.Binding.Code != nil {
			if err := g.Binding.Code.Validate(); err != nil {
				return err
			}
		}
	}
	return nil
}
