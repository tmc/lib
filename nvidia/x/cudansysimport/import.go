package cudansysimport

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// ErrUnavailable is returned by Reader for a table absent from the export.
var ErrUnavailable = errors.New("nsys: table unavailable")

// Column is a documented SQLite column. Nullable distinguishes an unavailable
// field from an integer zero. Order is the declared table order.
type Column struct {
	Name     string
	Type     string
	Nullable bool
}

// Table contains declared columns and exact JSON row projections of an export.
// Rows must include every declared column, using null for absent values.
type Table struct {
	// DatabaseSHA256 identifies the stable database snapshot used by Reader.
	// Reader must verify it against the actual source, not repeat an expected hash.
	DatabaseSHA256 string
	Columns        []Column
	Rows           []json.RawMessage
}

// Reader supplies bounded, read-only SQLite table access. It must return
// ErrUnavailable only for a missing table, not a query/tool failure. It must
// honor ctx, bound subprocess lifetime/output, and return at most limit+1 rows
// so a truncated table cannot be mistaken for a complete population.
// All calls must refer to one stable, read-only database snapshot.
type Reader interface {
	ReadTable(ctx context.Context, name string, limit int) (Table, error)
}

// Source preserves caller-declared export identities and clock choices.
// ToolVersion is the named supported release, not an inferred database version.
// ToolBuild identifies the actual exporter build. Clock is report-relative,
// unix-epoch or unestablished; ShiftNS preserves the explicit --ts-shift value.
// SHA256 values identify the original report and the imported database. They
// are not proof that a caller's declared exporter or clock choice is correct.
type Source struct {
	ToolVersion    string
	ToolBuild      string
	ReportPath     string
	ReportSHA256   string
	DatabaseSHA256 string
	Clock          string
	ShiftNS        int64
}

// Limits bounds total returned row bytes and row count across all tables.
// Zero fields select 16 MiB and 65536 rows; maxima are 64 MiB and 262144.
type Limits struct{ Bytes, Rows int }

// Provider records whether a supported table exists. Available with zero Rows
// means an empty exported table, not an observed absence of activity.
type Provider struct {
	Table     string
	Available bool
	Rows      int
	Reason    string
}

// Event retains one exported row. StartNS/EndNS use the source's clock domain;
// nil EndNS can be a marker or an open range and is not replaced with StartNS.
// Process retains the serialized target/process part of GlobalId. Raw retains
// all documented fields, including geometry, payload and enum values.
type Event struct {
	Table           string
	StartNS         int64
	EndNS           *int64
	Name            string
	NameUnavailable bool
	Correlation     *int64
	Process         *uint64
	Bytes           *int64
	Raw             json.RawMessage
}

// Join is a unique within-report CUDA API/activity correlation. It makes no
// performance or cross-report causal claim.
type Join struct{ Event, API int }

// Unjoined explains why a device activity has no unique API correlation.
type Unjoined struct {
	Event  int
	Reason string
}

// Report is imported evidence with basis external:nsys. It is never a local
// measurement or a certificate of whole-capture completion. SessionStartUTCNS
// retains a documented capture metadata value; it does not establish that
// exported timestamps were normalized to that epoch.
type Report struct {
	Source               Source
	SchemaVersion        string
	SelectedSchemaSHA256 string
	Basis                string
	SessionStartUTCNS    *int64
	Providers            []Provider
	Events               []Event
	Joins                []Join
	Unjoined             []Unjoined
	Metadata             []json.RawMessage
}

// Import reads the supported tables, checks their pinned column schemas and
// required row types, resolves string IDs and makes only unique same-process
// correlations within this report. Missing tables remain explicit providers.
func Import(ctx context.Context, reader Reader, source Source, limits Limits) (*Report, error) {
	if ctx == nil || reader == nil {
		return nil, errors.New("nsys: context and reader required")
	}
	if err := source.valid(); err != nil {
		return nil, err
	}
	if limits.Bytes == 0 {
		limits.Bytes = 16 << 20
	}
	if limits.Rows == 0 {
		limits.Rows = 65536
	}
	if limits.Bytes < 1 || limits.Bytes > 64<<20 || limits.Rows < 1 || limits.Rows > 262144 {
		return nil, errors.New("nsys: invalid limits")
	}
	version := map[string]string{"2024.2": "3.11.0", "2025.3": "3.20.2"}[source.ToolVersion]
	if version == "" {
		return nil, errors.New("nsys: unsupported tool release")
	}
	r := &Report{Source: source, SchemaVersion: version, SelectedSchemaSHA256: schemaSHA256[version], Basis: "external:nsys"}
	tables := make(map[string]Table)
	rowsLeft, bytesLeft := limits.Rows, limits.Bytes
	for _, name := range tableNames {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		t, err := reader.ReadTable(ctx, name, rowsLeft)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if errors.Is(err, ErrUnavailable) {
			r.Providers = append(r.Providers, Provider{Table: name, Reason: "table absent from export"})
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("nsys read %s: %w", name, err)
		}
		if len(t.Rows) > rowsLeft {
			return nil, errors.New("nsys: row limit exceeded")
		}
		if t.DatabaseSHA256 != source.DatabaseSHA256 {
			return nil, errors.New("nsys: database snapshot identity differs")
		}
		if err := checkColumns(t.Columns, schemas[version][name]); err != nil {
			return nil, fmt.Errorf("nsys %s: %w", name, err)
		}
		for _, row := range t.Rows {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if len(row) > bytesLeft {
				return nil, errors.New("nsys: byte limit exceeded")
			}
			bytesLeft -= len(row)
			if _, err := checkRow(row, t.Columns); err != nil {
				return nil, fmt.Errorf("nsys %s: %w", name, err)
			}
		}
		rowsLeft -= len(t.Rows)
		tables[name] = t
		r.Providers = append(r.Providers, Provider{Table: name, Available: true, Rows: len(t.Rows)})
	}
	meta, ok := tables["META_DATA_EXPORT"]
	if !ok {
		return nil, errors.New("nsys: schema metadata unavailable")
	}
	seen := make(map[string]bool)
	actual := ""
	for _, row := range meta.Rows {
		m, _ := checkRow(row, meta.Columns)
		var name string
		json.Unmarshal(m["name"], &name)
		if name == "" || seen[name] {
			return nil, errors.New("nsys: duplicate or empty export metadata")
		}
		seen[name] = true
		if name == "EXPORT_SCHEMA_VERSION" {
			json.Unmarshal(m["value"], &actual)
		}
		r.Metadata = append(r.Metadata, append(json.RawMessage(nil), row...))
	}
	if actual != version {
		return nil, fmt.Errorf("nsys: schema version %q does not match %q", actual, version)
	}
	stringsByID := make(map[int64]string)
	stringTable, stringsAvailable := tables["StringIds"]
	for _, row := range stringTable.Rows {
		m, _ := checkRow(row, stringTable.Columns)
		id, _ := integer(m["id"])
		var value string
		json.Unmarshal(m["value"], &value)
		if _, ok := stringsByID[id]; ok {
			return nil, errors.New("nsys: ambiguous string ID")
		}
		stringsByID[id] = value
	}
	if session, ok := tables["TARGET_INFO_SESSION_START_TIME"]; ok {
		if len(session.Rows) > 1 {
			return nil, errors.New("nsys: ambiguous session clock metadata")
		}
		if len(session.Rows) == 1 {
			m, _ := checkRow(session.Rows[0], session.Columns)
			r.SessionStartUTCNS = optionalInt(m, "utcEpochNs")
		}
	}
	for _, name := range tableNames[3:] {
		t, ok := tables[name]
		if !ok {
			continue
		}
		for _, row := range t.Rows {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			m, _ := checkRow(row, t.Columns)
			start, _ := integer(m["start"])
			end := optionalInt(m, "end")
			if end != nil && *end < start {
				return nil, errors.New("nsys: event end precedes start")
			}
			e := Event{Table: name, StartNS: start, EndNS: end, Correlation: optionalInt(m, "correlationId"), Bytes: optionalInt(m, "bytes"), Raw: append(json.RawMessage(nil), row...)}
			if e.Bytes != nil && *e.Bytes < 0 {
				return nil, errors.New("nsys: negative byte count")
			}
			global := optionalInt(m, "globalPid")
			if global == nil {
				global = optionalInt(m, "globalTid")
			}
			if global != nil {
				if *global < 0 {
					return nil, errors.New("nsys: negative global ID")
				}
				p := uint64(*global) &^ uint64(0xffffff)
				e.Process = &p
			}
			nameKey := "nameId"
			if name == "CUPTI_ACTIVITY_KIND_KERNEL" {
				nameKey = "shortName"
			}
			if name == "NVTX_EVENTS" {
				nameKey = "textId"
				if raw := m["text"]; len(raw) > 0 && string(raw) != "null" {
					json.Unmarshal(raw, &e.Name)
				}
			}
			if id := optionalInt(m, nameKey); id != nil && e.Name == "" {
				if !stringsAvailable {
					e.NameUnavailable = true
				} else {
					value, ok := stringsByID[*id]
					if !ok {
						return nil, errors.New("nsys: missing referenced string ID")
					}
					e.Name = value
				}
			}
			r.Events = append(r.Events, e)
		}
	}
	r.correlate()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return r, nil
}

func (s Source) valid() error {
	if s.ToolBuild == "" || s.ReportPath == "" {
		return errors.New("nsys: tool build and report path required")
	}
	for _, h := range []string{s.ReportSHA256, s.DatabaseSHA256} {
		b, err := hex.DecodeString(h)
		if err != nil || len(b) != sha256.Size || strings.Trim(h, "0") == "" || strings.ToLower(h) != h {
			return errors.New("nsys: exact nonzero SHA256 identities required")
		}
	}
	if s.Clock != "report-relative" && s.Clock != "unix-epoch" && s.Clock != "unestablished" {
		return errors.New("nsys: explicit timestamp basis required")
	}
	return nil
}

func (r *Report) correlate() {
	type key struct {
		process uint64
		id      int64
	}
	apis := make(map[key][]int)
	for i, e := range r.Events {
		if e.Table == "CUPTI_ACTIVITY_KIND_RUNTIME" && e.Process != nil && e.Correlation != nil {
			k := key{*e.Process, *e.Correlation}
			apis[k] = append(apis[k], i)
		}
	}
	for i, e := range r.Events {
		if e.Table != "CUPTI_ACTIVITY_KIND_KERNEL" && e.Table != "CUPTI_ACTIVITY_KIND_MEMCPY" && e.Table != "CUPTI_ACTIVITY_KIND_MEMSET" {
			continue
		}
		reason := "correlation or process unavailable"
		if e.Process != nil && e.Correlation != nil {
			matches := apis[key{*e.Process, *e.Correlation}]
			switch len(matches) {
			case 1:
				r.Joins = append(r.Joins, Join{Event: i, API: matches[0]})
				continue
			case 0:
				reason = "API correlation unavailable"
			default:
				reason = "ambiguous API correlation"
			}
		}
		r.Unjoined = append(r.Unjoined, Unjoined{Event: i, Reason: reason})
	}
}

func checkColumns(got, want []Column) error {
	if len(got) != len(want) || len(want) == 0 {
		return errors.New("unsupported column population")
	}
	for i, c := range got {
		if c != want[i] {
			return errors.New("unsupported column schema")
		}
	}
	return nil
}

func checkRow(raw json.RawMessage, cols []Column) (map[string]json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := uniqueRow(raw); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &m); err != nil || len(m) != len(cols) {
		return nil, errors.New("invalid row population")
	}
	for _, c := range cols {
		v, ok := m[c.Name]
		if !ok {
			return nil, errors.New("missing exact row column")
		}
		if string(v) == "null" {
			if !c.Nullable {
				return nil, errors.New("null required column")
			}
			continue
		}
		switch c.Type {
		case "INTEGER":
			if _, err := integer(v); err != nil {
				return nil, err
			}
		case "TEXT":
			var s string
			if err := json.Unmarshal(v, &s); err != nil {
				return nil, errors.New("invalid text column")
			}
		case "REAL":
			var f float64
			if err := json.Unmarshal(v, &f); err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, errors.New("invalid real column")
			}
		default:
			return nil, errors.New("unsupported column type")
		}
	}
	return m, nil
}

func integer(raw json.RawMessage) (int64, error) {
	var n int64
	if err := json.Unmarshal(raw, &n); err != nil {
		return 0, errors.New("invalid integer column")
	}
	return n, nil
}
func optionalInt(m map[string]json.RawMessage, key string) *int64 {
	raw, ok := m[key]
	if !ok || string(raw) == "null" {
		return nil
	}
	n, _ := integer(raw)
	return &n
}
