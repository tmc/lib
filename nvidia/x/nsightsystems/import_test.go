package nsightsystems

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type snapshot map[string]Table

func (s snapshot) ReadTable(ctx context.Context, name string, limit int) (Table, error) {
	if err := ctx.Err(); err != nil {
		return Table{}, err
	}
	t, ok := s[name]
	if !ok {
		return Table{}, ErrUnavailable
	}
	return t, nil
}

func testSource(version string) Source {
	return Source{ToolVersion: version, ToolBuild: version + "-documented-fixture", ReportPath: "synthetic.nsys-rep", ReportSHA256: strings.Repeat("a", 64), DatabaseSHA256: strings.Repeat("b", 64), Clock: "report-relative", ShiftNS: 17}
}

func makeRow(cols []Column, values map[string]any) json.RawMessage {
	m := make(map[string]any)
	for _, c := range cols {
		if c.Nullable {
			m[c.Name] = nil
		} else if c.Type == "TEXT" {
			m[c.Name] = ""
		} else {
			m[c.Name] = 0
		}
	}
	for k, v := range values {
		m[k] = v
	}
	b, err := json.Marshal(m)
	if err != nil {
		panic(err)
	}
	return b
}

func testSnapshot(version string) snapshot {
	s := snapshot{}
	for _, name := range tableNames {
		s[name] = Table{DatabaseSHA256: strings.Repeat("b", 64), Columns: append([]Column(nil), schemas[version][name]...)}
	}
	add := func(name string, values map[string]any) {
		t := s[name]
		t.Rows = append(t.Rows, makeRow(t.Columns, values))
		s[name] = t
	}
	add("META_DATA_EXPORT", map[string]any{"name": "EXPORT_SCHEMA_VERSION", "value": version})
	add("TARGET_INFO_SESSION_START_TIME", map[string]any{"utcEpochNs": 1700000000000000000})
	for id, name := range map[int]string{1: "kernel<float>", 2: "cudaLaunchKernel", 3: "pthread_mutex_lock"} {
		add("StringIds", map[string]any{"id": id, "value": name})
	}
	process := uint64(2)<<48 | 55<<24
	add("CUPTI_ACTIVITY_KIND_RUNTIME", map[string]any{"start": 90, "end": 95, "nameId": 2, "correlationId": 7, "globalTid": process | 1})
	add("CUPTI_ACTIVITY_KIND_KERNEL", map[string]any{"start": 100, "end": 130, "shortName": 1, "demangledName": 1, "correlationId": 7, "globalPid": process})
	add("CUPTI_ACTIVITY_KIND_MEMCPY", map[string]any{"start": 140, "end": 150, "bytes": 64, "correlationId": 8, "globalPid": process})
	add("CUPTI_ACTIVITY_KIND_MEMSET", map[string]any{"start": 160, "end": 170, "bytes": 128})
	add("NVTX_EVENTS", map[string]any{"start": 80, "text": "nccl:AllReduce", "jsonText": "{\"synthetic\":true}", "globalTid": process | 1})
	add("OSRT_API", map[string]any{"start": 85, "end": 90, "nameId": 3, "globalTid": process | 1})
	return s
}

func TestKnownVersionsAndUnits(t *testing.T) {
	for tool, version := range map[string]string{"2024.2": "3.11.0", "2025.3": "3.20.2"} {
		t.Run(tool, func(t *testing.T) {
			s := testSnapshot(version)
			r, err := Import(context.Background(), s, testSource(tool), Limits{})
			if err != nil {
				t.Fatal(err)
			}
			if r.Basis != "external:nsys" || r.SchemaVersion != version || r.Source != testSource(tool) || len(r.Events) != 6 || len(r.Joins) != 1 || len(r.Unjoined) != 2 {
				t.Fatalf("identity/population %+v", r)
			}
			kernel := r.Events[0]
			if kernel.StartNS != 100 || kernel.EndNS == nil || *kernel.EndNS != 130 || kernel.Name != "kernel<float>" {
				t.Fatalf("nanoseconds/name %+v", kernel)
			}
			if *r.Events[1].Bytes != 64 || *r.Events[2].Bytes != 128 || r.Events[3].Name != "cudaLaunchKernel" || r.Events[5].Name != "pthread_mutex_lock" {
				t.Fatal("units or API identities changed")
			}
			if r.Events[4].EndNS != nil || r.Events[4].Name != "nccl:AllReduce" || !strings.Contains(string(r.Events[4].Raw), "synthetic") {
				t.Fatal("NVTX marker/payload erased")
			}
			if r.SessionStartUTCNS == nil || *r.SessionStartUTCNS != 1700000000000000000 || r.Joins[0] != (Join{Event: 0, API: 3}) {
				t.Fatal("clock or process correlation changed")
			}
			if !reflect.DeepEqual(s, testSnapshotOrdered(s, version)) {
				t.Fatal("input mutated")
			}
		})
	}
}

// Preserve each original row's order; map iteration of constructed StringIds is
// deliberately immaterial to importer semantics.
func testSnapshotOrdered(original snapshot, version string) snapshot {
	s := testSnapshot(version)
	s["StringIds"] = original["StringIds"]
	return s
}

func TestMissingProvidersAndCorrelationDomains(t *testing.T) {
	s := testSnapshot("3.11.0")
	delete(s, "StringIds")
	delete(s, "OSRT_API")
	r, err := Import(context.Background(), s, testSource("2024.2"), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Events[0].NameUnavailable || len(r.Events) != 5 || r.Providers[8].Available || r.Providers[8].Reason == "" {
		t.Fatal("missing provider synthesized")
	}
	s = testSnapshot("3.11.0")
	table := s["CUPTI_ACTIVITY_KIND_RUNTIME"]
	table.Rows = append(table.Rows, append(json.RawMessage(nil), table.Rows[0]...))
	s["CUPTI_ACTIVITY_KIND_RUNTIME"] = table
	r, err = Import(context.Background(), s, testSource("2024.2"), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Joins) != 0 || r.Unjoined[0].Reason != "ambiguous API correlation" {
		t.Fatal("ambiguous API guessed")
	}
	// Matching correlation IDs from another serialized target/process must not join.
	table.Rows = table.Rows[:1]
	var row map[string]any
	json.Unmarshal(table.Rows[0], &row)
	row["globalTid"] = uint64(3)<<48 | 55<<24 | 1
	table.Rows[0], _ = json.Marshal(row)
	s["CUPTI_ACTIVITY_KIND_RUNTIME"] = table
	r, err = Import(context.Background(), s, testSource("2024.2"), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Joins) != 0 || r.Unjoined[0].Reason != "API correlation unavailable" {
		t.Fatal("cross-target join")
	}
}

func TestRefusals(t *testing.T) {
	tests := []struct {
		name   string
		change func(snapshot, *Source)
	}{
		{"unknown tool", func(s snapshot, id *Source) { id.ToolVersion = "2026.5" }},
		{"unknown schema", func(s snapshot, id *Source) {
			table := s["META_DATA_EXPORT"]
			table.Rows[0] = makeRow(table.Columns, map[string]any{"name": "EXPORT_SCHEMA_VERSION", "value": "3.99.0"})
			s["META_DATA_EXPORT"] = table
		}},
		{"missing metadata", func(s snapshot, id *Source) { delete(s, "META_DATA_EXPORT") }},
		{"duplicate metadata", func(s snapshot, id *Source) {
			table := s["META_DATA_EXPORT"]
			table.Rows = append(table.Rows, table.Rows[0])
			s["META_DATA_EXPORT"] = table
		}},
		{"unknown clock", func(s snapshot, id *Source) { id.Clock = "local-cupti" }},
		{"bad digest", func(s snapshot, id *Source) { id.DatabaseSHA256 = "00" }},
		{"stale snapshot", func(s snapshot, id *Source) { id.DatabaseSHA256 = strings.Repeat("c", 64) }},
		{"table snapshot drift", func(s snapshot, id *Source) {
			table := s["OSRT_API"]
			table.DatabaseSHA256 = strings.Repeat("c", 64)
			s["OSRT_API"] = table
		}},
		{"zero report", func(s snapshot, id *Source) { id.ReportSHA256 = strings.Repeat("0", 64) }},
		{"missing build", func(s snapshot, id *Source) { id.ToolBuild = "" }},
		{"column schema", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Columns = append([]Column(nil), table.Columns...)
			table.Columns[0].Type = "TEXT"
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"nullable schema", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Columns = append([]Column(nil), table.Columns...)
			table.Columns[0].Nullable = true
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"column order", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Columns = append([]Column(nil), table.Columns...)
			table.Columns[0], table.Columns[1] = table.Columns[1], table.Columns[0]
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"duplicate string", func(s snapshot, id *Source) {
			table := s["StringIds"]
			table.Rows = append(table.Rows, table.Rows[0])
			s["StringIds"] = table
		}},
		{"missing string", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Rows[0] = makeRow(table.Columns, map[string]any{"shortName": 999, "demangledName": 1})
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"truncated", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Rows[0] = table.Rows[0][:len(table.Rows[0])-1]
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"float integer", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Rows[0] = json.RawMessage(strings.Replace(string(table.Rows[0]), `"start":100`, `"start":100.0`, 1))
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"null required", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Rows[0] = makeRow(table.Columns, map[string]any{"start": nil})
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"end before start", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Rows[0] = makeRow(table.Columns, map[string]any{"start": 200, "end": 100, "shortName": 1})
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"duplicate key", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Rows[0] = append([]byte(`{"start":100,`), table.Rows[0][1:]...)
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"alias key", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
			table.Rows[0] = append([]byte(`{"START":100,`), table.Rows[0][1:]...)
			s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		}},
		{"negative bytes", func(s snapshot, id *Source) {
			table := s["CUPTI_ACTIVITY_KIND_MEMCPY"]
			table.Rows[0] = makeRow(table.Columns, map[string]any{"bytes": -1})
			s["CUPTI_ACTIVITY_KIND_MEMCPY"] = table
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := testSnapshot("3.11.0")
			id := testSource("2024.2")
			test.change(s, &id)
			if _, err := Import(context.Background(), s, id, Limits{}); err == nil {
				t.Fatal("invalid import accepted")
			}
		})
	}
}

func TestBoundsCancellationAndRawIsolation(t *testing.T) {
	s := testSnapshot("3.11.0")
	for _, l := range []Limits{{Bytes: 1}, {Rows: 1}, {Bytes: -1}, {Rows: 262145}} {
		if _, err := Import(context.Background(), s, testSource("2024.2"), l); err == nil {
			t.Fatal("bound ignored")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Import(ctx, s, testSource("2024.2"), Limits{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := Import(nil, s, testSource("2024.2"), Limits{}); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := Import(context.Background(), nil, testSource("2024.2"), Limits{}); err == nil {
		t.Fatal("nil reader accepted")
	}
	r, err := Import(context.Background(), s, testSource("2024.2"), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	r.Events[0].Raw[0] = '!'
	if s["CUPTI_ACTIVITY_KIND_KERNEL"].Rows[0][0] != '{' {
		t.Fatal("raw aliases reader")
	}
}

type failingReader struct{}

type cancelingReader struct {
	snapshot
	cancel context.CancelFunc
	absent bool
}

func (r cancelingReader) ReadTable(ctx context.Context, name string, limit int) (Table, error) {
	t, err := r.snapshot.ReadTable(ctx, name, limit)
	if name == "OSRT_API" {
		r.cancel()
		if r.absent {
			return Table{}, ErrUnavailable
		}
	}
	return t, err
}

func TestCancellationDuringFinalRead(t *testing.T) {
	for _, test := range []struct {
		name   string
		absent bool
	}{
		{"present", false},
		{"absent", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := cancelingReader{snapshot: testSnapshot("3.11.0"), cancel: cancel, absent: test.absent}
			report, err := Import(ctx, r, testSource("2024.2"), Limits{})
			if !errors.Is(err, context.Canceled) || report != nil {
				t.Fatalf("canceled import returned report: %v, error: %v", report, err)
			}
		})
	}
}

func (failingReader) ReadTable(context.Context, string, int) (Table, error) {
	return Table{}, errors.New("sqlite tool failed")
}

func TestToolFailureIsNotUnavailable(t *testing.T) {
	if _, err := Import(context.Background(), failingReader{}, testSource("2024.2"), Limits{}); err == nil || !strings.Contains(err.Error(), "sqlite tool failed") {
		t.Fatalf("tool failure hidden: %v", err)
	}
}

func FuzzRow(f *testing.F) {
	s := testSnapshot("3.11.0")
	f.Add([]byte(s["CUPTI_ACTIVITY_KIND_KERNEL"].Rows[0]))
	f.Add([]byte(`{}`))
	f.Add([]byte(`null`))
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) > 32<<10 {
			t.Skip()
		}
		s := testSnapshot("3.11.0")
		table := s["CUPTI_ACTIVITY_KIND_KERNEL"]
		table.Rows[0] = raw
		s["CUPTI_ACTIVITY_KIND_KERNEL"] = table
		r, err := Import(context.Background(), s, testSource("2024.2"), Limits{Bytes: 64 << 10, Rows: 32})
		if err == nil && (r.Basis != "external:nsys" || len(r.Events) != 6 || string(r.Events[0].Raw) != string(raw)) {
			t.Fatal("accepted row identity differs")
		}
	})
}

func TestOfficialSQLiteSchemaBodies(t *testing.T) {
	tool, err := exec.LookPath("sqlite3")
	if err != nil {
		t.Skip("explicit sqlite3 schema control unavailable")
	}
	for release, version := range map[string]string{"2024.2": "3.11.0", "2025.3": "3.20.2"} {
		t.Run(release, func(t *testing.T) {
			sql, err := os.ReadFile("testdata/schema-" + release + ".sql")
			if err != nil {
				t.Fatal(err)
			}
			db := filepath.Join(t.TempDir(), "synthetic.sqlite")
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, tool, "-batch", "-init", "/dev/null", db)
			cmd.Stdin = strings.NewReader(string(sql))
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("documented schema: %v %s", err, output)
			}
			for _, name := range tableNames {
				cmd := exec.CommandContext(ctx, tool, "-readonly", "-batch", "-init", "/dev/null", "-json", db, `PRAGMA table_info("`+name+`");`)
				out, err := cmd.Output()
				if err != nil {
					t.Fatal(err)
				}
				var rows []struct {
					Name    string `json:"name"`
					Type    string `json:"type"`
					NotNull int    `json:"notnull"`
				}
				if err := json.Unmarshal(out, &rows); err != nil {
					t.Fatal(err)
				}
				cols := []Column{}
				for _, row := range rows {
					cols = append(cols, Column{Name: row.Name, Type: row.Type, Nullable: row.NotNull == 0})
				}
				if err := checkColumns(cols, schemas[version][name]); err != nil {
					t.Fatalf("actual sqlite columns %s: %v", name, err)
				}
			}
		})
	}
}
