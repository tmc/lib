package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testNSYSOptions() nsysOptions {
	return nsysOptions{release: "2024.2", build: "synthetic-fixture", clock: "report-relative", timeout: 5 * time.Second, inputBytes: 4 << 20, bytes: 4 << 20, rows: 1000}
}
func testNSYSArgs() []string {
	return []string{"-nsys-release", "2024.2", "-nsys-build", "synthetic-fixture", "-nsys-clock", "report-relative"}
}

func TestNSYSCanonicalContract(t *testing.T) {
	for _, release := range []string{"2024.2", "2025.3"} {
		t.Run(release, func(t *testing.T) {
			schema, err := os.ReadFile("../../../../nvidia/x/nsightsystems/testdata/schema-" + release + ".sql")
			if err != nil {
				t.Fatal(err)
			}
			version := map[string]string{"2024.2": "3.11.0", "2025.3": "3.20.2"}[release]
			path := databaseSchema(t, string(schema)+strings.Replace(fixtureSQL, "3.11.0", version, 1))
			original, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			digest := sha256.Sum256(original)
			o := testNSYSOptions()
			o.release = release
			o.expectedSHA = hex.EncodeToString(digest[:])
			o.shift = 123
			var log bytes.Buffer
			rows, err := readNSYS(context.Background(), path, wholeCapture(), 500000, o, sqliteCommand, &log)
			if err != nil {
				t.Fatal(err)
			}
			if len(rows) != 4 || rows[0].start != 1000000 {
				t.Fatalf("shift applied twice or rows lost: %+v", rows)
			}
			for _, row := range rows {
				if row.args["basis"] != "external:nsys" || row.args["source_report_sha256"] != o.expectedSHA || row.args["snapshot_sha256"] != o.expectedSHA || row.args["schema_version"] != version || row.args["source_rowid"] == nil || row.args["source_raw_json"] == nil || row.args["capture_completeness"] != "unestablished" {
					t.Fatalf("missing source custody: %+v", row.args)
				}
			}
			if !strings.Contains(log.String(), "non-interval records omitted: 1") {
				t.Fatal(log.String())
			}
			after, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(original, after) {
				t.Fatal("source database changed", err)
			}
		})
	}
}

func TestNSYSRefusals(t *testing.T) {
	path := database(t, fixtureSQL)
	tests := []struct {
		name   string
		change func(*nsysOptions)
		path   string
	}{
		{"release", func(o *nsysOptions) { o.release = "2026.1" }, path},
		{"build", func(o *nsysOptions) { o.build = "" }, path},
		{"clock", func(o *nsysOptions) { o.clock = "nsys_ns" }, path},
		{"digest", func(o *nsysOptions) { o.expectedSHA = strings.Repeat("a", 64) }, path},
		{"input bytes", func(o *nsysOptions) { o.inputBytes = 16 }, path},
		{"rows", func(o *nsysOptions) { o.rows = 2 }, path},
		{"bytes", func(o *nsysOptions) { o.bytes = 128 }, path},
		{"timeout", func(o *nsysOptions) { o.timeout = -1 }, path},
		{"schema metadata", func(o *nsysOptions) {}, database(t, fixtureSQL+"UPDATE META_DATA_EXPORT SET value='3.20.2';")},
		{"column schema", func(o *nsysOptions) {}, database(t, fixtureSQL+"ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN unexpected INTEGER;")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := testNSYSOptions()
			tt.change(&o)
			if _, err := readNSYS(context.Background(), tt.path, wholeCapture(), 500000, o, sqliteCommand, io.Discard); err == nil {
				t.Fatal("accepted invalid import")
			}
		})
	}
	t.Run("journal", func(t *testing.T) {
		p := database(t, fixtureSQL)
		if err := os.WriteFile(p+"-wal", nil, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := readNSYS(context.Background(), p, wholeCapture(), 500000, testNSYSOptions(), sqliteCommand, io.Discard); err == nil {
			t.Fatal("accepted active WAL")
		}
	})
	t.Run("clock merge", func(t *testing.T) {
		args := append(testNSYSArgs(), "-nsys", path, "-spans", "missing", "-spans-offset-ns", "0", "-nsys-clock", "unestablished")
		var out bytes.Buffer
		if err := run(args, &out, io.Discard); err == nil || out.Len() != 0 {
			t.Fatal("accepted unestablished merge", err)
		}
	})
}

func TestNSYSWindowProvenance(t *testing.T) {
	w := wholeCapture()
	if err := w.Set("2000000:"); err != nil {
		t.Fatal(err)
	}
	var log bytes.Buffer
	rows, err := readNSYS(context.Background(), database(t, fixtureSQL), w, 500000, testNSYSOptions(), sqliteCommand, &log)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0].args["table_full_rows"] != int64(2) || rows[0].args["table_selected_rows"] != int64(1) || rows[0].args["source_rowid"] != int64(2) {
		t.Fatalf("window identity: %+v", rows)
	}
	if !strings.Contains(log.String(), "CUPTI_ACTIVITY_KIND_KERNEL available=true full_rows=2 selected_rows=1") || !strings.Contains(log.String(), "completeness unestablished") {
		t.Fatal(log.String())
	}
}

func TestNSYSAllCanonicalEvents(t *testing.T) {
	schema, err := os.ReadFile("../../../../nvidia/x/nsightsystems/testdata/schema-2024.2.sql")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("../../../../nvidia/x/nsightsystems/testdata/rows-2024.2.sql")
	if err != nil {
		t.Fatal(err)
	}
	rows, err := readNSYS(context.Background(), databaseSchema(t, string(schema)+string(data)), wholeCapture(), 500000, testNSYSOptions(), sqliteCommand, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	categories := map[string]bool{}
	for _, r := range rows {
		categories[r.category] = true
	}
	for _, want := range []string{"cuda_kernel", "cuda_memcpy", "cuda_memset", "cuda_runtime", "osrt"} {
		if !categories[want] {
			t.Errorf("missing %s", want)
		}
	}
}

func TestNSYSPathIsData(t *testing.T) {
	original := database(t, fixtureSQL)
	path := filepath.Join(t.TempDir(), "-capture; .shell not-a-command.sqlite")
	b, err := os.ReadFile(original)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readNSYS(context.Background(), path, wholeCapture(), 500000, testNSYSOptions(), sqliteCommand, io.Discard); err != nil {
		t.Fatal(err)
	}
}

func TestNSYSProviderAndDiagnosticContract(t *testing.T) {
	t.Run("missing strings remains unavailable", func(t *testing.T) {
		var log bytes.Buffer
		rows, err := readNSYS(context.Background(), database(t, fixtureSQL+"DROP TABLE StringIds;"), wholeCapture(), 500000, testNSYSOptions(), sqliteCommand, &log)
		if err != nil {
			t.Fatal(err)
		}
		if len(rows) != 4 || !strings.Contains(log.String(), "StringIds available=false") {
			t.Fatalf("provider availability lost: %s", &log)
		}
	})
	for _, sql := range []string{
		"CREATE TABLE DIAGNOSTIC_EVENT(timestamp INTEGER,severity INTEGER,message TEXT);",
		"CREATE TABLE DIAGNOSTIC_EVENT(timestamp INTEGER,severity INTEGER,text TEXT); INSERT INTO DIAGNOSTIC_EVENT VALUES(1,NULL,'lost');",
		"CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(code INTEGER,name TEXT,label TEXT);",
	} {
		t.Run(sql, func(t *testing.T) {
			if _, err := readNSYS(context.Background(), database(t, fixtureSQL+sql), wholeCapture(), 500000, testNSYSOptions(), sqliteCommand, io.Discard); err == nil {
				t.Fatal("accepted malformed diagnostic provider")
			}
		})
	}
}

func TestNSYSDiagnosticWriteFailure(t *testing.T) {
	_, err := readNSYS(context.Background(), database(t, fixtureSQL), wholeCapture(), 500000, testNSYSOptions(), sqliteCommand, errorWriter{})
	if !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("lost diagnostic writer error: %v", err)
	}
}

func TestNSYSNonregularInput(t *testing.T) {
	called := false
	process := func(context.Context, string, []string, io.Writer, io.Writer) error { called = true; return nil }
	_, err := readNSYS(context.Background(), t.TempDir(), wholeCapture(), 500000, testNSYSOptions(), process, io.Discard)
	if err == nil || called {
		t.Fatalf("non-regular input admitted: %v process=%t", err, called)
	}
}
